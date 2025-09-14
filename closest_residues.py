import argparse, csv, sys, math, os
from pathlib import Path
from collections import defaultdict, namedtuple
import numpy as np
from Bio.PDB import PDBParser, Polypeptide

ResidueKey = namedtuple("ResidueKey", ["chain", "resname", "resseq", "icode"])

WATER_NAMES = {"HOH", "WAT", "H2O"}

def is_water(res):
    return res.get_resname().strip() in WATER_NAMES

def is_protein_or_nucleic(res):
    # Standard residues have hetflag ' ' (blank); includes protein and nucleic acid residues in most PDBs
    return res.get_id()[0] == ' '

def is_ligand(res):
    # HET groups that are not water
    return res.get_id()[0] != ' ' and not is_water(res)

def altloc_ok(atom):
    # Keep primary conformer and unlabelled altloc
    alt = atom.get_altloc()
    return (not atom.is_disordered()) or (alt in (' ', 'A'))

def residue_key(res):
    chain_id = res.get_parent().id
    resname = res.get_resname().strip()
    hetflag, resseq, icode = res.get_id()
    icode = icode.strip() if isinstance(icode, str) else icode
    return ResidueKey(chain_id, resname, resseq, icode or '')

def describe_key(k: ResidueKey):
    rn = f"{k.resname}:{k.chain}:{k.resseq}{k.icode or ''}"
    return rn

def pick_auto_ligand(structure):
    # Return the ligand residue (HET, non-water) with the most atoms
    best = None
    best_count = -1
    for model in structure:
        for chain in model:
            for res in chain:
                if is_ligand(res):
                    atoms = [a for a in res.get_atoms() if altloc_ok(a)]
                    if len(atoms) > best_count:
                        best = res
                        best_count = len(atoms)
    return best

def find_res_by_token(structure, token):
    """
    token format: RESNAME:CHAIN:RESSEQ[ICODE]
    Examples: "HEM:A:401", "ADP:B:12A", "LIG:C:7"
    """
    try:
        resname, chain_id, restail = token.split(":")
    except ValueError:
        raise ValueError("Invalid --target-res format. Use RESNAME:CHAIN:RESSEQ[ICODE], e.g. LIG:A:101")
    resname = resname.strip()
    chain_id = chain_id.strip()
    # separate resseq and optional icode (letter)
    resseq_str = restail.strip()
    if resseq_str and resseq_str[-1].isalpha():
        icode = resseq_str[-1]
        resseq = int(resseq_str[:-1])
    else:
        resseq = int(resseq_str)
        icode = ' '
    for model in structure:
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        for res in chain:
            hetflag, rseq, ric = res.get_id()
            if res.get_resname().strip() == resname and rseq == resseq and (ric.strip() or ' ') == (icode.strip() or ' '):
                return res
    return None

def atoms_of_residue(res):
    return [a for a in res.get_atoms() if altloc_ok(a)]

def gather_search_residues(structure, include_waters=False):
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if is_protein_or_nucleic(res) or (include_waters and is_water(res)):
                    residues.append(res)
    return residues

def build_atom_table(residues):
    coords = []
    res_indices = []
    res_keys = []
    key_to_index = {}
    # map residue -> row index in res_min
    for idx, res in enumerate(residues):
        rk = residue_key(res)
        res_keys.append(rk)
        key_to_index[rk] = idx
        for atom in atoms_of_residue(res):
            coords.append(atom.coord)  # numpy array
            res_indices.append(idx)
    if coords:
        coords = np.vstack(coords)
        res_indices = np.asarray(res_indices, dtype=np.int32)
    else:
        coords = np.zeros((0,3), dtype=np.float64)
        res_indices = np.zeros((0,), dtype=np.int32)
    return coords, res_indices, res_keys

def target_atom_coords_from_residue(res):
    return np.vstack([a.coord for a in atoms_of_residue(res)]) if res else np.zeros((0,3))

def compute_min_distances(atom_coords, atom_res_idx, target_coords, n_res):
    # Initialize all residues with +inf min distance
    res_min = np.full((n_res,), np.inf, dtype=np.float64)
    if len(atom_coords) == 0 or len(target_coords) == 0:
        return res_min
    # For each target atom, update per-residue minima using numpy's reduction
    for t in target_coords:
        dists = np.linalg.norm(atom_coords - t, axis=1)
        # in-place groupwise min: res_min[res_idx] = min(res_min[res_idx], dists)
        np.minimum.at(res_min, atom_res_idx, dists)
    return res_min

def _derive_pdb_id(pdb_path: str) -> str:
    """Best-effort PDB id from filename, stripping common extensions."""
    name = Path(pdb_path).name
    # Strip multiple known suffixes if present (e.g., .pdb.gz)
    for sfx in (".gz", ".bz2", ".xz", ".pdb", ".cif", ".mmcif", ".ent"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
    # In case of double extensions like .pdb.gz, run once more
    for sfx in (".pdb", ".cif", ".mmcif", ".ent"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
    return name

def main():
    ap = argparse.ArgumentParser(description="Find residues closest to a ligand or point in a PDB.")
    ap.add_argument("pdb", help="Path to PDB file")
    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--target-res", help="Target as RESNAME:CHAIN:RESSEQ[ICODE], e.g. LIG:A:101")
    tgt.add_argument("--point", nargs=3, type=float, metavar=("X","Y","Z"), help="Target as 3D coordinate")
    ap.add_argument("--auto-ligand", action="store_true", help="Auto-pick largest non-water HET as target (default if nothing else given)")
    ap.add_argument("--top", type=int, default=20, help="Return top N residues (default: 20)")
    ap.add_argument("--cutoff", type=float, default=None, help="Optional Å cutoff; filter residues beyond this distance")
    ap.add_argument("--include-water", action="store_true", help="Include waters in the residue list")
    args = ap.parse_args()

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("input", args.pdb)
    except Exception as e:
        print(f"Failed to parse PDB: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine target coordinates
    target_coords = None
    target_label = None
    if args.point is not None:
        target_coords = np.array([args.point], dtype=np.float64)
        target_label = f"point({args.point[0]}, {args.point[1]}, {args.point[2]})"
    elif args.target_res:
        res = find_res_by_token(structure, args.target_res)
        if res is None:
            print(f"Could not find target residue {args.target_res}", file=sys.stderr)
            sys.exit(2)
        tc = target_atom_coords_from_residue(res)
        if tc.size == 0:
            print(f"Target residue {args.target_res} has no atoms (after altloc filtering).", file=sys.stderr)
            sys.exit(3)
        target_coords = tc
        target_label = f"residue {args.target_res}"
    else:
        # auto-ligand fallback
        res = pick_auto_ligand(structure) if (args.auto_ligand or True) else None
        if res is None:
            print("No ligand found (non-water HET). Provide --target-res or --point.", file=sys.stderr)
            sys.exit(4)
        tc = target_atom_coords_from_residue(res)
        target_coords = tc
        rk = residue_key(res)
        target_label = f"ligand {describe_key(rk)} (auto)"

    # Build searchable atom table for protein/nucleic (and optional waters)
    residues = gather_search_residues(structure, include_waters=args.include_water)
    if not residues:
        print("No residues to search (protein/nucleic) found.", file=sys.stderr)
        sys.exit(5)
    atom_coords, atom_res_idx, res_keys = build_atom_table(residues)
    res_min = compute_min_distances(atom_coords, atom_res_idx, target_coords, n_res=len(res_keys))

    # Rank and filter
    order = np.argsort(res_min)
    rows = []
    for rank, idx in enumerate(order, start=1):
        dist = float(res_min[idx])
        if math.isinf(dist):  # unreachable (no atoms)
            continue
        if args.cutoff is not None and dist > args.cutoff:
            continue
        k = res_keys[idx]
        rows.append((rank, k.chain, k.resname, f"{k.resseq}{k.icode or ''}", dist))
        if args.top is not None and len(rows) >= args.top:
            break

    # Prepare output text
    header_lines = []
    header_lines.append(f"# Closest residues to {target_label}")
    header_lines.append(f"{'Rank':>4}  {'Chain':<5} {'ResName':<6} {'ResID':<6} {'MinDist(Å)':>10}")
    body_lines = [f"{rank:>4}  {chain:<5} {resname:<6} {resid:<6} {dist:>10.3f}"
                  for rank, chain, resname, resid, dist in rows]
    output_text = "\n".join(header_lines + body_lines) + ("\n" if body_lines else "\n(no hits)\n")

    # Console print (optional)
    print(output_text, end="")

    # Ensure ./results and write file named {PDB_id}_{top}.txt
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = _derive_pdb_id(args.pdb)
    out_path = results_dir / f"{pdb_id}_{args.top}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
