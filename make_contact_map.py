import argparse
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

WATER_NAMES = {"HOH", "WAT", "H2O"}

def is_protein_res(res):
    hetflag, _, _ = res.get_id()
    name = res.get_resname().strip()
    return hetflag == " " and is_aa(name, standard=False)

def is_nucleic_res(res):
    hetflag, _, _ = res.get_id()
    if hetflag != " ":
        return False
    name = res.get_resname().strip().upper()
    return name in {"A","C","G","T","U","I","DA","DC","DG","DT","DI","ADE","CYT","GUA","URI","THY","PSU"}

def is_water(res):
    return res.get_resname().strip() in WATER_NAMES

def residue_label(res):
    chain_id = res.get_parent().id
    hetflag, resseq, icode = res.get_id()
    resname = res.get_resname().strip()
    icode = (icode or "").strip()
    return f"{resname}:{chain_id}:{resseq}{icode}"

def get_atom_coord(res, atom_name):
    if res is None:
        return None
    if atom_name in res:
        atom = res[atom_name]
        # altloc 처리: 우선순위 ' ' > 'A'
        if atom.is_disordered():
            alts = {a.get_altloc(): a for a in atom.disordered_get_list()}
            if " " in alts:
                atom = alts[" "]
            elif "A" in alts:
                atom = alts["A"]
            else:
                atom = list(alts.values())[0]
        return atom.coord.astype(np.float64)
    return None

def heavy_center(res):
    coords = []
    for atom in res.get_atoms():
        name = atom.element.strip().upper() if atom.element else ""
        if name != "H":
            coords.append(atom.coord)
    if coords:
        return np.mean(np.vstack(coords), axis=0).astype(np.float64)
    return None

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def rep_coord(res, mode="CA"):
    name = res.get_resname().strip().upper()
    if mode == "CA":
        return coalesce(get_atom_coord(res, "CA"), heavy_center(res))
    if mode == "CB":
        if name == "GLY":
            return coalesce(get_atom_coord(res, "CA"), heavy_center(res))
        return coalesce(get_atom_coord(res, "CB"), get_atom_coord(res, "CA"), heavy_center(res))
    if mode == "P":
        return coalesce(get_atom_coord(res, "P"), get_atom_coord(res, "C4'"), get_atom_coord(res, "C4*"), heavy_center(res))
    if mode == "COM":
        return heavy_center(res)
    # fallback
    return coalesce(get_atom_coord(res, "CA"), get_atom_coord(res, "P"), heavy_center(res))

def collect_residues(structure, only="both", chains=None, include_ligand=False):
    residues = []
    labels = []
    model = list(structure)[0]
    for chain in model:
        if chains and chain.id not in chains:
            continue
        for res in chain:
            if is_water(res):
                continue
            ok = False
            if only in ("both", "protein") and is_protein_res(res):
                ok = True
            if only in ("both", "nucleic") and is_nucleic_res(res):
                ok = True
            if include_ligand:
                hetflag, _, _ = res.get_id()
                if hetflag != " " and not is_water(res):
                    ok = True
            if ok:
                residues.append(res)
                labels.append(residue_label(res))
    return residues, labels

def compute_distance_matrix(residues, mode="CA"):
    n = len(residues)
    D = np.zeros((n, n), dtype=np.float64)
    if mode == "min":
        atoms_list = []
        for r in residues:
            atoms = []
            for a in r.get_atoms():
                el = (a.element or "").upper()
                if el != "H":
                    atoms.append(a.coord.astype(np.float64))
            if atoms:
                atoms_list.append(np.vstack(atoms))
            else:
                atoms_list.append(np.zeros((0, 3), dtype=np.float64))
        for i in range(n):
            Ai = atoms_list[i]
            for j in range(i + 1, n):
                Aj = atoms_list[j]
                if Ai.size == 0 or Aj.size == 0:
                    d = np.inf
                else:
                    dmin = np.inf
                    for a in Ai:
                        d = np.linalg.norm(Aj - a, axis=1).min()
                        if d < dmin:
                            dmin = d
                    d = dmin
                D[i, j] = D[j, i] = d if np.isfinite(d) else np.nan
        return D
    else:
        reps = []
        for r in residues:
            c = rep_coord(r, mode=mode)
            reps.append(None if c is None else np.asarray(c, dtype=np.float64))
        for i in range(n):
            ci = reps[i]
            for j in range(i + 1, n):
                cj = reps[j]
                if ci is None or cj is None:
                    d = np.nan
                else:
                    d = float(np.linalg.norm(ci - cj))
                D[i, j] = D[j, i] = d
        return D

def save_csv(D, labels, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for lab, row in zip(labels, D):
            w.writerow([lab] + [f"{x:.3f}" if np.isfinite(x) else "" for x in row])

def save_npz(D, labels, path):
    np.savez_compressed(path, distances=D, labels=np.array(labels, dtype=object))

def save_png_contact(D, cutoff, path):
    import matplotlib.pyplot as plt
    C = (D <= cutoff).astype(float)
    C = np.nan_to_num(C, nan=0.0)
    plt.figure(figsize=(6, 5))
    plt.imshow(C, interpolation="nearest")
    plt.title(f"Contact map (cutoff={cutoff} Å)")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Compute residue distance matrix and contact map from PDB.")
    ap.add_argument("pdb", help="Path to PDB file")
    ap.add_argument("--only", choices=["protein", "nucleic", "both"], default="both", help="Residue type to include (default: both)")
    ap.add_argument("--chains", nargs="*", help="Only include these chain IDs (e.g., A B H)")
    ap.add_argument("--include-ligand", action="store_true", help="Include HET groups (except waters)")
    ap.add_argument("--mode", choices=["CA", "CB", "P", "COM", "min"], default="CA", help="Representative coordinate: CA/CB/P/COM or minimum atom-atom distance (min)")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Contact cutoff in Å (default: 8.0)")
    ap.add_argument("--csv", help="Save distance matrix as CSV")
    ap.add_argument("--npz", help="Save distance matrix & labels as NPZ")
    ap.add_argument("--png", help="Save contact map PNG (uses --cutoff)")
    ap.add_argument("--labels", default="residues.txt", help="Save residue labels (default: residues.txt)")
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    residues, labels = collect_residues(
        structure,
        only=args.only,
        chains=set(args.chains) if args.chains else None,
        include_ligand=args.include_ligand
    )
    if not residues:
        print("No residues matched the selection.")
        return

    mode = args.mode
    if args.only == "nucleic" and args.mode == "CA":
        mode = "P"

    D = compute_distance_matrix(residues, mode=mode)

    if args.labels:
        with open(args.labels, "w", encoding="utf-8") as f:
            for i, lab in enumerate(labels):
                f.write(f"{i}\t{lab}\n")
        print(f"Saved labels -> {args.labels}")

    if args.csv:
        save_csv(D, labels, args.csv)
        print(f"Saved CSV -> {args.csv}")
    if args.npz:
        save_npz(D, labels, args.npz)
        print(f"Saved NPZ -> {args.npz}")
    if args.png:
        save_png_contact(D, args.cutoff, args.png)
        print(f"Saved contact map PNG -> {args.png}")

    # 콘솔 요약
    print(f"Residues: {len(residues)} | mode: {mode} | cutoff: {args.cutoff} Å")
    print(f"Distance matrix shape: {D.shape}")

if __name__ == "__main__":
    main()
