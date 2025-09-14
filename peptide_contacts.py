import argparse, sys, math, re, os
from pathlib import Path
import numpy as np
from collections import defaultdict
from Bio.PDB import PDBParser

WATERS = {"HOH","WAT","H2O"}
POS_RES = {"LYS","ARG","HIP","HSP"}  
NEG_RES = {"ASP","GLU"}           
METALS  = {"ZN","MG","CA","NA","K","FE","MN","CO","CU","NI"}
HBD_ATOMS = {"N","O","S"}        
HBA_ATOMS = {"N","O","S"}      

def alt_ok(atom):
    return (not atom.is_disordered()) or atom.get_altloc() in (" ","A")

def element_of(atom):
    e = (atom.element or "").upper()
    if not e:
        name = atom.get_name().strip().upper()
        e = ''.join([c for c in name if c.isalpha()])[:2]
    return e

def is_water(res):
    return res.get_resname().strip().upper() in WATERS

def is_protein_res(res):
    het,_,_ = res.get_id()
    return het == " "

def is_ligand_res(res):
    het,_,_ = res.get_id()
    return het != " " and not is_water(res)

def residue_label(res):
    chain = res.get_parent().id
    het, seq, icode = res.get_id()
    return f"{res.get_resname().strip()}:{chain}:{seq}{(icode or '').strip()}"

def atom_array(res):
    return [a for a in res.get_atoms() if alt_ok(a)]

def min_distance(resA, resB):
    dmin = np.inf; pair = None
    As = atom_array(resA); Bs = atom_array(resB)
    for a in As:
        for b in Bs:
            d = np.linalg.norm(a.coord - b.coord)
            if d < dmin:
                dmin = d; pair = (a, b)
    return dmin, pair

def classify_contacts(res1, res2):
    """res1-res2 간 상호작용 유형 분류 (대칭)."""
    contacts = set()
    hydrophobic = False
    for a in atom_array(res1):
        if element_of(a) != "C": continue
        for b in atom_array(res2):
            if element_of(b) != "C": continue
            if np.linalg.norm(a.coord - b.coord) <= 4.5:
                hydrophobic = True; break
        if hydrophobic: break
    if hydrophobic: contacts.add("hydrophobic")

    def is_hbd(atom): return element_of(atom) in HBD_ATOMS
    def is_hba(atom): return element_of(atom) in HBA_ATOMS
    hbond = False
    for a in atom_array(res1):
        for b in atom_array(res2):
            d = np.linalg.norm(a.coord - b.coord)
            if d <= 3.6:
                if (is_hbd(a) and is_hba(b)) or (is_hbd(b) and is_hba(a)):
                    hbond = True; break
        if hbond: break
    if hbond: contacts.add("hbond")

    r1 = res1.get_resname().strip().upper()
    r2 = res2.get_resname().strip().upper()
    pos1 = r1 in POS_RES; neg1 = r1 in NEG_RES
    pos2 = r2 in POS_RES; neg2 = r2 in NEG_RES

    has_pos1 = pos1 or any(element_of(a) == "N" for a in atom_array(res1))
    has_neg1 = neg1 or any(element_of(a) == "O" for a in atom_array(res1))
    has_pos2 = pos2 or any(element_of(a) == "N" for a in atom_array(res2))
    has_neg2 = neg2 or any(element_of(a) == "O" for a in atom_array(res2))
    salt = False
    if (has_pos1 and has_neg2) or (has_pos2 and has_neg1):
        d, _ = min_distance(res1, res2)
        if d <= 4.0: salt = True
    if salt: contacts.add("salt_bridge")

    def is_metal_atom(atom): return element_of(atom) in METALS
    def is_coord_atom(atom): return element_of(atom) in {"N","O","S"}
    metal = False
    for a in atom_array(res1):
        for b in atom_array(res2):
            d = np.linalg.norm(a.coord - b.coord)
            if (is_metal_atom(a) and is_coord_atom(b) and d <= 3.0) or (is_metal_atom(b) and is_coord_atom(a) and d <= 3.0):
                metal = True; break
        if metal: break
    if metal: contacts.add("metal_coord")

    return contacts

def parse_set_spec(structure, spec):
    spec = (spec or "").strip()
    residues = []
    model = list(structure)[0]
    if spec.lower() in ("protein", "all_protein"):
        for chain in model:
            for r in chain:
                if is_protein_res(r) and not is_water(r):
                    residues.append(r)
        return residues

    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    for tok in tokens:
        m_chain = re.fullmatch(r"(?:chain:)?([A-Za-z0-9])", tok)
        m_range = re.fullmatch(r"(?:chain:)?([A-Za-z0-9]):(\d+)-(\d+)", tok)
        m_res   = re.fullmatch(r"([A-Za-z0-9]{1,3}):([A-Za-z0-9]):(\d+)([A-Za-z]?)", tok)
        if m_range:
            ch, a, b = m_range.groups()
            a = int(a); b = int(b)
            if ch not in [c.id for c in model]: continue
            chain = model[ch]
            for r in chain:
                het, resseq, icode = r.get_id()
                if resseq >= a and resseq <= b and not is_water(r):
                    residues.append(r)
        elif m_chain:
            ch = m_chain.group(1)
            if ch not in [c.id for c in model]: continue
            chain = model[ch]
            for r in chain:
                if not is_water(r):
                    residues.append(r)
        elif m_res:
            resname, ch, seq, icode = m_res.groups()
            resname = resname.upper()
            seq = int(seq)
            icode = (icode or " ").strip() or " "
            if ch not in [c.id for c in model]: continue
            chain = model[ch]
            for r in chain:
                het, rseq, ric = r.get_id()
                if r.get_resname().strip().upper() == resname and rseq == seq and (ric.strip() or " ") == icode:
                    residues.append(r); break
        else:
            print(f"[WARN] Unrecognized token in set spec: '{tok}'", file=sys.stderr)
    return residues

def save_csv_matrix(D, labels, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for lab, row in zip(labels, D):
            w.writerow([lab] + [f"{x:.3f}" if np.isfinite(x) else "" for x in row])

def save_png_contact(D, cutoff, path, title="Contact map"):
    import matplotlib.pyplot as plt
    C = (D <= cutoff).astype(float)
    C = np.nan_to_num(C, nan=0.0)
    plt.figure(figsize=(6,5))
    plt.imshow(C, interpolation="nearest")
    plt.title(f"{title} (cutoff={cutoff} Å)")
    plt.xlabel("Residue (B)")
    plt.ylabel("Residue (A)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Peptide–peptide / ligand–protein contacts & distances")
    ap.add_argument("pdb", help="PDB file")
    ap.add_argument("--setA", help="Selection for Set A (e.g., 'A', 'A:5-30', 'ADP:B:123', or comma-joined)")
    ap.add_argument("--setB", help="Selection for Set B")
    ap.add_argument("--auto-ligand", action="store_true", help="If setA not given, pick largest non-water HET as A")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Cutoff for reporting contacts (Å)")
    ap.add_argument("--top", type=int, default=100, help="Report top-N closest residue pairs")
    ap.add_argument("--csv", help="Save A×B distance matrix as CSV")
    ap.add_argument("--png", help="Save contact map PNG (A×B, uses --cutoff)")
    ap.add_argument("--txt", help="Save text summary (top contacts & types)")
    ap.add_argument("--results-dir", default="./results", help="Output directory")
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("input", args.pdb)
    except Exception as e:
        print(f"Failed to parse PDB: {e}", file=sys.stderr); sys.exit(1)

    if args.setA:
        setA = parse_set_spec(structure, args.setA)
        if not setA:
            print("Set A selection returned no residues.", file=sys.stderr); sys.exit(2)
        setA_label = args.setA
    elif args.auto_ligand:
        best = None; best_atoms = -1
        for model in structure:
            for chain in model:
                for r in chain:
                    if is_ligand_res(r):
                        atoms = [a for a in r.get_atoms() if alt_ok(a)]
                        if len(atoms) > best_atoms:
                            best = r; best_atoms = len(atoms)
        if best is None:
            print("No ligand (non-water HET) found for set A.", file=sys.stderr); sys.exit(3)
        setA = [best]
        setA_label = f"ligand {residue_label(best)}"
    else:
        print("Please provide --setA (or use --auto-ligand).", file=sys.stderr); sys.exit(4)
        
    if args.setB:
        setB = parse_set_spec(structure, args.setB)
        if not setB:
            print("Set B selection returned no residues.", file=sys.stderr); sys.exit(5)
        setB_label = args.setB
    else:
        setB = parse_set_spec(structure, "protein")
        setB_label = "protein"

    labelsA = [residue_label(r) for r in setA]
    labelsB = [residue_label(r) for r in setB]

    nA, nB = len(setA), len(setB)
    D = np.zeros((nA, nB), dtype=np.float64)
    pairs = []
    for i, ra in enumerate(setA):
        for j, rb in enumerate(setB):
            d, _ = min_distance(ra, rb)
            D[i, j] = d
            pairs.append((d, i, j))

    pairs.sort(key=lambda x: x[0])
    lines = []
    header = f"# Contacts between A=({setA_label}) and B=({setB_label})  cutoff={args.cutoff} Å"
    lines.append(header)
    lines.append(f"{'Rank':>4}  {'A_residue':<18} {'B_residue':<18} {'dmin(Å)':>8}  {'types'}")

    n_reported = 0
    for rank, (d, i, j) in enumerate(pairs, start=1):
        if not np.isfinite(d) or d > args.cutoff:
            continue
        ra, rb = setA[i], setB[j]
        kinds = classify_contacts(ra, rb)
        kind_str = ",".join(sorted(kinds)) if kinds else "-"
        lines.append(f"{rank:>4}  {labelsA[i]:<18} {labelsB[j]:<18} {d:>8.3f}  {kind_str}")
        n_reported += 1
        if n_reported >= args.top:
            break

    report = "\n".join(lines) + ("\n" if n_reported else "\n(no contacts within cutoff)\n")
    print(report, end="")

    # 저장
    pdb_id = Path(args.pdb).name
    for sfx in (".gz",".bz2",".xz",".pdb",".cif",".mmcif",".ent"):
        if pdb_id.endswith(sfx): pdb_id = pdb_id[:-len(sfx)]

    if args.csv:
        save_csv_matrix(D, labelsB, args.csv)
        print(f"Saved CSV matrix -> {args.csv}")
    if args.png:
        title = f"A({setA_label}) vs B({setB_label})"
        save_png_contact(D, args.cutoff, args.png, title=title)
        print(f"Saved contact map PNG -> {args.png}")
    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report -> {args.txt}")
    else:
        out_txt = results_dir / f"{pdb_id}_A-B_contacts.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report -> {out_txt}")

if __name__ == "__main__":
    main()
