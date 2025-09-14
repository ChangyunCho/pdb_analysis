import argparse, sys, math
from pathlib import Path
import numpy as np
from collections import defaultdict, namedtuple
from Bio.PDB import PDBParser

WATERS = {"HOH","WAT","H2O"}
POS_RES = {"LYS","ARG","HIP","HSP"}   # positive charge
NEG_RES = {"ASP","GLU"}               # negative charge
METALS = {"ZN","MG","CA","NA","K","FE","MN","CO","CU","NI"} 
HBD_ATOMS = {"N","O","S"}             # donor heavy
HBA_ATOMS = {"N","O","S"}             # acceptor heavy

def alt_ok(atom):
    return (not atom.is_disordered()) or atom.get_altloc() in (" ","A")

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
    het,seq,icode = res.get_id()
    return f"{res.get_resname().strip()}:{chain}:{seq}{(icode or '').strip()}"

def pick_auto_ligand(structure):
    best=None; best_atoms=-1
    for model in structure:
        for chain in model:
            for res in chain:
                if is_ligand_res(res):
                    atoms=[a for a in res.get_atoms() if alt_ok(a)]
                    if len(atoms)>best_atoms:
                        best=res; best_atoms=len(atoms)
    return best

def find_res_by_token(structure, token):
    resname, chain, tail = token.split(":")
    resname=resname.strip().upper(); chain=chain.strip()
    if tail and tail[-1].isalpha():
        icode=tail[-1]; seq=int(tail[:-1])
    else:
        icode=" "; seq=int(tail)
    for model in structure:
        if chain not in [c.id for c in model]: continue
        ch=model[chain]
        for r in ch:
            het, rseq, ric = r.get_id()
            if r.get_resname().strip().upper()==resname and rseq==seq and (ric.strip() or " ")==(icode.strip() or " "):
                return r
    return None

def atom_array(res):
    return [a for a in res.get_atoms() if alt_ok(a)]

def min_distance(resA, resB):
    dmin=np.inf; pair=None
    As=atom_array(resA); Bs=atom_array(resB)
    for a in As:
        for b in Bs:
            d=np.linalg.norm(a.coord - b.coord)
            if d<dmin:
                dmin=d; pair=(a,b)
    return dmin, pair

def within_cutoff(resA, resB, cutoff):
    d,_=min_distance(resA,resB)
    return d<=cutoff

def element_of(atom):
    e=(atom.element or "").upper()
    if not e:  # fallback from atom name
        name=atom.get_name().strip().upper()
        if name.startswith(("C","N","O","S","P","H","MG","ZN","FE","CA","MN","CO","CU","NI","NA","K")):
            e=''.join([c for c in name if c.isalpha()])[:2]
    return e

def classify_contacts(lig, res):
    contacts=set()
    for a in atom_array(lig):
        ea=element_of(a)
        if ea!="C": continue
        for b in atom_array(res):
            eb=element_of(b)
            if eb!="C": continue
            if np.linalg.norm(a.coord-b.coord) <= 4.5:
                contacts.add("hydrophobic"); break
        if "hydrophobic" in contacts: break

    def is_hbd(atom):  # donor heavy approx
        return element_of(atom) in HBD_ATOMS
    def is_hba(atom):  # acceptor heavy approx
        return element_of(atom) in HBA_ATOMS
    hbond=False
    for a in atom_array(lig):
        for b in atom_array(res):
            da=np.linalg.norm(a.coord-b.coord)
            if da<=3.6:
                if is_hbd(a) and is_hba(b):
                    hbond=True; break
                if is_hbd(b) and is_hba(a):
                    hbond=True; break
        if hbond: break
    if hbond: contacts.add("hbond")

    rname=res.get_resname().strip().upper()
    neg_in_lig=False; pos_in_lig=False
    for a in atom_array(lig):
        e=element_of(a)
        if e=="O": neg_in_lig=True
        if e=="N": pos_in_lig=True
    salt=False
    if rname in POS_RES and neg_in_lig:
        d,_=min_distance(lig,res)
        if d<=4.0: salt=True
    if rname in NEG_RES and pos_in_lig:
        d,_=min_distance(lig,res)
        if d<=4.0: salt=True
    if salt: contacts.add("salt_bridge")

    def is_metal_atom(atom):
        return element_of(atom) in METALS
    def is_coord_atom(atom):
        return element_of(atom) in {"N","O","S"}
    metal=False
    for a in atom_array(lig):
        for b in atom_array(res):
            ea=element_of(a); eb=element_of(b)
            d=np.linalg.norm(a.coord-b.coord)
            if is_metal_atom(a) and is_coord_atom(b) and d<=3.0: metal=True; break
            if is_metal_atom(b) and is_coord_atom(a) and d<=3.0: metal=True; break
        if metal: break
    if metal: contacts.add("metal_coord")

    return contacts

def main():
    ap=argparse.ArgumentParser(description="Ligand–protein interaction scanner")
    ap.add_argument("pdb", help="PDB file")
    tg=ap.add_mutually_exclusive_group()
    tg.add_argument("--target-res", help="RES:CHAIN:SEQ[ICODE], e.g., ADP:B:123 or CA:A:501")
    tg.add_argument("--auto", action="store_true", help="Auto pick largest non-water HET (default if none given)")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Neighbor residue cutoff (Å) for reporting (default 5.0)")
    ap.add_argument("--results-dir", default="./results", help="Output dir (default ./results)")
    args=ap.parse_args()

    parser=PDBParser(QUIET=True)
    try:
        structure=parser.get_structure("input", args.pdb)
    except Exception as e:
        print(f"Failed to parse PDB: {e}", file=sys.stderr); sys.exit(1)

    if args.target_res:
        lig=find_res_by_token(structure, args.target_res)
        if lig is None:
            print(f"Target residue not found: {args.target_res}", file=sys.stderr); sys.exit(2)
        lig_label=residue_label(lig)
    else:
        lig=pick_auto_ligand(structure)
        if lig is None:
            print("No ligand (non-water HET) found; specify --target-res", file=sys.stderr); sys.exit(3)
        lig_label=residue_label(lig)

    neighbors=[]
    for model in structure:
        for chain in model:
            for res in chain:
                if res is lig: continue
                if not is_protein_res(res): continue
                d,_=min_distance(lig,res)
                if d<=args.cutoff:
                    neighbors.append((d,res))
    neighbors.sort(key=lambda x:x[0])

    lines=[]
    header=f"# Interactions around {lig_label} (cutoff {args.cutoff} Å)"
    lines.append(header)
    lines.append(f"{'Rank':>4}  {'Residue':<15} {'dmin(Å)':>8}  {'types'}")
    for i,(d,res) in enumerate(neighbors, start=1):
        kinds=classify_contacts(lig,res)
        kinds_str=",".join(sorted(kinds)) if kinds else "-"
        lines.append(f"{i:>4}  {residue_label(res):<15} {d:>8.3f}  {kinds_str}")

    text="\n".join(lines)+"\n"
    print(text, end="")

    # Save
    outdir=Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)
    pdb_id=Path(args.pdb).name
    for sfx in (".gz",".bz2",".xz",".pdb",".cif",".mmcif",".ent"):
        if pdb_id.endswith(sfx): pdb_id=pdb_id[:-len(sfx)]
    outpath=outdir/f"{pdb_id}_interactions.txt"
    with open(outpath,"w",encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {outpath}")

if __name__=="__main__":
    main()
