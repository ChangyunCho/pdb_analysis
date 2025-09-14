import argparse
from pathlib import Path
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import write as seqio_write
from Bio.Data import IUPACData

AA_MAP_BASE = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
AA_MAP_EXTRA = {
    "MSE": "M",  # Selenomethionine
    "SEC": "U",  # Selenocysteine
    "PYL": "O",  # Pyrrolysine
}
AA_MAP_BASE.update(AA_MAP_EXTRA)

NUC_MAP = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DI": "I",
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "I",
    "ADE": "A", "CYT": "C", "GUA": "G", "URI": "U", "THY": "T", "PSU": "U",
}

def aa_three_to_one(name):
    return AA_MAP_BASE.get(name.strip().upper(), "X")

def nuc_to_one(name):
    key = name.strip().upper()
    if key in NUC_MAP:
        return NUC_MAP[key]
    if len(key) == 2 and key[0] == "D" and key[1] in {"A","C","G","T","I"}:
        return key[1]
    if key in {"A","C","G","T","U","I"}:
        return key
    return "N"

def is_standard_protein_res(res):
    hetflag, _, _ = res.get_id()
    name = res.get_resname().strip().upper()
    return hetflag == " " and (name in AA_MAP_BASE)

def is_nucleic_res(res):
    hetflag, _, _ = res.get_id()
    name = res.get_resname().strip().upper()
    if hetflag != " ":
        return False
    if name in NUC_MAP:
        return True
    if len(name) == 2 and name[0] == "D" and name[1] in {"A","C","G","T","I"}:
        return True
    if name in {"A","C","G","T","U","I"}:
        return True
    return False

def extract_protein_sequences(structure, pdb_id):
    ppb = PPBuilder()
    records = []
    for model in structure:
        for chain in model:
            chain_id = chain.id
            peptides = ppb.build_peptides(chain, aa_only=True)
            if peptides:
                for i, pep in enumerate(peptides, start=1):
                    residues = list(pep)
                    seq = "".join(aa_three_to_one(r.get_resname()) for r in residues)
                    if seq:
                        records.append(
                            SeqRecord(Seq(seq),
                                      id=f"{pdb_id}|chain:{chain_id}|poly:{i}",
                                      description="protein")
                        )
            else:
                residues = [r for r in chain if is_standard_protein_res(r)]
                if residues:
                    seq = "".join(aa_three_to_one(r.get_resname()) for r in residues)
                    if seq:
                        records.append(
                            SeqRecord(Seq(seq),
                                      id=f"{pdb_id}|chain:{chain_id}|poly:1",
                                      description="protein(manual)")
                        )
    return records

def extract_nucleic_sequences(structure, pdb_id):
    records = []
    for model in structure:
        for chain in model:
            chain_id = chain.id
            nuc_res = [r for r in chain if is_nucleic_res(r)]
            if not nuc_res:
                continue
            seq = "".join(nuc_to_one(r.get_resname()) for r in nuc_res)
            if seq:
                records.append(
                    SeqRecord(Seq(seq),
                              id=f"{pdb_id}|chain:{chain_id}",
                              description="nucleic")
                )
    return records

def derive_pdb_id(p):
    name = p.name
    for sfx in (".gz", ".bz2", ".xz", ".pdb", ".cif", ".mmcif", ".ent"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
    for sfx in (".pdb", ".cif", ".mmcif", ".ent"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
    return name or "input"

def main():
    ap = argparse.ArgumentParser(description="Extract protein/nucleic sequences from a PDB into FASTA.")
    ap.add_argument("pdb", help="Path to PDB file")
    ap.add_argument("--only", choices=["protein", "nucleic", "both"], default="both",
                    help="Extract only protein, only nucleic, or both (default: both)")
    ap.add_argument("--out", help="Output FASTA path (default: stdout)")
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    pdb_id = derive_pdb_id(pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))

    records = []
    if args.only in ("protein", "both"):
        records.extend(extract_protein_sequences(structure, pdb_id))
    if args.only in ("nucleic", "both"):
        records.extend(extract_nucleic_sequences(structure, pdb_id))

    if not records:
        print("# No sequences detected (protein/nucleic). Check PDB contents.")
        return

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            seqio_write(records, f, "fasta")
        print(f"Wrote FASTA to {args.out}")
    else:
        seqio_write(records, None, "fasta")

if __name__ == "__main__":
    main()
