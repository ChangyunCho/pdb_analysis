#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_contact_map.py
- PDB에서 체인별 residue 대표 좌표를 뽑아 NxN 거리 행렬을 계산하고,
  cutoff 이하 contact를 1로 표시한 contact map을 생성합니다.

예시 사용:
  # 1) 기본(CA 기반) 거리행렬/컨택맵 생성 + PNG/CSV/NPZ 저장
  python make_contact_map.py 4ZFG.pdb --cutoff 8.0 --png contact.png --csv dist.csv --npz dist.npz

  # 2) 최소 원자-원자 거리 기반
  python make_contact_map.py complex.pdb --mode min --cutoff 5.0 --chains A B

  # 3) CB 기반 (Gly는 자동으로 CA 사용)
  python make_contact_map.py protein.pdb --mode CB --cutoff 8 --only protein

  # 4) 핵산만 분석 + 대표 원자 P (없으면 C4'로 대체)
  python make_contact_map.py nucleic.pdb --only nucleic --mode P

출력:
 - 거리 행렬: NxN (대칭, 대각선 0)
 - contact map: (옵션) cutoff 이하 True/1
 - 인덱스–잔기 라벨 파일: residues.txt
"""

import argparse
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

WATER_NAMES = {"HOH", "WAT", "H2O"}

# ---------------------- Utilities ----------------------
def is_protein_res(res):
    hetflag, _, _ = res.get_id()
    name = res.get_resname().strip()
    # 표준/수정 아미노산 모두 허용 (hetflag == " " 기준)
    return hetflag == " " and is_aa(name, standard=False)

def is_nucleic_res(res):
    hetflag, _, _ = res.get_id()
    if hetflag != " ":
        return False
    name = res.get_resname().strip().upper()
    # 단순 휴리스틱: DNA/RNA 잔기명
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
    """지정 원자 좌표 (altloc A/blank 우선). 없으면 None."""
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
    """무거운 원자(비수소) 중심(COM). 없으면 None."""
    coords = []
    for atom in res.get_atoms():
        name = atom.element.strip().upper() if atom.element else ""
        if name != "H":
            coords.append(atom.coord)
    if coords:
        return np.mean(np.vstack(coords), axis=0).astype(np.float64)
    return None

def coalesce(*vals):
    """앞에서부터 None이 아닌 첫 값을 반환 (모두 None이면 None)."""
    for v in vals:
        if v is not None:
            return v
    return None

def rep_coord(res, mode="CA"):
    """
    대표 좌표 선택:
      - mode=CA (protein 기본): CA
      - mode=CB: CB (GLY는 CA로 대체)
      - mode=P  (nucleic 기본): P, 없으면 C4'
      - mode=COM: heavy-atom center
    """
    name = res.get_resname().strip().upper()
    if mode == "CA":
        return coalesce(get_atom_coord(res, "CA"), heavy_center(res))
    if mode == "CB":
        if name == "GLY":
            return coalesce(get_atom_coord(res, "CA"), heavy_center(res))
        return coalesce(get_atom_coord(res, "CB"),
                        get_atom_coord(res, "CA"),
                        heavy_center(res))
    if mode == "P":
        return coalesce(get_atom_coord(res, "P"),
                        get_atom_coord(res, "C4'"),
                        get_atom_coord(res, "C4*"),
                        heavy_center(res))
    if mode == "COM":
        return heavy_center(res)
    # fallback
    return coalesce(get_atom_coord(res, "CA"),
                    get_atom_coord(res, "P"),
                    heavy_center(res))

def collect_residues(structure, only="both", chains=None, include_ligand=False):
    """모델 0 기준으로 필터링된 residue 리스트와 라벨 반환."""
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
                # HET group 포함 (물 제외)
                hetflag, _, _ = res.get_id()
                if hetflag != " " and not is_water(res):
                    ok = True
            if ok:
                residues.append(res)
                labels.append(residue_label(res))
    return residues, labels

# ---------------------- Core computations ----------------------
def compute_distance_matrix(residues, mode="CA"):
    """
    mode가 'min'이면 최소 원자-원자 거리 (느리지만 정확).
    그 외(CA/CB/P/COM)는 대표 좌표 간 유클리드 거리.
    """
    n = len(residues)
    D = np.zeros((n, n), dtype=np.float64)
    if mode == "min":
        # 최소 원자-원자 거리
        atoms_list = []
        for r in residues:
            atoms = []
            for a in r.get_atoms():
                el = (a.element or "").upper()
                if el != "H":  # 보통 H 제외
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
                    # 모든 쌍 거리의 최소 (메모리 절약 루프)
                    dmin = np.inf
                    for a in Ai:
                        d = np.linalg.norm(Aj - a, axis=1).min()
                        if d < dmin:
                            dmin = d
                    d = dmin
                D[i, j] = D[j, i] = d if np.isfinite(d) else np.nan
        return D
    else:
        # 대표 좌표 기반
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

# ---------------------- Saving helpers ----------------------
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
    # contact: True(1) if distance <= cutoff
    C = (D <= cutoff).astype(float)
    # NaN은 0 취급
    C = np.nan_to_num(C, nan=0.0)
    plt.figure(figsize=(6, 5))
    plt.imshow(C, interpolation="nearest")
    plt.title(f"Contact map (cutoff={cutoff} Å)")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute residue distance matrix and contact map from PDB.")
    ap.add_argument("pdb", help="Path to PDB file")
    ap.add_argument("--only", choices=["protein", "nucleic", "both"], default="both",
                    help="Residue type to include (default: both)")
    ap.add_argument("--chains", nargs="*", help="Only include these chain IDs (e.g., A B H)")
    ap.add_argument("--include-ligand", action="store_true", help="Include HET groups (except waters)")
    ap.add_argument("--mode", choices=["CA", "CB", "P", "COM", "min"], default="CA",
                    help="Representative coordinate: CA/CB/P/COM or minimum atom-atom distance (min)")
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

    # mode 기본값 보정: nucleic만 선택 시 기본 대표점을 P로 자동 전환
    mode = args.mode
    if args.only == "nucleic" and args.mode == "CA":
        mode = "P"

    D = compute_distance_matrix(residues, mode=mode)

    # 라벨 저장
    if args.labels:
        with open(args.labels, "w", encoding="utf-8") as f:
            for i, lab in enumerate(labels):
                f.write(f"{i}\t{lab}\n")
        print(f"Saved labels -> {args.labels}")

    # 결과 저장
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
