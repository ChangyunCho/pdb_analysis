# extract_sequences.py
# PDB 파일에서 chain 별로 서열 추출
python extract_sequences.py ./PDB/4ZFG.pdb --out ./results/4ZFG.fasta

# closest_residues.py
# ligand (HETATM 기준) 가까운 residue ranking
python closest_residues.py ./PDB/4ZFG.pdb --top 15

# make_contact_map.py
# PDB 파일의 좌표 정보를 기반으로 dist matrix 혹은 contact map을 시각화 (npz 저장)
python make_contact_map.py ./PDB/4ZFG.pdb --npz ./results/4ZFG_dist.npz --png ./results/4ZFG_contact.png

# peptide_contacts.py
# peptide complex 에서 chain 간 거리 계산
# 1 펩타이드-펩타이드(체인 A vs B) 최소거리/상호작용
python peptide_contacts.py ./PDB/4ZFG.pdb --setA A --setB H --png ./results/A_H_contact.png --csv ./results/A_H_dist.csv
# 2 구간-구간 (A:5-30 vs B:15-40)
python peptide_contacts.py 4ZFG.pdb --setA A:5-30 --setB H:15-40 --txt ./results/A5-30_H15-40_interactions.txt
# 3 리간드-펩타이드 (특정 리간드 vs 체인 H)
python peptide_contacts.py 4ZFG.pdb --setA ADP:B:123 --setB H --png ./results/lig_H.png
# 4 리간드-단백질(전체) (HET 자동 + 단백질 전체)
python peptide_contacts.py 4ZFG.pdb --auto-ligand --setB protein --top 50