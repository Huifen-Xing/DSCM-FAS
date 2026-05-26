# DSCM-FAS
The source code for Dual Semantic Consistency Module for Domain Generalizable Face Anti-Spoofing

## 1. Installation

- Ubuntu 20.04  
- CUDA 11.8  
- Python 3.8  
- pytorch == 2.0.0  

---

## 2. Dataset

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets. Put datasets into the directory of `datasets/FAS`.

- Idiap Replay Attack: https://www.idiap.ch/en/scientific-research/data/replayattack  
- OULU-NPU: https://sites.google.com/site/oulunpudatabase  
- CASIA-MFSD: http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp  
- MSU-MFSD: https://drive.google.com/drive/folders/1nJCPdJ7R67xOiklF1omkfz4yHeJwhQsz  

Data pre-processing: Follow the preprocessing steps in SAFAS.

---

## 3. Demo

Run:

```bash
./train.py --protocol [O_C_I_to_M/O_M_I_to_C/O_C_M_to_I/I_C_M_to_O]
