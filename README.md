# DSCM-FAS
The source code for Dual Semantic Consistency Module for Domain Generalizable Face Anti-Spoofing

## 1. Installation

### Environment
- Ubuntu 20.04
- CUDA 11.8
- Python 3.8
- PyTorch 2.0.0

### Install dependencies
```bash
# create conda environment (recommended)
conda create -n fas python=3.8
conda activate fas

# install pytorch (CUDA 11.8)
pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt
