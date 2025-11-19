#FDS Kaggle Competition – Toolbox (Pacha)

This repository contains the helper code used for the FDS Pokémon Battle Prediction Challenge (Sapienza University, 2025).

##Structure
- src/features.py → Feature engineering pipeline  
- src/models.py → Final ExtraTrees model  
- src/utils.py → JSONL loader  

##How to use this repo in Kaggle

```python
!git clone https://github.com/veravionnet2239308/fds-kagglecompetition-veravionnet.git

import sys
sys.path.append("/kaggle/working/fds-kagglecompetition-veravionnet/src")

from features import extract_features
from models import build_model
from utils import load_jsonl
