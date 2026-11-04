#!/usr/bin/env python3
"""Write a .nnue file from a training checkpoint:  python3 export.py nets/checkpoint.pt out.nnue"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jqnnue as J  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
ck = torch.load(sys.argv[1], map_location="cpu")
model = J.build_model(ck["l1"], torch)
model.load_state_dict(ck["model"])
J.write_nnue(J.quantize(model), sys.argv[2], f"PikaJieQi NNUE L1={ck['l1']} step={ck['step']}")
print(f"wrote {sys.argv[2]} (L1={ck['l1']}, step {ck['step']})")
