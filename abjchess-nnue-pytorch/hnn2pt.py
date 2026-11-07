#!/usr/bin/env python3
# This file is part of 幻神揭棋引擎 (based on AB JChess), GNU GPL v3 or later.
"""Convert an engine HNN small net (.hnn, or a compressed .nnue such as abjchess-20261030.nnue) back into a
train_hnn.py checkpoint (.pt), so that training can continue from it with  train_hnn.py --init net.pt.

The weights are de-quantised exactly (the export scales of HNN._export_base), so exporting the resulting
checkpoint again gives a byte-identical .hnn (checked with --verify).  Only the standard JQHNNUE1 (version 1,
L1=16, L2=32) layout written by the released nets is supported.

  python3 hnn2pt.py abjchess-20261030.nnue base.pt --verify
"""
import argparse
import filecmp
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hnn_ref as HR  # noqa: E402
import train_hnn as TH  # noqa: E402


def load_hnn_path(path):
    magic = open(path, "rb").read(8)
    if magic == HR.MAGIC:
        return Path(path)
    if magic in TH.JQNZ_MAGICS:
        return TH.jqnz_decode(path)
    raise SystemExit(f"{path}: not an HNN small net (magic {magic!r})")


def convert(path):
    hp = load_hnn_path(path)
    b = open(hp, "rb").read()
    version, w, nf, scheme, npsqt, nst, l1, l2 = struct.unpack_from("<8I", b, 8)
    if version != 1 or (l1, l2) != (16, 32):
        raise SystemExit(f"unsupported HNN layout: version {version}, L1 {l1}, L2 {l2} (only version 1, 16/32)")
    net = HR.Net(str(hp))
    m = TH.HNN(w, scheme, factor=False)
    f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.float64)).float()
    with torch.no_grad():
        m.ft_bias.copy_(f32(net.ftB) / 127)
        m.ft[:, :w].copy_(f32(net.ftW) / 127)
        m.ft[:, w:].copy_(f32(net.psqt) / 16)
        m.l1b.copy_(f32(net.l1b) / (64 * 127))
        m.l1w.copy_(f32(net.l1w) / 64)
        m.l2b.copy_(f32(net.l2b) / (64 * 127))
        m.l2w.copy_(f32(net.l2w[:, :, :2 * (16 - 1)]) / 64)
        m.ob.copy_(f32(net.fb) / 16)
        m.ow.copy_(f32(net.fw) * 127 / 16)
    return m, hp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", help=".hnn or compressed .nnue (JQHNNZ01/JQHNNL01)")
    ap.add_argument("out", help="output checkpoint .pt for train_hnn.py --init")
    ap.add_argument("--verify", action="store_true", help="re-export and check the .hnn is byte-identical")
    a = ap.parse_args()
    m, hp = convert(a.net)
    torch.save(m.state(), a.out)
    print(f"{a.net}: width {m.W}, kb-scheme {m.scheme} -> {a.out}")
    if a.verify:
        with tempfile.TemporaryDirectory() as d:
            re = Path(d) / "re.hnn"
            TH.HNN.from_state(torch.load(a.out, map_location="cpu", weights_only=False)).export(str(re))
            ok = filecmp.cmp(re, hp, shallow=False)
            print("verify: re-exported .hnn is", "byte-identical" if ok else "DIFFERENT")
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
