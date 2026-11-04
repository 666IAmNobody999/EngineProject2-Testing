#!/usr/bin/env python3
"""
Check that the engine computes exactly what the trainer exported.

  python3 verify.py --engine ../../src/PikaJieQi --net nets/pikajieqi.nnue --data check.bin --fen check.fen

check.bin / check.fen come from one `gennnue ... out=check.bin fen=check.fen` run
(same order). For every position the integer reference implementation in
jqnnue.py is compared with the engine command `nnueeval`.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jqnnue as J  # noqa: E402


def read_net(path):
    with open(path, "rb") as f:
        b = f.read()
    import struct
    ver, h, n = struct.unpack_from("<III", b, 0)
    o = 12 + n
    o += 4
    # infer L1 from the file size
    for l1 in (128, 256, 384, 512, 768, 1024, 1536, 2048):
        if h == J.file_hash(l1):
            break
    else:
        sys.exit("unknown network hash")
    q = {"l1": l1}

    def take(dt, count):
        nonlocal o
        a = np.frombuffer(b, dtype=dt, count=count, offset=o)
        o += a.nbytes
        return a

    q["ft_b"] = take("<i2", l1)
    q["ft_w"] = take("<i2", l1 * J.NUM_FEATURES).reshape(J.NUM_FEATURES, l1)
    q["psqt"] = take("<i4", J.PSQT_BUCKETS * J.NUM_FEATURES).reshape(J.NUM_FEATURES, J.PSQT_BUCKETS)
    for k in ("fc0", "fc1", "fc2"):
        q[k + "_w"], q[k + "_b"] = [], []
    dims = (("fc0", J.FC0_OUT + 1, l1), ("fc1", J.FC1_OUT, J.FC0_OUT * 2), ("fc2", 1, J.FC1_OUT))
    for _ in range(J.LAYER_STACKS):
        o += 4
        for k, nout, nin in dims:
            q[k + "_b"].append(take("<i4", nout))
            q[k + "_w"].append(take("i1", nout * J._pad32(nin)).reshape(nout, -1)[:, :nin])
    assert o == len(b), "trailing bytes"
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fen", required=True)
    ap.add_argument("--count", type=int, default=200)
    args = ap.parse_args()

    q = read_net(args.net)
    recs = J.load_records([args.data])
    fens = [l.split("|")[0].strip() for l in open(args.fen) if l.strip()]
    n = min(args.count, len(recs), len(fens))
    ref = [J.reference_eval(q, recs[i]) for i in range(n)]

    cmds = [f"setoption name EvalFile value {os.path.abspath(args.net)}", "setoption name Use NNUE value true"]
    for f in fens[:n]:
        cmds += [f"position fen {f}", "nnueeval"]
    cmds.append("quit")
    out = subprocess.run([args.engine], input="\n".join(cmds) + "\n", capture_output=True, text=True).stdout
    eng = [int(l.split()[1]) for l in out.splitlines() if l.startswith("nnue ")]
    if len(eng) != n:
        sys.exit(f"engine returned {len(eng)} values for {n} positions:\n{out[-2000:]}")
    bad = [(i, r, e) for i, (r, e) in enumerate(zip(ref, eng)) if r != e]
    print(f"{n} positions, {n - len(bad)} identical, {len(bad)} different")
    for i, r, e in bad[:10]:
        print(f"  #{i}: reference {r} engine {e}  {fens[i]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
