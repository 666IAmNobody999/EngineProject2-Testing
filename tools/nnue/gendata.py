#!/usr/bin/env python3
"""
Generate NNUE training data with several engine processes in parallel.

  python3 gendata.py --engine ../../src/PikaJieQi --out data --positions 2000000 --workers 8 --depth 8

Each worker runs the UCI command `gennnue` with its own seed and writes
data/part-<seed>.bin (appending, so an interrupted run can simply be started
again with new seeds). With --net the workers search with that NNUE network
(later generations: data from the previous net, as Stockfish did).
"""
import argparse
import os
import subprocess
import sys
import threading
import time


def worker(args, seed, positions, log):
    out = os.path.join(args.out, f"part-{seed}.bin")
    cmds = [f"setoption name Hash value {args.hash}", "setoption name Threads value 1"]
    if args.net:
        cmds += [f"setoption name EvalFile value {os.path.abspath(args.net)}", "setoption name Use NNUE value true"]
    gen = (f"gennnue out={out} positions={positions} depth={args.depth} nodes={args.nodes} "
           f"random={args.random} seed={seed} evallimit={args.evallimit} maxply={args.maxply}")
    if args.book:
        gen += f" book={os.path.abspath(args.book)}"
    cmds += [gen, "quit"]
    p = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("\n".join(cmds) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if "gennnue" in line or "NNUE" in line:
            log(f"[seed {seed}] {line.strip()}")
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--positions", type=int, default=1_000_000, help="total positions")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=0, help="nodes per move instead of depth (0 = use depth)")
    ap.add_argument("--random", type=int, default=6, help="random moves at the start of every game")
    ap.add_argument("--evallimit", type=int, default=3000)
    ap.add_argument("--maxply", type=int, default=400)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000 * 100)
    ap.add_argument("--net", default="", help="search with this NNUE network")
    ap.add_argument("--book", default="", help="file with start FENs, one per line")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lock = threading.Lock()

    def log(s):
        with lock:
            print(s, flush=True)

    per = (args.positions + args.workers - 1) // args.workers
    ts = [threading.Thread(target=worker, args=(args, args.seed + i + 1, per, log), daemon=True)
          for i in range(args.workers)]
    for t in ts:
        t.start()
    try:
        for t in ts:
            t.join()
    except KeyboardInterrupt:
        print("interrupted: finished games are already saved in", args.out)
        sys.exit(1)


if __name__ == "__main__":
    main()
