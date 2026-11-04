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


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
procs = []


def worker(args, seed, positions, log):
    # The engine runs inside the data folder and writes a relative file name,
    # so the folder may contain spaces / non-ASCII characters.
    out = f"part-{seed}.bin"
    cmds = [f"setoption name Hash value {args.hash}", "setoption name Threads value 1"]
    if args.net:
        cmds += [f"setoption name EvalFile value {os.path.abspath(args.net)}", "setoption name Use NNUE value true"]
    gen = (f"gennnue out={out} positions={positions} depth={args.depth} nodes={args.nodes} "
           f"random={args.random} seed={seed} evallimit={args.evallimit} maxply={args.maxply}")
    if args.book:
        gen += f" book={os.path.abspath(args.book)}"
    cmds += [gen, "quit"]
    p = subprocess.Popen([os.path.abspath(args.engine)], cwd=args.out, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace", bufsize=1,
                         creationflags=CREATE_NO_WINDOW)
    procs.append(p)
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
    ap.add_argument("--stop-file", default="", help="stop when this file appears (default: <out>/STOP)")
    args = ap.parse_args()
    stop_file = args.stop_file or os.path.join(args.out, "STOP")

    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(stop_file):
        os.remove(stop_file)
    lock = threading.Lock()

    def log(s):
        with lock:
            print(s, flush=True)

    per = (args.positions + args.workers - 1) // args.workers
    ts = [threading.Thread(target=worker, args=(args, args.seed + i + 1, per, log), daemon=True)
          for i in range(args.workers)]
    for t in ts:
        t.start()
    def stop_all(msg):
        for p in procs:
            if p.poll() is None:
                p.kill()
        print(msg, "- finished games are already saved in", args.out, flush=True)

    try:
        while any(t.is_alive() for t in ts):
            if os.path.exists(stop_file):
                stop_all("stop file found")
                os.remove(stop_file)
                sys.exit(0)
            time.sleep(1)
    except KeyboardInterrupt:
        stop_all("interrupted")
        sys.exit(1)
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out) if f.endswith(".bin"))
    print(f"done: {total // 264} positions in {args.out}", flush=True)


if __name__ == "__main__":
    main()
