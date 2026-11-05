#!/usr/bin/env python3
# 幻神揭棋引擎 abj-base: SPSA tuner for the engine's search parameters (GPLv3).
"""
SPSA tuning of the hidden search parameters (setoption name <Param> value <n>), Fishtest style.

Every iteration plays one game pair (same reveal seed, colours swapped) between
theta + c_k * flip and theta - c_k * flip, then moves theta by R_k * c_k * result * flip,
where result = wins - losses of the "+" side in the pair (-2 .. 2).

  c_k = c / (k + 1)^gamma,  a_k = a / (A + k + 1)^alpha,  R_k = a_k / c_k^2
  with c = c_end * N^gamma, a = r_end * c_end^2 * (A + N)^alpha  (N = iterations).

Parameters come from a JSON file (see tools/spsa_params.json):
  [{"name": "RfpMargin", "start": 90, "min": 30, "max": 300, "c_end": 8, "r_end": 0.002}, ...]

Run (Windows: python tools\\hs_spsa.py --engine 幻神揭棋引擎.exe ...):
  python3 tools/hs_spsa.py --engine ./abjchess --params tools/spsa_params.json \\
      --iterations 20000 --tc 5+0.05 --concurrency 16 --state spsa_state.json
The state file is rewritten after every finished pair; rerun with the same --state to resume.
Rule of thumb: tens of thousands of pairs at a short TC; check the result with hs_match.py
(theta vs defaults, >= 1000 games at a short and a longer TC) before adopting anything.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hs_match  # noqa: E402


def load_state(path, params, n_iter, A_frac):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        for p in params:
            st["theta"].setdefault(p["name"], float(p["start"]))
        return st
    return {"k": 0, "theta": {p["name"]: float(p["start"]) for p in params}, "n": n_iter,
            "A": A_frac * n_iter, "pairs": 0, "w": 0, "l": 0, "d": 0, "history": []}


def save_state(path, st):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def clamp(v, p):
    return max(p["min"], min(p["max"], v))


def play_pair(job):
    (k, engine, plus, minus, go_cmd, seed, adj_cp, adj_plies, max_plies) = job
    score = 0.0
    wins = losses = draws = 0
    for swap in (False, True):
        _, res, reason, s, _ = hs_match.play_game(
            (0, engine, engine, plus, minus, go_cmd, seed, swap, adj_cp, adj_plies, max_plies, ""))
        if s is None:
            return k, None, 0, 0, 0
        if s == 1.0: wins += 1
        elif s == 0.0: losses += 1
        else: draws += 1
    return k, wins - losses, wins, losses, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--movetime", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--tc-margin", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--fixed", default="", help="options for both sides, e.g. Hash=16,Threads=1")
    ap.add_argument("--state", default="spsa_state.json")
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--A-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--adj-cp", type=int, default=2000)
    ap.add_argument("--adj-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=400)
    a = ap.parse_args()

    with open(a.params, encoding="utf-8") as f:
        params = json.load(f)
    if a.nodes:
        go_cmd = f"go nodes {a.nodes}"
    elif a.movetime:
        go_cmd = f"go movetime {a.movetime}"
    else:
        base, _, inc = a.tc.partition("+")
        go_cmd = f"tc {int(float(base) * 1000)} {int(float(inc or 0) * 1000)} {a.tc_margin}"
    fixed = hs_match.parse_opts(a.fixed)
    fixed.setdefault("Hash", "16")
    st = load_state(a.state, params, a.iterations, a.A_frac)
    N, A = st["n"], st["A"]
    for p in params:
        p["c"] = p["c_end"] * N ** a.gamma
        p["a"] = p["r_end"] * p["c_end"] ** 2 * (A + N) ** a.alpha
    rng = random.Random(a.seed * 7919 + st["k"])
    t0 = time.time()

    def make_job(k):
        flips, plus, minus = {}, dict(fixed), dict(fixed)
        for p in params:
            ck = p["c"] / (k + 1) ** a.gamma
            fl = rng.choice((-1, 1))
            flips[p["name"]] = (fl, ck)
            th = st["theta"][p["name"]]
            plus[p["name"]] = str(int(round(clamp(th + ck * fl, p))))
            minus[p["name"]] = str(int(round(clamp(th - ck * fl, p))))
        job = (k, a.engine, plus, minus, go_cmd, a.seed * 1000003 + k, a.adj_cp, a.adj_plies, a.max_plies)
        return job, flips

    pending = {}
    next_k = st["k"]
    with ProcessPoolExecutor(max_workers=a.concurrency) as ex:
        while next_k < N or pending:
            while next_k < N and len(pending) < a.concurrency:
                job, flips = make_job(next_k)
                pending[ex.submit(play_pair, job)] = (next_k, flips)
                next_k += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                k, flips = pending.pop(fut)
                _, result, w, l, d = fut.result()
                if result is None:
                    print(f"pair {k}: crash, skipped", flush=True)
                    continue
                for p in params:
                    fl, ck = flips[p["name"]]
                    ak = p["a"] / (A + k + 1) ** a.alpha
                    rk = ak / ck ** 2
                    st["theta"][p["name"]] = clamp(st["theta"][p["name"]] + rk * ck * result * fl, p)
                st["pairs"] += 1; st["w"] += w; st["l"] += l; st["d"] += d
                st["k"] = max(st["k"], k + 1)
                if st["pairs"] % 50 == 0:
                    st["history"].append({"pairs": st["pairs"], "theta": dict(st["theta"])})
                save_state(a.state, st)
                if st["pairs"] % 10 == 0:
                    el = time.time() - t0
                    th = ", ".join(f"{n}={v:.1f}" for n, v in st["theta"].items())
                    print(f"[{st['pairs']}/{N} pairs, {el/60:.1f} min] {th}", flush=True)
    print("\nFinal parameters (setoption form):")
    for p in params:
        print(f"setoption name {p['name']} value {int(round(st['theta'][p['name']]))}")
    print("\n--o1 for hs_match.py:")
    print(",".join(f"{p['name']}={int(round(st['theta'][p['name']]))}" for p in params))


if __name__ == "__main__":
    main()
