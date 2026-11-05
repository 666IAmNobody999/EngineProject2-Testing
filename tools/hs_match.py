#!/usr/bin/env python3
"""
Minimal Jieqi (揭棋) self-play arbiter for A/B testing two UCI builds of PikaJieQi.

The arbiter owns the hidden-piece bags, draws the identity of every revealed
piece at random and sends both engines the same move list using the engine's
move syntax:  <from><to>[<revealed piece>][<captured hidden piece>]
(e.g. "b2e2C" = the hidden piece on b2 moved to e2 and turned out to be a cannon,
 "e3e6Pn" = hidden pawn moved and captured a hidden black knight).

Game end: no legal move (engine answers "bestmove (none)") loses, as in xiangqi.
Adjudication: both engines agree on |score| >= --adj-cp for --adj-plies plies, or
a mate score for 2 plies; 3-fold repetition, --max-plies or 120 plies without a
capture/reveal are draws.

Time control: --movetime ms, --nodes n, or --tc base+inc (seconds; real clocks, a flag
fall loses; --tc-margin ms of grace for process/pipe latency).

Usage (幻神揭棋引擎 abj-base):
  python3 tools/hs_match.py --e1 ./A --e2 ./B --games 400 --movetime 100 \
      --o1 "SeDepth=6" --o2 "" --concurrency 2
  python3 tools/hs_match.py --e1 ./A --e2 ./A --o1 "TmFall=100" --games 200 --tc 5+0.05
Each pair of games (2k, 2k+1) uses the same reveal sequence seed with colours swapped.
"""
import argparse, math, os, random, subprocess, sys, time
# 幻神揭棋引擎 abj-base: A/B match arbiter (GPLv3).
from multiprocessing import Pool

FILES = "abcdefghi"
RED_BAG = {"R": 2, "A": 2, "C": 2, "P": 5, "N": 2, "B": 2}


def start_board():
    b = {}
    for f in "abcdefghi":
        b[f + "0"] = ("dark", "w")
        b[f + "9"] = ("dark", "b")
    for f in "bh":
        b[f + "2"] = ("dark", "w")
        b[f + "7"] = ("dark", "b")
    for f in "acegi":
        b[f + "3"] = ("dark", "w")
        b[f + "6"] = ("dark", "b")
    b["e0"] = ("K", "w")
    b["e9"] = ("k", "b")
    return b


class Engine:
    def __init__(self, path, options, log=None):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", bufsize=1)
        self.log = log
        self.send("uci")
        self.wait_for("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, s):
        if self.log: self.log.write(">> " + s + "\n")
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait_for(self, token, timeout=60):
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if self.log: self.log.write("<< " + line)
            if line.startswith(token):
                return line.strip()
            if time.time() - t0 > timeout:
                raise RuntimeError("engine timeout waiting for " + token)

    def newgame(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")

    def go(self, moves, go_cmd):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.t_go = time.time()
        self.send(go_cmd)
        score = None
        mate = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if self.log: self.log.write("<< " + line)
            if line.startswith("info") and " score " in line and " multipv 1 " in line + " ":
                parts = line.split()
                i = parts.index("score")
                if parts[i + 1] == "cp":
                    score, mate = int(parts[i + 2]), None
                elif parts[i + 1] == "mate":
                    mate, score = int(parts[i + 2]), None
            elif line.startswith("bestmove"):
                mv = line.split()[1]
                return (None if mv == "(none)" else mv), score, mate

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def board_key(board, stm):
    return stm + "".join(f"{sq}{v[0]}{v[1]}" for sq, v in sorted(board.items()))


def play_game(args_tuple):
    (gid, e1, e2, o1, o2, go_cmd, seed, swap, adj_cp, adj_plies, max_plies, logdir) = args_tuple[:12]
    clock = None
    if go_cmd.startswith("tc "):  # "tc <base ms> <inc ms> <margin ms>"
        _, base, inc, margin = go_cmd.split()
        clock = {"w": float(base), "b": float(base), "inc": int(inc), "margin": float(margin)}
    rng = random.Random(seed)
    log = open(os.path.join(logdir, f"game{gid}.log"), "w", encoding="utf-8") if logdir else None
    # red = engine index 0 unless swapped
    paths = [e1, e2]
    opts = [o1, o2]
    red_idx = 1 if swap else 0
    engines = [Engine(paths[0], opts[0], log), Engine(paths[1], opts[1], log)]
    for e in engines: e.newgame()
    board = start_board()
    bags = {"w": dict(RED_BAG), "b": {k.lower(): v for k, v in RED_BAG.items()}}
    moves = []
    stm = "w"
    reps = {}
    quiet = 0
    hist = []  # (side, score_cp or None, mate or None)
    result = None  # "1-0" red wins, "0-1", "1/2-1/2"
    reason = ""
    try:
        while True:
            idx = red_idx if stm == "w" else 1 - red_idx
            if clock is not None:
                cmd = (f"go wtime {max(1, int(clock['w']))} btime {max(1, int(clock['b']))} "
                       f"winc {clock['inc']} binc {clock['inc']}")
            else:
                cmd = go_cmd
            mv, score, mate = engines[idx].go(moves, cmd)
            if clock is not None:
                clock[stm] -= (time.time() - engines[idx].t_go) * 1000.0
                if clock[stm] < -clock['margin']:
                    result = "0-1" if stm == "w" else "1-0"
                    reason = "time forfeit"
                    break
                clock[stm] = max(clock[stm], 0) + clock['inc']
            if mv is None:
                result = "0-1" if stm == "w" else "1-0"
                reason = "no legal moves"
                break
            frm, to = mv[:2], mv[2:4]
            if frm not in board or board[frm][1] != stm:
                result = "0-1" if stm == "w" else "1-0"
                reason = f"illegal move {mv}"
                break
            tok = mv[:4]
            piece = board[frm]
            captured = board.get(to)
            if piece[0] == "dark":
                bag = bags[stm]
                pool = [k for k, n in bag.items() for _ in range(n)]
                rev = rng.choice(pool)
                bag[rev] -= 1
                tok += rev
                piece = (rev, stm)
            if captured is not None:
                if captured[0] == "dark":
                    obag = bags[captured[1]]
                    pool = [k for k, n in obag.items() for _ in range(n)]
                    cap = rng.choice(pool)
                    obag[cap] -= 1
                    tok += cap
                if captured[0] in ("K", "k"):
                    result = "1-0" if stm == "w" else "0-1"
                    reason = "king captured"
                    break
                del board[to]
            del board[frm]
            board[to] = piece
            moves.append(tok)
            quiet = 0 if (captured is not None or len(tok) > 4) else quiet + 1
            hist.append((stm, score, mate))
            stm = "b" if stm == "w" else "w"
            key = board_key(board, stm)
            reps[key] = reps.get(key, 0) + 1
            if reps[key] >= 3:
                result, reason = "1/2-1/2", "repetition"
                break
            if len(moves) >= max_plies:
                result, reason = "1/2-1/2", "max plies"
                break
            if quiet >= 120:
                result, reason = "1/2-1/2", "120 quiet plies"
                break
            # adjudication (scores are from the mover's point of view)
            if len(hist) >= adj_plies:
                recent = hist[-adj_plies:]
                red_view = []
                for side, sc, mt in recent:
                    if mt is not None:
                        v = 100000 if mt > 0 else -100000
                    elif sc is not None:
                        v = sc
                    else:
                        v = 0
                    red_view.append(v if side == "w" else -v)
                if all(v >= adj_cp for v in red_view):
                    result, reason = "1-0", "adjudication"
                    break
                if all(v <= -adj_cp for v in red_view):
                    result, reason = "0-1", "adjudication"
                    break
            if len(hist) >= 2 and all(h[2] is not None for h in hist[-2:]):
                side, _, mt = hist[-1]
                winner = side if mt > 0 else ("b" if side == "w" else "w")
                result, reason = ("1-0" if winner == "w" else "0-1"), "mate score"
                break
    except Exception as ex:
        result, reason = "crash", f"{type(ex).__name__}: {ex} (stm={stm}, idx={idx})"
    finally:
        for e in engines: e.quit()
        if log:
            log.write(f"RESULT {result} {reason}\nMOVES {' '.join(moves)}\n")
            log.close()
    # convert to engine-1 point of view
    if result == "1-0":
        e1_score = 1.0 if red_idx == 0 else 0.0
    elif result == "0-1":
        e1_score = 0.0 if red_idx == 0 else 1.0
    elif result == "1/2-1/2":
        e1_score = 0.5
    else:
        e1_score = None
    return gid, result, reason, e1_score, len(moves)


def elo(w, l, d):
    n = w + l + d
    if n == 0: return 0.0, 0.0
    s = (w + d / 2) / n
    s = min(max(s, 1e-6), 1 - 1e-6)
    e = -400 * math.log10(1 / s - 1)
    # standard error of the score -> elo
    p_w, p_l, p_d = w / n, l / n, d / n
    var = p_w * (1 - s) ** 2 + p_l * (0 - s) ** 2 + p_d * (0.5 - s) ** 2
    se = math.sqrt(var / n) if n else 0
    lo = -400 * math.log10(1 / min(max(s - 1.96 * se, 1e-6), 1 - 1e-6) - 1)
    hi = -400 * math.log10(1 / min(max(s + 1.96 * se, 1e-6), 1 - 1e-6) - 1)
    return e, (hi - lo) / 2


def parse_opts(s):
    d = {}
    if s:
        for kv in s.split(","):
            k, v = kv.split("=")
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True); ap.add_argument("--e2", required=True)
    ap.add_argument("--o1", default=""); ap.add_argument("--o2", default="")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--tc", default="", help="time control 'base+inc' in seconds, e.g. 5+0.05")
    ap.add_argument("--tc-margin", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--adj-cp", type=int, default=2000)
    ap.add_argument("--adj-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--logdir", default="")
    a = ap.parse_args()
    if a.nodes:
        go_cmd = f"go nodes {a.nodes}"
    elif a.tc:
        base, _, inc = a.tc.partition("+")
        go_cmd = f"tc {int(float(base) * 1000)} {int(float(inc or 0) * 1000)} {a.tc_margin}"
    else:
        go_cmd = f"go movetime {a.movetime or 100}"
    if a.logdir: os.makedirs(a.logdir, exist_ok=True)
    o1, o2 = parse_opts(a.o1), parse_opts(a.o2)
    jobs = [(g, a.e1, a.e2, o1, o2, go_cmd, a.seed * 100000 + g // 2, g % 2 == 1, a.adj_cp, a.adj_plies,
             a.max_plies, a.logdir) for g in range(a.games)]
    w = l = d = crash = 0
    t0 = time.time()
    with Pool(a.concurrency) as pool:
        for gid, res, reason, s, plies in pool.imap_unordered(play_game, jobs):
            if s is None: crash += 1
            elif s == 1.0: w += 1
            elif s == 0.0: l += 1
            else: d += 1
            e, pm = elo(w, l, d)
            print(f"game {gid:3d} {res:7s} {plies:3d} plies ({reason})  | e1: +{w} -{l} ={d} "
                  f"crash={crash}  Elo {e:+.0f} ± {pm:.0f}  [{time.time()-t0:.0f}s]", flush=True)
    e, pm = elo(w, l, d)
    print(f"\nFINAL e1 vs e2: +{w} -{l} ={d} (crash {crash})  Elo {e:+.1f} ± {pm:.1f}")


if __name__ == "__main__":
    main()
