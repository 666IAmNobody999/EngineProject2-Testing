#!/usr/bin/env python3
"""
幻神揭棋引擎 NNUE trainer (CPU or CUDA, PyTorch).

  python3 train.py data/*.bin --out nets
  python3 train.py data/*.bin --out nets --resume nets/checkpoint.pt

Stop at any time with Ctrl+C (or SIGTERM, or by creating the file nets/STOP,
which the Windows trainer GUI does): the trainer finishes the current step,
writes nets/checkpoint.pt and nets/pikajieqi.nnue, and exits.
nets/pikajieqi-best.nnue is always the net with the lowest validation loss. A second
Ctrl+C aborts immediately. Every --save-every steps it also writes
nets/pikajieqi.nnue plus a numbered copy, so a killed run still leaves a net.

Loss (as nnue-pytorch): both the search score and the network output are
turned into win probabilities with sigmoid(v / scale), the target mixes the
score with the game result,  t = lambda * p(score) + (1 - lambda) * result,
and the loss is mean(|p(net) - t| ^ 2.5).
"""

import argparse
import math
import os
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jqnnue as J  # noqa: E402

stop_requested = False


def on_signal(signum, frame):
    global stop_requested
    if stop_requested:
        print("\nsecond interrupt: aborting without saving", flush=True)
        os._exit(1)
    stop_requested = True
    print("\nstop requested: saving checkpoint and network after this step ...", flush=True)


def batch_tensors(recs, idx):
    r = recs[idx]
    us = r["us"].astype(np.int64)
    them = r["them"].astype(np.int64)
    us[us == J.PAD] = J.NUM_FEATURES
    them[them == J.PAD] = J.NUM_FEATURES
    return (torch.from_numpy(us), torch.from_numpy(them),
            torch.from_numpy(r["bucket"].astype(np.int64)),
            torch.from_numpy(r["score"].astype(np.float32)),
            torch.from_numpy((r["result"].astype(np.float32) + 1.0) / 2.0))


def loss_fn(out, score, result, args):
    out = out.view(-1)                       # (B, 1) -> (B,): no (B, B) broadcast
    q = torch.sigmoid(out * J.NNUE2SCORE / args.scale)
    p = torch.sigmoid(score / args.scale)
    t = args.lambda_ * p + (1.0 - args.lambda_) * result
    return torch.abs(q - t).pow(2.5).mean()


def save(model, opt, sched, step, args, tag=None):
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "checkpoint.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "l1": model.l1}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)
    q = J.quantize(model)
    net = os.path.join(args.out, "pikajieqi.nnue")
    J.write_nnue(q, net + ".tmp", f"PikaJieQi NNUE L1={model.l1} step={step}")
    os.replace(net + ".tmp", net)
    if tag is not None:
        J.write_nnue(q, os.path.join(args.out, f"pikajieqi-{tag}.nnue"),
                     f"PikaJieQi NNUE L1={model.l1} step={step}")
    return net


@torch.no_grad()
def evaluate(model, recs, idx, args):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(idx), args.batch):
        us, them, b, s, r = batch_tensors(recs, idx[i:i + args.batch])
        out = model(us.to(args.device), them.to(args.device), b.to(args.device))
        tot += float(loss_fn(out, s.to(args.device), r.to(args.device), args)) * len(us)
        n += len(us)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help=".bin files written by gennnue, or folders containing them")
    ap.add_argument("--out", default="nets", help="output directory (checkpoint + .nnue)")
    ap.add_argument("--l1", type=int, default=512, help="L1 width, must match the engine (make NNUE_L1=...)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.992, help="lr decay per epoch")
    ap.add_argument("--epoch-size", type=int, default=1_000_000, help="positions per epoch")
    ap.add_argument("--epochs", type=int, default=400, help="stop after this many epochs")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.8,
                    help="1.0 = learn the search score only, 0.0 = game result only")
    ap.add_argument("--scale", type=float, default=361.0, help="sigmoid scale, engine units")
    ap.add_argument("--val", type=float, default=0.01, help="fraction held out for validation")
    ap.add_argument("--save-every", type=int, default=2000, help="steps between saves")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--resume", default="", help="checkpoint.pt to continue from")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stop-file", default="", help="stop when this file appears (default: <out>/STOP)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # A directory stands for all part-*.bin files in it
    paths = []
    for d in args.data:
        if os.path.isdir(d):
            paths += sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("part-") and f.endswith(".bin"))
        else:
            paths.append(d)
    recs = J.load_records(paths)
    n = len(recs)
    if n < 1000:
        sys.exit(f"only {n} positions: generate more data first")
    perm = rng.permutation(n)
    nval = max(1000, int(n * args.val)) if args.val > 0 else 0
    val_idx, train_idx = np.sort(perm[:nval]), perm[nval:]
    print(f"{n} positions ({len(train_idx)} train, {nval} validation), L1={args.l1}, device={args.device}")

    model = J.build_model(args.l1, torch).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    steps_per_epoch = max(1, args.epoch_size // args.batch)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=steps_per_epoch, gamma=args.gamma)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device)
        if ck["l1"] != args.l1:
            sys.exit(f"checkpoint has L1={ck['l1']}, not {args.l1}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    stop_file = args.stop_file or os.path.join(args.out, "STOP")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(stop_file):
        os.remove(stop_file)
    best_file = os.path.join(args.out, "best.txt")
    best = float(open(best_file).read().split()[0]) if args.resume and os.path.exists(best_file) else float("inf")

    def keep_best(vl, step):
        nonlocal_best[0] = min(nonlocal_best[0], vl)
        if vl <= nonlocal_best[0]:
            J.write_nnue(J.quantize(model), os.path.join(args.out, "pikajieqi-best.nnue"),
                         f"PikaJieQi NNUE L1={model.l1} step={step} best")
            with open(best_file, "w") as f:
                f.write(f"{vl} {step}\n")
            print(f"step {step}: new best validation loss {vl:.6f} -> pikajieqi-best.nnue", flush=True)
    nonlocal_best = [best]

    total_steps = args.epochs * steps_per_epoch
    t0, run_loss, run_n = time.time(), 0.0, 0
    while step < total_steps and not stop_requested:
        if step % 20 == 0 and os.path.exists(stop_file):
            print("stop file found: saving checkpoint and network ...", flush=True)
            break
        idx = np.sort(rng.choice(train_idx, size=args.batch, replace=False if len(train_idx) > args.batch else True))
        us, them, b, s, r = batch_tensors(recs, idx)
        out = model(us.to(args.device), them.to(args.device), b.to(args.device))
        loss = loss_fn(out, s.to(args.device), r.to(args.device), args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        model.clip_weights()
        step += 1
        run_loss += float(loss.detach())
        run_n += 1

        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step {step}  epoch {step / steps_per_epoch:.2f}  loss {run_loss / run_n:.6f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {step * args.batch / max(dt, 1e-9):.0f} pos/s", flush=True)
            run_loss, run_n = 0.0, 0
        if step % args.save_every == 0:
            vl = evaluate(model, recs, val_idx, args) if nval else float("nan")
            net = save(model, opt, sched, step, args, tag=step)
            print(f"step {step}: validation loss {vl:.6f}, saved {net}", flush=True)
            if nval:
                keep_best(vl, step)

    vl = evaluate(model, recs, val_idx, args) if nval else float("nan")
    net = save(model, opt, sched, step, args, tag=step)
    if nval:
        keep_best(vl, step)
    if os.path.exists(stop_file):
        os.remove(stop_file)
    print(f"finished at step {step}: validation loss {vl:.6f}")
    print(f"network: {net}   (resume with --resume {os.path.join(args.out, 'checkpoint.pt')})")


if __name__ == "__main__":
    main()
