"""
幻神揭棋引擎 (PikaJieQi) NNUE: data format, PyTorch model, quantisation and an
integer reference implementation that matches the engine bit for bit.

Architecture (same family as Stockfish SFNNv5 / Pikafish):

  features  HalfKAv2_hm_JieQi, 8130 inputs per perspective
            6 king buckets x 15 planes x 90 squares (12 piece planes, king,
            own hidden pieces, enemy hidden pieces) + 30 hidden-bag features
  FT        8130 -> L1 (int16) per perspective, + 8 PSQT buckets (int32)
            clipped to [0,1], pairwise product of the two halves -> L1/2,
            both perspectives concatenated -> L1
  stacks    8 layer stacks chosen by (pieces on board - 1) / 4:
            L1 -> 16 (15 + 1 forwarded) -> [sqr-crelu(15), crelu(15)] = 30
            -> 32 crelu -> 1
  output    (stack + forward + (psqt_us - psqt_them)/2) * 600 = engine units
"""

import os
import struct
import numpy as np

# ---- feature set (must match src/nnue/features/half_ka_v2_hm.h) -------------
SQUARE_NB = 90
PS_NB = 15 * SQUARE_NB
KING_BUCKETS = 6
BAG_BASE = KING_BUCKETS * PS_NB
BAG_PER_SIDE = 15
NUM_FEATURES = BAG_BASE + 2 * BAG_PER_SIDE          # 8130
MAX_ACTIVE = 64
PAD = 0xFFFF

FEATURE_HASH = 0x4A1E0C17
VERSION = 0x7AF32F20
PSQT_BUCKETS = 8
LAYER_STACKS = 8
FC0_OUT = 15
FC1_OUT = 32
OUTPUT_SCALE = 16
WEIGHT_SCALE_BITS = 6
NNUE2SCORE = 600

# ---- training record (must match src/datagen.cpp) --------------------------
RECORD_DTYPE = np.dtype([
    ("score", "<i2"), ("result", "i1"), ("stm", "u1"), ("bucket", "u1"),
    ("n_us", "u1"), ("n_them", "u1"), ("reserved", "u1"),
    ("us", "<u2", MAX_ACTIVE), ("them", "<u2", MAX_ACTIVE)])
assert RECORD_DTYPE.itemsize == 264


class _MultiFile:
    """Several memory-mapped .bin files seen as one array, without copying
    them into RAM (np.concatenate would need the whole data set in memory)."""

    def __init__(self, arrs):
        self.arrs = arrs
        self.starts = np.cumsum([0] + [len(a) for a in arrs])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, idx):
        if np.isscalar(idx):
            f = int(np.searchsorted(self.starts, idx, side="right") - 1)
            return self.arrs[f][idx - self.starts[f]]
        idx = np.asarray(idx)
        out = np.empty(len(idx), dtype=RECORD_DTYPE)
        f = np.searchsorted(self.starts, idx, side="right") - 1
        for i in np.unique(f):
            m = f == i
            out[m] = self.arrs[i][idx[m] - self.starts[i]]
        return out


def load_records(paths):
    """Memory-map one or more .bin files from `gennnue` (data stays on disk)."""
    # A generator killed while writing may leave a partial record at the end
    # of a file: it is ignored.
    arrs = []
    for p in paths:
        n = os.path.getsize(p) // RECORD_DTYPE.itemsize
        if n:
            arrs.append(np.memmap(p, dtype=RECORD_DTYPE, mode="r", shape=(n,)))
    if not arrs:
        raise ValueError("no training data")
    return arrs[0] if len(arrs) == 1 else _MultiFile(arrs)


def factor_map():
    """Virtual 'P' feature for the factoriser: the same piece on the same
    square, independent of the king bucket. Bag features have none (-> pad)."""
    m = np.full(NUM_FEATURES + 1, PS_NB, dtype=np.int64)
    idx = np.arange(BAG_BASE)
    m[idx] = idx % PS_NB
    return m


# ---- hashes (must match the C++ get_hash_value() chain) --------------------
M32 = 0xFFFFFFFF


def _affine_hash(prev, out_dims):
    h = (0xCC03DAE4 + out_dims) & M32
    h ^= prev >> 1
    h ^= (prev << 31) & M32
    return h & M32


def _crelu_hash(prev):
    return (0x538D24C7 + prev) & M32


def ft_hash(l1):
    return (FEATURE_HASH ^ (l1 * 2)) & M32


def network_hash(l1):
    h = (0xEC42E90D ^ (l1 * 2)) & M32
    h = _affine_hash(h, FC0_OUT + 1)
    h = _crelu_hash(h)
    h = _affine_hash(h, FC1_OUT)
    h = _crelu_hash(h)
    h = _affine_hash(h, 1)
    return h


def file_hash(l1):
    return ft_hash(l1) ^ network_hash(l1)


# ---- PyTorch model ---------------------------------------------------------
def build_model(l1, torch):
    nn = torch.nn

    # Rough material values (engine units) to start the PSQT part from
    piece_value = {0: 1284, 2: 218, 4: 650, 6: 164, 8: 707, 10: 271, 13: 472}

    class JieQiNNUE(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = l1
            width = l1 + PSQT_BUCKETS
            self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, width, mode="sum", padding_idx=NUM_FEATURES)
            self.ft_bias = nn.Parameter(torch.zeros(l1))
            # factoriser: shared piece-square weights, folded into ft at export
            self.fact = nn.EmbeddingBag(PS_NB + 1, width, mode="sum", padding_idx=PS_NB)
            self.fc0 = nn.Linear(l1, (FC0_OUT + 1) * LAYER_STACKS)
            self.fc1 = nn.Linear(FC0_OUT * 2, FC1_OUT * LAYER_STACKS)
            self.fc2 = nn.Linear(FC1_OUT, LAYER_STACKS)
            self.register_buffer("vmap", torch.from_numpy(factor_map()))
            self._init()

        def _init(self):
            with torch.no_grad():
                self.ft.weight.normal_(0.0, 0.01)
                self.ft.weight[:, l1:] = 0.0
                self.ft.weight[NUM_FEATURES].zero_()
                self.ft_bias.fill_(0.1)
                self.fact.weight.zero_()
                for plane, v in piece_value.items():
                    for own, sign in ((plane, 1.0), (plane + 1, -1.0)):
                        sl = slice(own * SQUARE_NB, (own + 1) * SQUARE_NB)
                        self.fact.weight[sl, l1:] = sign * v / NNUE2SCORE
                # same initial weights for every layer stack
                for lin, n in ((self.fc0, FC0_OUT + 1), (self.fc1, FC1_OUT), (self.fc2, 1)):
                    w0, b0 = lin.weight[:n].clone(), lin.bias[:n].clone()
                    for i in range(LAYER_STACKS):
                        lin.weight[i * n:(i + 1) * n] = w0
                        lin.bias[i * n:(i + 1) * n] = b0
                self.fc2.weight.zero_()
                self.fc2.bias.zero_()

        def clip_weights(self):
            with torch.no_grad():
                lim = 127.0 / (1 << WEIGHT_SCALE_BITS)
                self.fc0.weight.clamp_(-lim, lim)
                self.fc1.weight.clamp_(-lim, lim)
                lim2 = 127.0 * 127.0 / (NNUE2SCORE * OUTPUT_SCALE)
                self.fc2.weight.clamp_(-lim2, lim2)

        def forward(self, us, them, bucket):
            # us / them: (B, 64) int64 feature indices, NUM_FEATURES = unused
            def acc(idx):
                return self.ft(idx) + self.fact(self.vmap[idx])
            a_us, a_them = acc(us), acc(them)
            h = l1 // 2
            b = bucket.view(-1, 1)

            def pair(a):
                x = torch.clamp(a[:, :l1] + self.ft_bias, 0.0, 1.0)
                return x[:, :h] * x[:, h:] * (127.0 / 128.0)

            l0 = torch.cat([pair(a_us), pair(a_them)], dim=1)
            psqt = (a_us[:, l1:].gather(1, b) - a_them[:, l1:].gather(1, b)) / 2.0

            n0 = FC0_OUT + 1
            x = self.fc0(l0).view(-1, LAYER_STACKS, n0)
            x = x.gather(1, b.view(-1, 1, 1).expand(-1, 1, n0)).squeeze(1)
            fwd = x[:, FC0_OUT:]
            y = x[:, :FC0_OUT]
            y = torch.clamp(torch.cat([y * y * (127.0 / 128.0), y], dim=1), 0.0, 1.0)
            x = self.fc1(y).view(-1, LAYER_STACKS, FC1_OUT)
            x = x.gather(1, b.view(-1, 1, 1).expand(-1, 1, FC1_OUT)).squeeze(1)
            x = torch.clamp(x, 0.0, 1.0)
            x = self.fc2(x).gather(1, b)
            return x + fwd + psqt                      # x 600 = engine units

    return JieQiNNUE()


# ---- quantisation / .nnue export -------------------------------------------
def quantize(model):
    """Integer parameters exactly as stored in the .nnue file."""
    import torch
    l1 = model.l1
    with torch.no_grad():
        vmap = model.vmap[:NUM_FEATURES]
        W = (model.ft.weight[:NUM_FEATURES] + model.fact.weight[vmap]).double().cpu().numpy()
        q = {"l1": l1}
        q["ft_b"] = np.round(model.ft_bias.double().cpu().numpy() * 127).astype(np.int16)
        ftw = np.round(W[:, :l1] * 127)
        if np.abs(ftw).max() > 32767:
            raise ValueError("feature transformer weight overflow")
        q["ft_w"] = ftw.astype(np.int16)
        q["psqt"] = np.round(W[:, l1:] * NNUE2SCORE * OUTPUT_SCALE).astype(np.int32)
        s = float(1 << WEIGHT_SCALE_BITS)

        def lin(layer, n, wscale, bscale):
            w = layer.weight.double().cpu().numpy().reshape(LAYER_STACKS, n, -1)
            b = layer.bias.double().cpu().numpy().reshape(LAYER_STACKS, n)
            return (np.clip(np.round(w * wscale), -127, 127).astype(np.int8),
                    np.round(b * bscale).astype(np.int32))

        q["fc0_w"], q["fc0_b"] = lin(model.fc0, FC0_OUT + 1, s, s * 127)
        q["fc1_w"], q["fc1_b"] = lin(model.fc1, FC1_OUT, s, s * 127)
        out = NNUE2SCORE * OUTPUT_SCALE
        q["fc2_w"], q["fc2_b"] = lin(model.fc2, 1, out / 127.0, out)
    return q


def write_nnue(q, path, description="PikaJieQi NNUE (tools/nnue/train.py)"):
    l1 = q["l1"]
    desc = description.encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<III", VERSION, file_hash(l1), len(desc)))
        f.write(desc)
        f.write(struct.pack("<I", ft_hash(l1)))
        f.write(q["ft_b"].astype("<i2").tobytes())
        f.write(q["ft_w"].astype("<i2").tobytes())
        f.write(q["psqt"].astype("<i4").tobytes())
        nh = network_hash(l1)

        def affine(w, b, padded_in):
            f.write(b.astype("<i4").tobytes())
            wp = np.zeros((w.shape[0], padded_in), dtype=np.int8)
            wp[:, :w.shape[1]] = w
            f.write(wp.tobytes())

        for i in range(LAYER_STACKS):
            f.write(struct.pack("<I", nh))
            affine(q["fc0_w"][i], q["fc0_b"][i], _pad32(l1))
            affine(q["fc1_w"][i], q["fc1_b"][i], _pad32(FC0_OUT * 2))
            affine(q["fc2_w"][i], q["fc2_b"][i], _pad32(FC1_OUT))


def _pad32(n):
    return (n + 31) // 32 * 32


# ---- integer reference (same arithmetic as the engine) ---------------------
def _tdiv(a, b):
    """C++ integer division (truncates toward zero)."""
    return np.where(a >= 0, a // b, -((-a) // b))


def reference_eval(q, rec):
    """Engine output (Value, side to move) for one record, in integers."""
    l1 = q["l1"]
    b = int(rec["bucket"])

    def acc(idx, n):
        idx = idx[:n].astype(np.int64)
        a = q["ft_b"].astype(np.int64) + q["ft_w"][idx].astype(np.int64).sum(0)
        p = q["psqt"][idx, b].astype(np.int64).sum()
        return a, p

    a_us, p_us = acc(rec["us"], rec["n_us"])
    a_th, p_th = acc(rec["them"], rec["n_them"])
    h = l1 // 2

    def pair(a):
        x = np.clip(a, 0, 127)
        return (x[:h] * x[h:]) // 128

    x0 = np.concatenate([pair(a_us), pair(a_th)])
    fc0 = q["fc0_w"][b].astype(np.int64) @ x0 + q["fc0_b"][b]
    sq = np.minimum(127, (fc0[:FC0_OUT] * fc0[:FC0_OUT]) >> (2 * WEIGHT_SCALE_BITS + 7))
    cr = np.clip(fc0[:FC0_OUT] >> WEIGHT_SCALE_BITS, 0, 127)
    x1 = np.concatenate([sq, cr])
    fc1 = q["fc1_w"][b].astype(np.int64) @ x1 + q["fc1_b"][b]
    x2 = np.clip(fc1 >> WEIGHT_SCALE_BITS, 0, 127)
    fc2 = int((q["fc2_w"][b].astype(np.int64) @ x2 + q["fc2_b"][b])[0])
    fwd = int(_tdiv(np.int64(fc0[FC0_OUT]) * (NNUE2SCORE * OUTPUT_SCALE), 127 * (1 << WEIGHT_SCALE_BITS)))
    psqt = int(_tdiv(np.int64(p_us - p_th), 2))
    return int(_tdiv(np.int64(psqt + fc2 + fwd), OUTPUT_SCALE))
