/*
  幻神揭棋引擎 (PikaJieQi), a UCI Jieqi engine derived from Pikafish / Stockfish.

  AB-JChess "HNN" network support (see abjnet.h).
  The network format, feature layout, layer stack selection and output
  post-processing follow AB-JChess hnnue.cpp (https://github.com/lxsgx23/AB-JChess,
  GPLv3), based on jqe, Copyright (c) 2026 Devin O'Keefe (MIT, see
  abjnet/LICENSE-jqe-MIT.txt). nnkern.h and jqnz.h are taken from AB-JChess.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "abjnet.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include "../misc.h"
#include "../position.h"
#include "jqnz.h"
#include "nnkern.h"

#if defined(NNUE_EMBEDDING)
// Defined in abjnet/embedded_net.S (outside the namespace, so that the
// universal build's ISA slices keep a private copy each).
extern "C" {
extern const unsigned char pikafish_embedded_abjnet[];
extern const unsigned char pikafish_embedded_abjnet_end[];
}
#endif

namespace Stockfish::ABJNet {

bool enabled = false;
int  scale   = 1000;

namespace {

constexpr int NUM_SQ = 90, NPSQT = 16, NSTACKS = 16, L1 = 16, L2 = 32, MAX_W = 1024;
constexpr int PIECE_FEATS = 16 * NUM_SQ;   // per king bucket
constexpr int URN_FEATS = 30;
constexpr int URN_OFF[6] = {0, 2, 4, 6, 8, 10};          // R N B A C P
constexpr int PIECE_VAL[6] = {943, 438, 168, 190, 491, 101};
constexpr int KB_OF_SCHEME[4] = {1, 3, 2, 6};
constexpr PieceType SLOT_PT[6] = {ROOK, KNIGHT, BISHOP, ADVISOR, CANNON, PAWN};

bool g_loaded = false;
unsigned g_gen = 0;
int g_W = 0, g_scheme = 0, g_nfeat = 0, g_urnBase = 0;
int16_t* g_ftW = nullptr;
int32_t* g_psqt = nullptr;
uint8_t* g_l1w = nullptr;
alignas(64) int16_t g_ftB[MAX_W];
alignas(64) int32_t g_l1b[NSTACKS][L1];
alignas(64) int32_t g_l2b[NSTACKS][L2];
alignas(64) int8_t  g_l2w[NSTACKS][L2 * L2];
int32_t g_fb[NSTACKS];
alignas(64) int8_t  g_fw[NSTACKS][L2];
int8_t g_kbTab[9], g_mirTab[9];
std::string g_desc;

void set_scheme(int scheme) {
    for (int r = 0; r < 3; ++r)
        for (int f = 3; f <= 5; ++f)
        {
            const int i = r * 3 + f - 3;
            int b = 0, m = 0;
            if (scheme == 1) b = f - 3;
            else if (scheme == 2) { m = f > 4; b = f == 4; }
            else if (scheme == 3) { m = f > 4; b = 2 * r + (f == 4); }
            g_kbTab[i] = int8_t(b);
            g_mirTab[i] = int8_t(m);
        }
}

inline int orient(int persp, int s) { return persp == WHITE ? s : (9 - s / 9) * 9 + s % 9; }

inline void king_state(int persp, int ksq, int& kb, int& mir) {
    const int o = orient(persp, ksq);
    const int r = std::min(o / 9, 2), f = std::clamp(o % 9, 3, 5);
    const int i = r * 3 + f - 3;
    kb = g_kbTab[i];
    mir = g_mirTab[i];
}

// Our piece types -> AB-JChess type - 1 (K R N B A C P, 7 = face-down)
inline int kind_of(Piece pc) {
    if (int(pc) & 16)
        return 7;
    switch (PieceType(int(pc) & 7))
    {
    case KING:    return 0;
    case ROOK:    return 1;
    case KNIGHT:  return 2;
    case BISHOP:  return 3;
    case ADVISOR: return 4;
    case CANNON:  return 5;
    default:      return 6; // PAWN
    }
}

inline Color owner_of(Piece pc) { return Color((int(pc) >> 3) & 1); }

void* g_mem = nullptr;

bool load_raw(const uint8_t* data, size_t size) {
    if (size < 40 || std::memcmp(data, "JQHNNUE1", 8) != 0)
        return false;
    uint32_t h[8];
    std::memcpy(h, data + 8, sizeof h);
    const int W = int(h[1]), scheme = int(h[3]);
    // Only version 1 (16/16/16/32 head) for now; version 3 heads are refused
    if (h[0] != 1 || W < 128 || W > MAX_W || W % 128 || h[3] > 3 || h[4] != NPSQT || h[5] != NSTACKS
        || int(h[6]) != L1 || int(h[7]) != L2)
        return false;
    const int nfeat = KB_OF_SCHEME[scheme] * PIECE_FEATS + URN_FEATS;
    if (int(h[2]) != nfeat)
        return false;
    const size_t headSz = 64 + size_t(W) * L1 + 128 + 1024 + 4 + 32;
    const size_t need = 40 + size_t(W) * 2 + size_t(nfeat) * W * 2 + size_t(nfeat) * NPSQT * 4 + NSTACKS * headSz;
    if (size != need)
        return false;

    const size_t szW = (size_t(nfeat) * W * 2 + 63) & ~size_t(63);
    const size_t szP = (size_t(nfeat) * NPSQT * 4 + 63) & ~size_t(63);
    const size_t szL = size_t(NSTACKS) * W * L1;
    void* mem = std_aligned_alloc(64, szW + szP + szL);
    if (!mem)
        return false;
    if (g_mem)
        std_aligned_free(g_mem);
    g_mem = mem;
    uint8_t* base = static_cast<uint8_t*>(mem);
    g_ftW  = reinterpret_cast<int16_t*>(base);
    g_psqt = reinterpret_cast<int32_t*>(base + szW);
    g_l1w  = base + szW + szP;

    const uint8_t* p = data + 40;
    std::memset(g_ftB, 0, sizeof g_ftB);
    std::memcpy(g_ftB, p, size_t(W) * 2); p += size_t(W) * 2;
    std::memcpy(g_ftW, p, size_t(nfeat) * W * 2); p += size_t(nfeat) * W * 2;
    std::memcpy(g_psqt, p, size_t(nfeat) * NPSQT * 4); p += size_t(nfeat) * NPSQT * 4;
    for (int s = 0; s < NSTACKS; ++s)
    {
        std::memcpy(g_l1b[s], p, 64); p += 64;
        std::memcpy(g_l1w + size_t(s) * W * L1, p, size_t(W) * L1); p += size_t(W) * L1;
        std::memcpy(g_l2b[s], p, 128); p += 128;
        std::memcpy(g_l2w[s], p, 1024); p += 1024;
        std::memcpy(&g_fb[s], p, 4); p += 4;
        std::memcpy(g_fw[s], p, 32); p += 32;
    }
    g_W = W;
    g_scheme = scheme;
    g_nfeat = nfeat;
    g_urnBase = KB_OF_SCHEME[scheme] * PIECE_FEATS;
    set_scheme(scheme);
    ++g_gen;
    g_loaded = true;
    g_desc = "AB-JChess HNN v1, FT " + std::to_string(W) + ", king buckets " + std::to_string(KB_OF_SCHEME[scheme]);
    return true;
}

bool load_bytes(const uint8_t* data, size_t n) {
    if (jqnz::is_compressed(data, n))
    {
        std::vector<uint8_t> raw;
        std::string err;
        if (!jqnz::decode(data, n, raw, &err))
            return false;
        return load_raw(raw.data(), raw.size());
    }
    return load_raw(data, n);
}

bool load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return !buf.empty() && load_bytes(buf.data(), buf.size());
}

std::string g_name;

// ---- features and the per-thread accumulator cache -------------------------

inline int piece_feat(int persp, Piece pc, int s, int kb, int mir) {
    int o = orient(persp, s);
    if (mir)
        o = (o / 9) * 9 + 8 - o % 9;
    return kb * PIECE_FEATS + ((owner_of(pc) == persp ? 0 : 8) + kind_of(pc)) * NUM_SQ + o;
}

inline int urn_feat(int persp, int owner, int slot, int j) { // urn[owner][slot] >= j, j >= 1
    return g_urnBase + (owner == persp ? 0 : 15) + URN_OFF[slot] + j - 1;
}

constexpr int MAX_F = 72;

struct FinnyEntry {
    alignas(64) int16_t v[MAX_W];
    alignas(64) int32_t ps[NPSQT];
    uint16_t f[MAX_F];
    int n = -1;
    unsigned gen = ~0u;
};

struct alignas(64) ThreadCache {
    FinnyEntry e[2 * 6 * 2];
};

ThreadCache* thread_cache() {
    static thread_local ThreadCache* tc = nullptr;
    if (!tc)
        tc = new ThreadCache();   // freed at exit by the OS (one per search thread)
    return tc;
}

// Brings the cache entry of (p, kb, mir) to the feature list f (n entries).
FinnyEntry* finny(ThreadCache* tc, int p, int kb, int mir, uint16_t* f, int n) {
    FinnyEntry* e = &tc->e[(p * 6 + kb) * 2 + mir];
    if (e->gen != g_gen) { e->n = -1; e->gen = g_gen; }
    std::sort(f, f + n);
    alignas(64) static const int32_t zero[NPSQT] = {};
    if (e->n < 0)
        nnk::apply(g_ftW, g_psqt, g_W, e->v, e->ps, g_ftB, zero, f, n, nullptr, 0);
    else
    {
        uint16_t add[MAX_F], rem[MAX_F];
        int na = 0, nr = 0, i = 0, j = 0;
        while (i < n || j < e->n)
        {
            if (j >= e->n || (i < n && f[i] < e->f[j])) add[na++] = f[i++];
            else if (i >= n || e->f[j] < f[i]) rem[nr++] = e->f[j++];
            else { ++i; ++j; }
        }
        // nnk::apply handles a bounded number of rows per call; split big diffs
        int ia = 0, ir = 0;
        while (ia < na || ir < nr)
        {
            const int ca = std::min(na - ia, 16), cr = std::min(nr - ir, 16);
            nnk::apply(g_ftW, g_psqt, g_W, e->v, e->ps, e->v, e->ps, add + ia, ca, rem + ir, cr);
            ia += ca; ir += cr;
        }
    }
    std::memcpy(e->f, f, sizeof(uint16_t) * size_t(n));
    e->n = n;
    return e;
}

} // namespace

bool loaded() { return g_loaded; }
std::string description() { return g_loaded ? g_desc + " (" + g_name + ")" : std::string("none"); }

bool init(const std::string& name) {
    const std::string file = name.empty() ? std::string(DefaultName) : name;
    if (g_loaded && file == g_name)
        return true;
    bool ok = false;
#if defined(NNUE_EMBEDDING)
    if (file == DefaultName)
        ok = load_bytes(pikafish_embedded_abjnet,
                        size_t(pikafish_embedded_abjnet_end - pikafish_embedded_abjnet));
#endif
    for (const std::string& dir : { std::string(""), CommandLine::binaryDirectory })
        if (!ok)
            ok = load_file(dir + file);
    if (ok)
        g_name = file;
    return ok;
}

int raw_value(const Position& pos) {
    ThreadCache* tc = thread_cache();
    const Color stm = pos.side_to_move();

    int material = 0, pieces = 0, dark = 0;
    for (int c = 0; c < 2; ++c)
        for (int sl = 0; sl < 6; ++sl)
            material += pos.bag_count(Color(c), SLOT_PT[sl]) * PIECE_VAL[sl];

    uint16_t f[2][MAX_F];
    int n[2] = {0, 0}, kb[2], mir[2];
    for (int p = 0; p < 2; ++p)
        king_state(p, int(pos.square<KING>(Color(p))), kb[p], mir[p]);

    Bitboard b = pos.pieces();
    while (b)
    {
        const Square s = pop_lsb(b);
        const Piece pc = pos.piece_on(s);
        ++pieces;
        const int k = kind_of(pc);
        if (k == 7) ++dark;
        else if (k >= 1 && k <= 5) material += PIECE_VAL[k - 1];
        for (int p = 0; p < 2; ++p)
            if (n[p] < MAX_F - URN_FEATS)
                f[p][n[p]++] = uint16_t(piece_feat(p, pc, int(s), kb[p], mir[p]));
    }
    for (int c = 0; c < 2; ++c)
        for (int sl = 0; sl < 6; ++sl)
        {
            const int cnt = std::min(pos.bag_count(Color(c), SLOT_PT[sl]), sl == 5 ? 5 : 2);
            for (int j = 1; j <= cnt; ++j)
                for (int p = 0; p < 2; ++p)
                    f[p][n[p]++] = uint16_t(urn_feat(p, c, sl, j));
        }

    FinnyEntry* us   = finny(tc, stm,  kb[stm],  mir[stm],  f[stm],  n[stm]);
    FinnyEntry* them = finny(tc, ~stm, kb[~stm], mir[~stm], f[~stm], n[~stm]);

    const int bucket = std::min((pieces - 1) / 4, 7) + (dark > 0 ? 8 : 0);
    alignas(64) uint8_t x[MAX_W];
#ifdef JQ_PAIRWISE_NNZ
    alignas(32) uint16_t nnz[MAX_W / 4 + 16];
    const int nnzN = nnk::pairwise_nnz(us->v, them->v, g_W, x, nnz);
#else
    const uint16_t* nnz = nullptr;
    const int nnzN = 0;
    nnk::pairwise(us->v, them->v, g_W, x);
#endif
    const int psqt = (us->ps[bucket] - them->ps[bucket]) / 2;
    const int positional = nnk::propagate(g_W, g_l1b[bucket], g_l1w + size_t(bucket) * size_t(g_W) * L1,
                                          g_l2b[bucket], g_l2w[bucket], g_fb[bucket], g_fw[bucket], x, nnz, nnzN);
    const int nn = (psqt + positional) / 16;
    const int materialScale = 935 + material * 93 / 5116;
    int value = (nn * materialScale) / 1024;
    value = value * (218 - std::min(pos.no_capture(), 120)) / 176;
    return std::clamp(value, -29507, 29507);
}

Value evaluate(const Position& pos) {
    const int64_t v = int64_t(raw_value(pos)) * scale;
    const int64_t s = v >= 0 ? (v + 500) / 1000 : -((-v + 500) / 1000);
    return Value(std::clamp<int64_t>(s, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1));
}

} // namespace Stockfish::ABJNet
