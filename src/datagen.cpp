/*
  幻神揭棋引擎 (PikaJieQi) - NNUE training data generation.

  UCI command (one process = one search thread; run several processes in
  parallel, see tools/nnue/gendata.py):

    gennnue out=data.bin [positions=1000000] [depth=8] [nodes=0] [random=6]
            [seed=1] [evallimit=3000] [adjudicate=2500] [maxply=400]
            [minply=8] [fen=data.fen] [book=openings.fen]

  The engine plays games against itself. Every game deals the hidden pieces of
  both sides at random (the "true" identities), so the reveals during the game
  are as random as over the board. Each recorded position stores the NNUE
  feature indices of both perspectives (side to move first), the search score
  and the game result, so the trainer needs no 揭棋 rules at all.

  Record layout (little endian, 264 bytes, see tools/nnue/jqdata.py):
    int16  score    search score, side to move, engine units
    int8   result   1 win / 0 draw / -1 loss for the side to move
    uint8  stm      0 white, 1 black
    uint8  bucket   layer stack bucket, (pieces on board - 1) / 4
    uint8  nUs, nThem  number of used indices below
    uint8  reserved
    uint16 us[64], them[64]   feature indices, unused = 0xFFFF
*/

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "nnue/nnue_architecture.h"

namespace Stockfish {

namespace {

using FeatureSet = Eval::NNUE::FeatureSet;
constexpr int MaxIdx = 64;
static_assert(FeatureSet::MaxActiveDimensions <= MaxIdx);
static_assert(FeatureSet::Dimensions < 0xFFFF);

#pragma pack(push, 1)
struct Record {
  int16_t  score;
  int8_t   result;
  uint8_t  stm;
  uint8_t  bucket;
  uint8_t  nUs;
  uint8_t  nThem;
  uint8_t  reserved;
  uint16_t us[MaxIdx];
  uint16_t them[MaxIdx];
};
#pragma pack(pop)
static_assert(sizeof(Record) == 264);

const char* StartFEN = "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1";

template<Color P>
int fill(const Position& pos, uint16_t* out) {
  FeatureSet::IndexList list;
  FeatureSet::append_active_indices<P>(pos, list);
  int n = 0;
  for (auto i : list)
      out[n++] = uint16_t(i);
  for (int k = n; k < MaxIdx; ++k)
      out[k] = 0xFFFF;
  return n;
}

void make_record(const Position& pos, Value score, Record& r) {
  std::memset(&r, 0, sizeof(r));
  r.score  = int16_t(std::clamp(int(score), -32000, 32000));
  r.stm    = uint8_t(pos.side_to_move());
  r.bucket = uint8_t((pos.count<ALL_PIECES>() - 1) / 4);
  if (pos.side_to_move() == WHITE)
      r.nUs = fill<WHITE>(pos, r.us), r.nThem = fill<BLACK>(pos, r.them);
  else
      r.nUs = fill<BLACK>(pos, r.us), r.nThem = fill<WHITE>(pos, r.them);
}

// Rebuild the position of the game from the start FEN and the moves played
// (the moves carry the revealed / captured hidden identities, as from a GUI).
void setup(Position& pos, StateListPtr& states, const std::string& fen, const std::vector<Move>& moves) {
  states = StateListPtr(new std::deque<StateInfo>(1));
  pos.set(fen, &states->back(), Threads.main());
  for (Move m : moves)
  {
      states->emplace_back();
      pos.do_move_temp(m, states->back(), type_of(get_Piece(m)));
  }
}

bool can_attack(const Position& pos, Color c) {
  return pos.pieces(c, ROOK) | pos.pieces(c, KNIGHT) | pos.pieces(c, CANNON) | pos.pieces(c, PAWN)
      || pos.rest_count(c) > 0;
}

std::string kv(const std::string& args, const std::string& key, const std::string& def) {
  std::istringstream is(args);
  std::string tok;
  while (is >> tok)
      if (tok.rfind(key + "=", 0) == 0)
          return tok.substr(key.size() + 1);
  return def;
}

} // namespace

namespace Datagen {

void run(std::istream& is) {

  std::string args, tok;
  while (is >> tok)
      args += tok + " ";

  const std::string outName = kv(args, "out", "jieqi_data.bin");
  const std::string fenName = kv(args, "fen", "");
  const std::string bookName = kv(args, "book", "");
  const int64_t positions = std::stoll(kv(args, "positions", "1000000"));
  const int depth       = std::stoi(kv(args, "depth", "8"));
  const int64_t nodes   = std::stoll(kv(args, "nodes", "0"));
  const int randomPlies = std::stoi(kv(args, "random", "6"));
  const int evalLimit   = std::stoi(kv(args, "evallimit", "3000"));
  const int adjudicate  = std::stoi(kv(args, "adjudicate", "2500"));
  const int maxPly      = std::stoi(kv(args, "maxply", "400"));
  const int minPly      = std::stoi(kv(args, "minply", "8"));
  uint64_t seed         = std::stoull(kv(args, "seed", "1"));
  PRNG rng(seed ? seed * 0x9E3779B97F4A7C15ULL : 1);

  std::vector<std::string> book;
  if (!bookName.empty())
  {
      std::ifstream bf(bookName);
      std::string line;
      while (std::getline(bf, line))
          if (!line.empty() && Position::validate_fen(line).empty())
              book.push_back(line);
  }

  std::ofstream out(outName, std::ios::binary | std::ios::app);
  std::ofstream fenOut;
  if (!fenName.empty())
      fenOut.open(fenName, std::ios::app);
  if (!out)
  {
      sync_cout << "info string gennnue: cannot open " << outName << sync_endl;
      return;
  }

  sync_cout << "info string gennnue: out=" << outName << " positions=" << positions
            << " depth=" << depth << " nodes=" << nodes << " random=" << randomPlies
            << " seed=" << seed << " features=" << FeatureSet::Name << sync_endl;

  Search::Silent = true;
  Position pos;
  StateListPtr states;
  int64_t written = 0, games = 0;
  int wins[3] = {0, 0, 0};
  const auto t0 = std::chrono::steady_clock::now();

  while (written < positions)
  {
      const std::string fen = book.empty() ? std::string(StartFEN)
                                           : book[rng.rand<uint64_t>() % book.size()];
      std::vector<Move> moves;
      setup(pos, states, fen, moves);

      // Deal the true identities of the hidden pieces of both sides
      Piece truth[SQUARE_NB];
      std::fill(truth, truth + SQUARE_NB, NO_PIECE);
      for (Color c : { WHITE, BLACK })
      {
          std::vector<Piece> bag;
          for (PieceType t = ROOK; t <= BISHOP; ++t)
              for (int k = 0; k < pos.bag_count(c, t); ++k)
                  bag.push_back(make_piece(c, t));
          for (size_t i = bag.size(); i > 1; --i)
              std::swap(bag[i - 1], bag[rng.rand<uint64_t>() % i]);
          size_t n = 0;
          for (Square s = SQ_A0; s < SQUARE_NB; ++s)
              if (pos.piece_on(s) != NO_PIECE && color_of(pos.piece_on(s)) == c && pos.isDark(s) && n < bag.size())
                  truth[s] = bag[n++];
      }

      Search::clear();
      std::vector<Record> recs;
      std::vector<Key> keys;
      int result = 0;          // from white's point of view: 1, 0, -1
      int adjCount = 0, quiet = 0;

      for (int ply = 0; ; ++ply)
      {
          setup(pos, states, fen, moves);
          const Color us = pos.side_to_move();
          const int sgn = us == WHITE ? 1 : -1;

          MoveList<LEGAL> legal(pos);
          if (legal.size() == 0) { result = -sgn; break; }   // no move: side to move loses
          if (ply >= maxPly || quiet >= 120 || (!can_attack(pos, WHITE) && !can_attack(pos, BLACK)))
              { result = 0; break; }
          if (std::count(keys.begin(), keys.end(), pos.key()) >= 2)
              { result = 0; break; }
          keys.push_back(pos.key());

          Move m;
          if (ply < randomPlies)
              m = *(legal.begin() + rng.rand<uint64_t>() % legal.size());
          else
          {
              const bool inCheck = bool(pos.checkers());
              Search::LimitsType limits;
              limits.startTime = now();
              limits.depth = nodes ? 0 : depth;
              limits.nodes = nodes;
              Threads.start_thinking(pos, states, limits);
              Threads.main()->wait_for_search_finished();

              const auto& rm = Threads.main()->rootMoves[0];
              m = rm.pv[0];
              const Value v = rm.score;
              if (m == MOVE_NONE) { result = -sgn; break; }

              // Positions are rebuilt below from the move list, so pos is still
              // the root here only through its moved states: rebuild first.
              setup(pos, states, fen, moves);

              if (v >= VALUE_MATE_IN_MAX_PLY)  { result =  sgn; break; }
              if (v <= -VALUE_MATE_IN_MAX_PLY) { result = -sgn; break; }

              adjCount = std::abs(v) >= adjudicate ? adjCount + 1 : 0;
              if (adjCount >= 8) { result = v > 0 ? sgn : -sgn; break; }

              // Keep quiet positions: no check, best move not a capture,
              // score not decided. (Stockfish's "smart fen skipping".)
              if (   ply >= minPly
                  && !inCheck
                  && pos.piece_on(to_sq(m)) == NO_PIECE
                  && std::abs(v) < evalLimit)
              {
                  Record r;
                  make_record(pos, v, r);
                  recs.push_back(r);
                  if (fenOut.is_open())
                      fenOut << pos.fen() << " | " << int(v) << "\n";
              }
          }

          // Identities carried by the move: the revealed piece (moving a hidden
          // piece) and the captured hidden piece.
          const Square from = from_sq(m), to = to_sq(m);
          Piece pGet = pos.isDark(from) ? truth[from] : NO_PIECE;
          Piece pCap = pos.piece_on(to) != NO_PIECE && pos.isDark(to) ? truth[to] : NO_PIECE;
          quiet = (pGet || pos.piece_on(to) != NO_PIECE) ? 0 : quiet + 1;
          truth[from] = truth[to] = NO_PIECE;
          moves.push_back(make_move(m, pGet, pCap));
      }

      // Game result for each record, from its side to move
      for (auto& r : recs)
          r.result = int8_t(r.stm == WHITE ? result : -result);
      const int64_t take = std::min<int64_t>(int64_t(recs.size()), positions - written);
      out.write(reinterpret_cast<const char*>(recs.data()), std::streamsize(take * sizeof(Record)));
      out.flush();
      written += take;
      games++;
      wins[result + 1]++;

      if (games % 10 == 0 || written >= positions)
      {
          double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          sync_cout << "info string gennnue: games " << games << " positions " << written
                    << " (" << int(written / std::max(sec, 0.001)) << "/s)"
                    << " white wins " << wins[2] << " draws " << wins[1] << " black wins " << wins[0]
                    << sync_endl;
      }
  }

  Search::Silent = false;
  sync_cout << "info string gennnue: done, " << written << " positions in " << games << " games" << sync_endl;
}

} // namespace Datagen

} // namespace Stockfish
