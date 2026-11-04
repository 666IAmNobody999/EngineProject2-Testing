/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of input features HalfKAv2_hm of NNUE evaluation function

#include "half_ka_v2_hm.h"

#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {

  // Index of a feature for a given king position and another piece on some square
  template<Color Perspective>
  inline IndexType HalfKAv2_hm::make_index(Square s, Piece pc, Square ksq) {
    s = Square( Perspective == BLACK ? Rotate[s] : s);
    s = Square(KingBuckets[ksq] >> 3 ? Mirror[s] : s);
    return IndexType(s + PieceSquareIndex[Perspective][pc] + PS_NB * (KingBuckets[ksq] & 0x7));
  }

  // Get a list of indices for active features
  template<Color Perspective>
  void HalfKAv2_hm::append_active_indices(
    const Position& pos,
    IndexList& active
  ) {
    Square ksq = pos.square<KING>(Perspective);
    Bitboard bb = pos.pieces();
    while (bb)
    {
      Square s = pop_lsb(bb);
      active.push_back(make_index<Perspective>(s, pos.piece_on(s), ksq));
    }

    for (Color c : { WHITE, BLACK })
      for (PieceType t = ROOK; t <= BISHOP; ++t)
      {
        int n = std::min(pos.bag_count(c, t), BagMax[t]);
        for (int k = 0; k < n; ++k)
          active.push_back(make_bag_index<Perspective>(c, t, k));
      }
  }

  // Explicit template instantiations
  template void HalfKAv2_hm::append_active_indices<WHITE>(const Position& pos, IndexList& active);
  template void HalfKAv2_hm::append_active_indices<BLACK>(const Position& pos, IndexList& active);

  // append_changed_indices() : get a list of indices for recently changed features
  template<Color Perspective>
  void HalfKAv2_hm::append_changed_indices(
    Square ksq,
    const DirtyPiece& dp,
    IndexList& removed,
    IndexList& added
  ) {
    for (int i = 0; i < dp.dirty_num; ++i) {
      if (dp.from[i] != SQ_NONE)
        removed.push_back(make_index<Perspective>(dp.from[i], dp.piece[i], ksq));
      if (dp.to[i] != SQ_NONE)
        added.push_back(make_index<Perspective>(dp.to[i], dp.piece[i], ksq));
    }
    // A reveal took one piece out of a bag: the count went from n + 1 to n,
    // so the thermometer feature k = n switches off (if it was within range).
    for (int i = 0; i < dp.bagNum; ++i)
      if (dp.bagCount[i] < BagMax[dp.bagType[i]])
        removed.push_back(make_bag_index<Perspective>(dp.bagColor[i], dp.bagType[i], dp.bagCount[i]));
  }

  // Explicit template instantiations
  template void HalfKAv2_hm::append_changed_indices<WHITE>(Square ksq, const DirtyPiece& dp, IndexList& removed, IndexList& added);
  template void HalfKAv2_hm::append_changed_indices<BLACK>(Square ksq, const DirtyPiece& dp, IndexList& removed, IndexList& added);

  int HalfKAv2_hm::update_cost(const StateInfo* st) {
    return st->dirtyPiece.dirty_num + st->dirtyPiece.bagNum;
  }

  int HalfKAv2_hm::refresh_cost(const Position& pos) {
    return pos.count<ALL_PIECES>() + pos.rest_count(WHITE) + pos.rest_count(BLACK);
  }

  bool HalfKAv2_hm::requires_refresh(const StateInfo* st, Color perspective) {
    return st->dirtyPiece.refresh || st->dirtyPiece.piece[0] == make_piece(perspective, KING);
  }

}  // namespace Stockfish::Eval::NNUE::Features
