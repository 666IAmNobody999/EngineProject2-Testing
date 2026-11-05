/*
  幻神揭棋引擎 (PikaJieQi), a UCI Jieqi engine derived from Pikafish / Stockfish.

  AB-JChess "HNN" network support.
  Contains code derived from AB-JChess (https://github.com/lxsgx23/AB-JChess,
  GPLv3), which in turn is based on jqe, Copyright (c) 2026 Devin O'Keefe
  (MIT licence, see abjnet/LICENSE-jqe-MIT.txt) and on Pikafish / Stockfish.

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

// Evaluation with an AB-JChess HNN net ("JQHNNUE1", raw or JQHNNL01/JQHNNZ01
// compressed, e.g. abjchess-20261030.nnue). The engine ships it as
// pikafish.nnue, embedded in the binary when the file is present at build time.
//
// Inputs per perspective: king bucket x 16 piece kinds (own/opponent x K R N B
// A C P face-down) x 90 squares plus 30 bag thresholds; the accumulator is kept
// per thread in a small cache keyed by (perspective, king bucket, mirror) and
// updated by the difference of the sorted feature lists, so no changes to
// do_move / getDark are needed.

#ifndef ABJNET_H_INCLUDED
#define ABJNET_H_INCLUDED

#include <string>

#include "../types.h"

namespace Stockfish {

class Position;

namespace ABJNet {

constexpr const char* DefaultName = "pikafish.nnue";

// Loads the net named by the option (embedded copy first when the name is the
// default, then the working and engine directories). Returns true on success.
bool init(const std::string& name);
bool loaded();
std::string description();

extern bool enabled;   // "ABJ NNUE" option and a net is loaded
extern int  scale;     // permille, "ABJScale"

// Raw AB-JChess value (side to move), before our scale.
int  raw_value(const Position& pos);
Value evaluate(const Position& pos);

} // namespace ABJNet
} // namespace Stockfish

#endif
