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

#include <iostream>

#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "winperf.h"

using namespace Stockfish;

int main(int argc, char* argv[]) {

  // Windows (ported from 幻神): a helper started by acquire_power_plan() only
  // waits for its engine to exit and then restores the previous power plan.
  if (WinPerf::run_power_watcher(argc, argv))
      return 0;

  std::cout << engine_info() << std::endl;

  // Windows: CPU/IO priority "High" for this process, idle other engines and
  // switch to the high-performance power plan while the engine runs.
  WinPerf::apply_process_tuning();
  WinPerf::idle_other_pikafish();
  WinPerf::acquire_power_plan();

  CommandLine::init(argc, argv);
  UCI::init(Options);
  Tune::init();
  PSQT::init();
  Bitboards::init();
  Position::init();
  Threads.set(size_t(Options["Threads"]));
  Search::clear(); // After threads are up
  Eval::NNUE::init();

  UCI::loop(argc, argv);

  Threads.set(0);
  WinPerf::release_power_plan();
  return 0;
}
