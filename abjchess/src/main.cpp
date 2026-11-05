/*
  ABJchess (AB JChess), a UCI Jieqi engine.
  Contains code derived from AB-JChess / Pikafish / Stockfish:
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)
  Based on jqe, Copyright (c) 2026 Devin O'Keefe (MIT licence, see abjchess/LICENSE-jqe-MIT.txt).
  See THIRD_PARTY_NOTICES.md at the repository root.

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
#include <iostream>
#include "winperf.h"
#if defined(_WIN32)
// 幻神揭棋引擎: UTF-8 console so the Chinese engine name / paths are not garbled (declared here instead of <windows.h>)
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
extern "C" __declspec(dllimport) int __stdcall SetConsoleCP(unsigned int);
#endif

namespace jq {
int uci_loop(int argc, char** argv);
}

int main(int argc, char** argv) {
    // 幻神揭棋引擎 (ported from 幻神): a helper started by acquire_power_plan() only waits for
    // its engine to exit and then restores the previous power plan.
    if (jq::WinPerf::run_power_watcher(argc, argv)) return 0;
#if defined(_WIN32)
    SetConsoleOutputCP(65001u);
    SetConsoleCP(65001u);
#endif
    std::ios::sync_with_stdio(false);
    // Windows only (no-ops elsewhere): high CPU / I/O priority, other Pikafish instances to Idle,
    // the "幻神终极效能" power plan while the engine runs.
    jq::WinPerf::apply_process_tuning();
    jq::WinPerf::idle_other_pikafish();
    jq::WinPerf::acquire_power_plan();
    const int rc = jq::uci_loop(argc, argv);
    jq::WinPerf::release_power_plan();
    return rc;
}
