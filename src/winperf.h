/*
  Pikafish, a UCI chess playing engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Pikafish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Pikafish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WINPERF_H_INCLUDED
#define WINPERF_H_INCLUDED

#include <cstddef>
#include <vector>

// Windows-only process tuning. Every function is a no-op on other platforms.
//
//  * apply_process_tuning(): CPU priority "High", I/O priority "High",
//    memory priority "Normal" and dynamic thread priority boost enabled.
//    Cheap enough to be re-applied before every search (GUIs may reset them).
//  * acquire_power_plan() / release_power_plan(): switch Windows to the
//    engine's "幻神终极效能" power plan (Bitsum Highest Performance settings:
//    core parking off, processor state 100%, P-cores preferred) while at least
//    one engine instance is running and restore the user's previous plan when the last one exits.
//    A tiny watcher process at CPU priority "Idle" (the same executable started
//    with --pikafish-power-watch <pid>) restores the plan even if the GUI kills the
//    engine instead of sending "quit".
//  * idle_other_pikafish(): sets every other process of the same user whose
//    executable name contains "pikafish" (case-insensitive) and that is not
//    this engine's own executable to CPU priority "Idle".
//  * plan_physical_core_pins(n): when n search threads fit on the physical
//    cores, one logical processor (global index group * 64 + bit) per thread,
//    each on a different physical core: performance cores first (hybrid CPUs),
//    the core of logical processor 0 last (it serves most interrupts and OS
//    work), only processors in the process affinity. Empty otherwise (more
//    threads than physical cores, or not Windows).
//  * pin_current_thread(cpu): hard-binds the calling thread to that processor.
//  * run_power_watcher(): returns true when this process is such a watcher;
//    main() must then return immediately.

namespace Stockfish::WinPerf {

void apply_process_tuning();
void acquire_power_plan();
void release_power_plan();
bool run_power_watcher(int argc, char* argv[]);
void idle_other_pikafish();
std::vector<int> plan_physical_core_pins(std::size_t threads);
bool             pin_current_thread(int cpu);

// Platform-independent part of plan_physical_core_pins() (unit-testable).
struct PhysicalCore {
    int  efficiencyClass;  // higher = faster (P-core > E-core)
    int  firstCpu;         // first allowed logical processor, group * 64 + bit
    bool hasCpu0;          // contains logical processor 0 of group 0
};
std::vector<int> select_physical_cores(std::vector<PhysicalCore> cores, std::size_t threads);

}  // namespace Stockfish::WinPerf

#endif  // #ifndef WINPERF_H_INCLUDED
