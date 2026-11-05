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
// 幻神揭棋引擎 abj-base: ported from 幻神 (Pikafish fork, GPLv3) - process tuning, the "幻神终极效能"
// power plan, idling other Pikafish instances and physical-core pinning. No effect on search results.

#include "winperf.h"

#include <algorithm>

namespace jq::WinPerf {

std::vector<int> select_physical_cores(std::vector<PhysicalCore> cores, std::size_t threads) {
    if (threads == 0 || threads > cores.size())
        return {};

    std::stable_sort(cores.begin(), cores.end(), [](const PhysicalCore& x, const PhysicalCore& y) {
        if (x.efficiencyClass != y.efficiencyClass)
            return x.efficiencyClass > y.efficiencyClass;
        if (x.hasCpu0 != y.hasCpu0)
            return y.hasCpu0;
        return x.firstCpu < y.firstCpu;
    });

    std::vector<int> pins;
    for (std::size_t i = 0; i < threads; ++i)
        pins.push_back(cores[i].firstCpu);
    return pins;
}

}  // namespace jq::WinPerf

#if !defined(_WIN32)

namespace jq::WinPerf {

void apply_process_tuning() {}
void acquire_power_plan() {}
void release_power_plan() {}
bool run_power_watcher(int, char*[]) { return false; }
void idle_other_pikafish() {}
std::vector<int> plan_physical_core_pins(std::size_t) { return {}; }
bool             pin_current_thread(int) { return false; }

}  // namespace jq::WinPerf

#else

    #if !defined(NOMINMAX)
        #define NOMINMAX
    #endif
    #if !defined(WIN32_LEAN_AND_MEAN)
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <tlhelp32.h>

    #include <atomic>
    #include <cwctype>
    #include <cstdint>
    #include <cstdlib>
    #include <cstring>
    #include <fstream>
    #include <string>
    #include <vector>

namespace jq::WinPerf {

namespace {

// All system calls below are resolved at run time, so the engine still starts on
// Windows versions (or SDKs) that lack one of them, and no extra import library
// is needed for any toolchain. A failing call is silently ignored.
template<typename F>
F load(const wchar_t* dll, const char* name) {
    HMODULE mod = GetModuleHandleW(dll);
    if (!mod)
        mod = LoadLibraryW(dll);
    F       fn   = nullptr;
    FARPROC proc = mod ? GetProcAddress(mod, name) : nullptr;
    if (proc)
        std::memcpy(&fn, &proc, sizeof(fn));
    return fn;
}

// ---------------------------------------------------------------------------
// Process tuning: I/O priority, memory priority, dynamic priority boost
// ---------------------------------------------------------------------------

using NtSetInformationProcessFn = LONG(WINAPI*)(HANDLE, ULONG, PVOID, ULONG);
using SetProcessInformationFn   = BOOL(WINAPI*)(HANDLE, int, LPVOID, DWORD);

constexpr ULONG ProcessIoPriority     = 33;  // PROCESSINFOCLASS::ProcessIoPriority
constexpr ULONG IoPriorityNormal      = 2;
constexpr ULONG IoPriorityHigh        = 3;
constexpr int   ProcessMemoryPriority = 0;  // PROCESS_INFORMATION_CLASS::ProcessMemoryPriority
constexpr ULONG MemoryPriorityNormal  = 5;  // MEMORY_PRIORITY_NORMAL

bool enable_privilege(const wchar_t* name) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
        return false;

    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount           = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueW(nullptr, name, &tp.Privileges[0].Luid)
           && AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr)
           && GetLastError() == ERROR_SUCCESS;  // ERROR_NOT_ALL_ASSIGNED: not held
    CloseHandle(token);
    return ok;
}

// ---------------------------------------------------------------------------
// Power plan
// ---------------------------------------------------------------------------

using PowerGetActiveSchemeFn  = DWORD(WINAPI*)(HKEY, GUID**);
using PowerSetActiveSchemeFn  = DWORD(WINAPI*)(HKEY, const GUID*);
using PowerEnumerateFn        = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, int, ULONG, UCHAR*, DWORD*);
using PowerReadFriendlyNameFn = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, UCHAR*, DWORD*);
using PowerDuplicateSchemeFn  = DWORD(WINAPI*)(HKEY, const GUID*, GUID**);
using PowerWriteFriendlyNameFn = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, UCHAR*, DWORD);
using PowerWriteValueIndexFn   = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, DWORD);

constexpr int  AccessScheme = 16;  // POWER_DATA_ACCESSOR::ACCESS_SCHEME
constexpr GUID UltimateGuid = {0xe9a42b02, 0xd5df, 0x448d, {0xaa, 0x00, 0x03, 0xf1, 0x47, 0x49, 0xeb, 0x61}};
constexpr GUID HighPerfGuid = {0x8c5e7fda, 0xe8bf, 0x4a96, {0x9a, 0x85, 0xa6, 0xe2, 0x3a, 0x8c, 0x63, 0x5c}};

// Name of the engine's own power plan.
constexpr wchar_t PlanName[] = L"\u5E7B\u795E\u7EC8\u6781\u6548\u80FD";  // 幻神终极效能
// Bitsum Highest Performance (installed by Process Lasso / ParkControl) is
// copied when present; it is never modified or renamed, so Process Lasso keeps
// finding it.
constexpr wchar_t BitsumName[] = L"Bitsum Highest Performance";

// Processor settings of Bitsum Highest Performance, applied to the plan on
// both AC and DC (https://bitsum.com/bhp/: core parking off, CPU never below
// its base frequency; Process Lasso changelog: prefer P-cores for short and
// long running threads). Settings missing on a CPU (e.g. no hybrid cores) are
// skipped by Windows.
constexpr GUID SubProcessor = {0x54533251, 0x82be, 0x4824, {0x96, 0xc1, 0x47, 0xb6, 0x0b, 0x74, 0x0d, 0x00}};
struct PlanSetting {
    GUID  setting;
    DWORD value;
};
constexpr PlanSetting PlanSettings[] = {
  {{0x0cc5b647, 0xc1df, 0x4637, {0x89, 0x1a, 0xde, 0xc3, 0x5c, 0x31, 0x85, 0x84}}, 100},  // CPMINCORES
  {{0x0cc5b647, 0xc1df, 0x4637, {0x89, 0x1a, 0xde, 0xc3, 0x5c, 0x31, 0x85, 0x85}}, 100},  // CPMINCORES1
  {{0xea062031, 0x0e34, 0x4ff1, {0x9b, 0x6d, 0xeb, 0x10, 0x59, 0x33, 0x40, 0x28}}, 100},  // CPMAXCORES
  {{0xea062031, 0x0e34, 0x4ff1, {0x9b, 0x6d, 0xeb, 0x10, 0x59, 0x33, 0x40, 0x29}}, 100},  // CPMAXCORES1
  {{0x893dee8e, 0x2bef, 0x41e0, {0x89, 0xc6, 0xb5, 0x5d, 0x09, 0x29, 0x96, 0x4c}}, 100},  // PROCTHROTTLEMIN
  {{0x893dee8e, 0x2bef, 0x41e0, {0x89, 0xc6, 0xb5, 0x5d, 0x09, 0x29, 0x96, 0x4d}}, 100},  // PROCTHROTTLEMIN1
  {{0xbc5038f7, 0x23e0, 0x4960, {0x96, 0xda, 0x33, 0xab, 0xaf, 0x59, 0x35, 0xec}}, 100},  // PROCTHROTTLEMAX
  {{0xbc5038f7, 0x23e0, 0x4960, {0x96, 0xda, 0x33, 0xab, 0xaf, 0x59, 0x35, 0xed}}, 100},  // PROCTHROTTLEMAX1
  {{0x93b8b6dc, 0x0698, 0x4d1c, {0x9e, 0xe4, 0x06, 0x44, 0xe9, 0x00, 0xc8, 0x5d}}, 2},    // SCHEDPOLICY: prefer performant
  {{0xbae08b81, 0x2d5e, 0x4688, {0xad, 0x6a, 0x13, 0x24, 0x33, 0x56, 0x65, 0x4b}}, 2},    // SHORTSCHEDPOLICY
};
constexpr char WatchSwitch[] = "--pikafish-power-watch";

struct PowerApi {
    PowerGetActiveSchemeFn  getActive = load<PowerGetActiveSchemeFn>(L"powrprof.dll", "PowerGetActiveScheme");
    PowerSetActiveSchemeFn  setActive = load<PowerSetActiveSchemeFn>(L"powrprof.dll", "PowerSetActiveScheme");
    PowerEnumerateFn        enumerate = load<PowerEnumerateFn>(L"powrprof.dll", "PowerEnumerate");
    PowerReadFriendlyNameFn readName  = load<PowerReadFriendlyNameFn>(L"powrprof.dll", "PowerReadFriendlyName");
    PowerDuplicateSchemeFn  duplicate = load<PowerDuplicateSchemeFn>(L"powrprof.dll", "PowerDuplicateScheme");
    PowerWriteFriendlyNameFn writeName = load<PowerWriteFriendlyNameFn>(L"powrprof.dll", "PowerWriteFriendlyName");
    PowerWriteValueIndexFn   writeAC   = load<PowerWriteValueIndexFn>(L"powrprof.dll", "PowerWriteACValueIndex");
    PowerWriteValueIndexFn   writeDC   = load<PowerWriteValueIndexFn>(L"powrprof.dll", "PowerWriteDCValueIndex");

    bool ok() const { return getActive && setActive && enumerate; }
};

PowerApi& power() {
    static PowerApi api;
    return api;
}

bool same(const GUID& a, const GUID& b) { return std::memcmp(&a, &b, sizeof(GUID)) == 0; }

bool active_scheme(GUID& out) {
    GUID* p = nullptr;
    if (power().getActive(nullptr, &p) != ERROR_SUCCESS || !p)
        return false;
    out = *p;
    LocalFree(p);
    return true;
}

std::wstring friendly_name(const GUID& g) {
    if (!power().readName)
        return {};
    DWORD size = 0;
    DWORD rc   = power().readName(nullptr, &g, nullptr, nullptr, nullptr, &size);
    if ((rc != ERROR_SUCCESS && rc != ERROR_MORE_DATA) || size < sizeof(wchar_t))
        return {};
    std::vector<UCHAR> buf(size + sizeof(wchar_t), 0);
    if (power().readName(nullptr, &g, nullptr, nullptr, buf.data(), &size) != ERROR_SUCCESS)
        return {};
    return std::wstring(reinterpret_cast<const wchar_t*>(buf.data()));
}

std::wstring data_dir() {
    wchar_t buf[MAX_PATH];
    DWORD   n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH)
    {
        n = GetTempPathW(MAX_PATH, buf);
        if (n == 0 || n >= MAX_PATH)
            return {};
    }
    std::wstring dir(buf);
    if (!dir.empty() && dir.back() != L'\\')
        dir += L'\\';
    dir += L"Pikafish";
    CreateDirectoryW(dir.c_str(), nullptr);
    return dir + L'\\';
}

// GUID of the plan created by the engine (kept so that it is created only once).
std::wstring created_scheme_file() { return data_dir() + L"power_plan.guid"; }
std::wstring state_file() { return data_dir() + L"powerplan.state"; }

template<typename Fn>
bool for_each_scheme(Fn fn) {
    for (ULONG i = 0;; ++i)
    {
        GUID  s{};
        DWORD size = sizeof(s);
        if (power().enumerate(nullptr, nullptr, nullptr, AccessScheme, i, reinterpret_cast<UCHAR*>(&s), &size)
            != ERROR_SUCCESS)
            return false;
        if (fn(s))
            return true;
    }
}

bool scheme_exists(const GUID& g) {
    return for_each_scheme([&](const GUID& s) { return same(s, g); });
}

bool scheme_named(const wchar_t* name, GUID& out) {
    return for_each_scheme([&](const GUID& s) {
        if (friendly_name(s) != name)
            return false;
        out = s;
        return true;
    });
}

bool duplicate_scheme(const GUID& source, GUID& out) {
    GUID* dup = nullptr;
    if (!power().duplicate || power().duplicate(nullptr, &source, &dup) != ERROR_SUCCESS || !dup)
        return false;
    out = *dup;
    LocalFree(dup);
    return true;
}

// Applies the Bitsum Highest Performance processor settings (idempotent).
void configure_plan(const GUID& plan) {
    for (const auto& ps : PlanSettings)
    {
        if (power().writeAC)
            power().writeAC(nullptr, &plan, &SubProcessor, &ps.setting, ps.value);
        if (power().writeDC)
            power().writeDC(nullptr, &plan, &SubProcessor, &ps.setting, ps.value);
    }
}

// Finds the engine's "幻神终极效能" plan, creating it once when missing: a copy
// of Bitsum Highest Performance if installed, else of the (possibly hidden)
// Ultimate Performance template, else of High Performance. The Bitsum settings
// are (re-)applied on every start.
bool find_power_plan(GUID& out) {
    std::wstring createdFile = created_scheme_file();
    bool         found       = false;
    {
        std::ifstream f(createdFile.c_str(), std::ios::binary);
        GUID          g{};
        if (f.read(reinterpret_cast<char*>(&g), sizeof(g)) && scheme_exists(g))
        {
            out   = g;
            found = true;
        }
    }
    if (!found)
        found = scheme_named(PlanName, out);

    if (!found)
    {
        GUID bitsum{};
        if (!(scheme_named(BitsumName, bitsum) && duplicate_scheme(bitsum, out))
            && !duplicate_scheme(UltimateGuid, out) && !duplicate_scheme(HighPerfGuid, out))
            return false;

        if (power().writeName)
        {
            std::wstring name(PlanName);
            power().writeName(nullptr, &out, nullptr, nullptr, reinterpret_cast<UCHAR*>(name.data()),
                              DWORD((name.size() + 1) * sizeof(wchar_t)));
        }
        std::ofstream f(createdFile.c_str(), std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(&out), sizeof(out));
    }

    configure_plan(out);
    return true;
}

// Shared state of all running engine instances (several engines may run at the
// same time, e.g. in an engine match): the plan to restore and the processes
// still using the engine's power plan. Guarded by a named mutex.
struct Instance {
    DWORD    pid;
    FILETIME created;
};

struct State {
    GUID                  original{};
    GUID                  ultimate{};  // the engine's plan
    std::uint32_t         switched = 0;  // the engine changed the plan and must restore it
    std::vector<Instance> instances;
};

constexpr std::uint32_t StateMagic = 0x57504B50;  // "PKPW"

bool read_state(State& s) {
    std::ifstream f(state_file().c_str(), std::ios::binary);
    std::uint32_t magic = 0, n = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != StateMagic)
        return false;
    f.read(reinterpret_cast<char*>(&s.original), sizeof(s.original));
    f.read(reinterpret_cast<char*>(&s.ultimate), sizeof(s.ultimate));
    f.read(reinterpret_cast<char*>(&s.switched), sizeof(s.switched));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!f || n > 4096)
        return false;
    s.instances.resize(n);
    f.read(reinterpret_cast<char*>(s.instances.data()), std::streamsize(n * sizeof(Instance)));
    return bool(f);
}

void write_state(const State& s) {
    std::ofstream f(state_file().c_str(), std::ios::binary | std::ios::trunc);
    std::uint32_t n = std::uint32_t(s.instances.size());
    f.write(reinterpret_cast<const char*>(&StateMagic), sizeof(StateMagic));
    f.write(reinterpret_cast<const char*>(&s.original), sizeof(s.original));
    f.write(reinterpret_cast<const char*>(&s.ultimate), sizeof(s.ultimate));
    f.write(reinterpret_cast<const char*>(&s.switched), sizeof(s.switched));
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(s.instances.data()), std::streamsize(n * sizeof(Instance)));
}

bool same_time(const FILETIME& a, const FILETIME& b) {
    return a.dwLowDateTime == b.dwLowDateTime && a.dwHighDateTime == b.dwHighDateTime;
}

bool alive(const Instance& in) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, in.pid);
    if (!h)
        return GetLastError() == ERROR_ACCESS_DENIED;  // exists, but not ours to inspect
    FILETIME c, e, k, u;
    bool     ok = WaitForSingleObject(h, 0) == WAIT_TIMEOUT && GetProcessTimes(h, &c, &e, &k, &u)
          && same_time(c, in.created);
    CloseHandle(h);
    return ok;
}

void prune(State& s) {
    std::vector<Instance> live;
    for (const auto& in : s.instances)
        if (alive(in))
            live.push_back(in);
    s.instances.swap(live);
}

class StateLock {
   public:
    StateLock() {
        mutex = CreateMutexW(nullptr, FALSE, L"Local\\Pikafish.PowerPlan");
        if (mutex)
        {
            DWORD rc = WaitForSingleObject(mutex, 10000);
            locked   = rc == WAIT_OBJECT_0 || rc == WAIT_ABANDONED;
        }
    }
    ~StateLock() {
        if (locked)
            ReleaseMutex(mutex);
        if (mutex)
            CloseHandle(mutex);
    }
    explicit operator bool() const { return locked; }

   private:
    HANDLE mutex  = nullptr;
    bool   locked = false;
};

// Removes `pid` from the shared state; the last instance restores the plan.
void release_instance(DWORD pid) {
    if (!power().ok())
        return;
    StateLock lock;
    if (!lock)
        return;
    State s;
    if (!read_state(s))
        return;

    std::vector<Instance> rest;
    for (const auto& in : s.instances)
        if (in.pid != pid)
            rest.push_back(in);
    s.instances.swap(rest);
    prune(s);

    if (!s.instances.empty())
    {
        write_state(s);
        return;
    }

    // Restore only if the plan is still the one the engine selected, so a plan
    // the user picked by hand while the engine was running is kept.
    GUID cur{};
    if (s.switched && active_scheme(cur) && same(cur, s.ultimate))
        power().setActive(nullptr, &s.original);
    DeleteFileW(state_file().c_str());
}

void spawn_watcher(DWORD pid) {
    std::vector<wchar_t> exe(32768);
    DWORD                n = GetModuleFileNameW(nullptr, exe.data(), DWORD(exe.size()));
    if (n == 0 || n >= exe.size())
        return;

    std::wstring cmd = L"\"" + std::wstring(exe.data(), n) + L"\" ";
    for (const char* c = WatchSwitch; *c; ++c)
        cmd += wchar_t(*c);
    cmd += L" " + std::to_wstring(pid);

    // No inherited handles: the watcher must not keep the GUI's pipes open.
    // Break away from the GUI's job object when allowed, so that killing the
    // engine's job does not take the watcher with it.
    // CPU priority "Idle": the watcher only sleeps until the engine exits.
    DWORD flags = CREATE_NO_WINDOW | IDLE_PRIORITY_CLASS;
    for (DWORD extra : {DWORD(CREATE_BREAKAWAY_FROM_JOB), DWORD(0)})
    {
        STARTUPINFOW        si{};
        PROCESS_INFORMATION pi{};
        si.cb = sizeof(si);
        std::vector<wchar_t> line(cmd.begin(), cmd.end());
        line.push_back(L'\0');
        if (CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, flags | extra, nullptr, nullptr,
                           &si, &pi))
        {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            return;
        }
    }
}

std::atomic<bool> powerAcquired{false};

BOOL WINAPI console_handler(DWORD type) {
    if (type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT)
        release_power_plan();
    return FALSE;  // let the default handling continue
}

}  // namespace

void apply_process_tuning() {
    HANDLE proc = GetCurrentProcess();

    // 1. CPU priority: High.
    if (GetPriorityClass(proc) != HIGH_PRIORITY_CLASS)
        SetPriorityClass(proc, HIGH_PRIORITY_CLASS);

    // 2. Dynamic thread priority boost on (FALSE = boost not disabled).
    BOOL boostDisabled = FALSE;
    if (GetProcessPriorityBoost(proc, &boostDisabled) && boostDisabled)
        SetProcessPriorityBoost(proc, FALSE);

    // 3. I/O priority: High. Windows only grants it with
    //    SeIncreaseBasePriorityPrivilege (an elevated administrator); otherwise
    //    fall back to Normal so that a GUI-lowered priority is still undone.
    static auto  ntSet   = load<NtSetInformationProcessFn>(L"ntdll.dll", "NtSetInformationProcess");
    static ULONG ioLevel = IoPriorityHigh;
    static bool  ioTried = false;
    if (ntSet && ioLevel)
    {
        ULONG io = ioLevel;
        LONG  st = ntSet(proc, ProcessIoPriority, &io, sizeof(io));
        if (st < 0 && !ioTried)
        {
            if (enable_privilege(L"SeIncreaseBasePriorityPrivilege"))
                st = ntSet(proc, ProcessIoPriority, &io, sizeof(io));
            if (st < 0)
            {
                io      = ioLevel = IoPriorityNormal;
                st      = ntSet(proc, ProcessIoPriority, &io, sizeof(io));
                ioLevel = st < 0 ? 0 : IoPriorityNormal;
            }
        }
        ioTried = true;
    }

    // 4. Memory priority: Normal (Windows 8 and later).
    static auto setInfo = load<SetProcessInformationFn>(L"kernel32.dll", "SetProcessInformation");
    if (setInfo)
    {
        ULONG mem = MemoryPriorityNormal;  // MEMORY_PRIORITY_INFORMATION
        setInfo(proc, ProcessMemoryPriority, &mem, sizeof(mem));
    }
}

void acquire_power_plan() {
    if (powerAcquired.load() || !power().ok())
        return;

    Instance self{GetCurrentProcessId(), {}};
    FILETIME e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &self.created, &e, &k, &u))
        return;

    {
        StateLock lock;
        if (!lock)
            return;

        State s;
        bool  hadState = read_state(s);
        prune(s);

        if (s.instances.empty())
        {
            GUID cur{}, ult{};
            if (!active_scheme(cur) || !find_power_plan(ult))
                return;

            // Every engine was killed before it could restore (and its watcher
            // was killed as well): the plan saved then is still the one to restore.
            bool stale = hadState && s.switched && same(s.ultimate, ult) && same(cur, ult);
            if (!stale)
            {
                s          = State{};
                s.original = cur;
                s.ultimate = ult;
                s.switched = !same(cur, ult) && power().setActive(nullptr, &ult) == ERROR_SUCCESS;
            }
            if (same(cur, ult))
                power().setActive(nullptr, &ult);  // re-activate so the refreshed settings apply
        }

        s.instances.push_back(self);
        write_state(s);
        powerAcquired = true;
    }

    std::atexit(release_power_plan);
    SetConsoleCtrlHandler(console_handler, TRUE);
    spawn_watcher(self.pid);
}

void release_power_plan() {
    if (powerAcquired.exchange(false))
        release_instance(GetCurrentProcessId());
}

bool run_power_watcher(int argc, char* argv[]) {
    if (argc < 3 || std::strcmp(argv[1], WatchSwitch) != 0)
        return false;

    // Keep the watcher at "Idle" even if it was started some other way.
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);

    DWORD pid = DWORD(std::strtoul(argv[2], nullptr, 10));
    if (HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid))
    {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
    release_instance(pid);
    return true;
}

namespace {

std::wstring lower(std::wstring s) {
    for (auto& c : s)
        c = wchar_t(std::towlower(c));
    return s;
}

std::wstring image_path(HANDLE proc) {
    std::vector<wchar_t> buf(32768);
    DWORD                size = DWORD(buf.size());
    if (!QueryFullProcessImageNameW(proc, 0, buf.data(), &size))
        return {};
    return lower(std::wstring(buf.data(), size));
}

// TOKEN_USER of a process (owner SID), empty on failure.
std::vector<BYTE> token_user(HANDLE proc) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(proc, TOKEN_QUERY, &token))
        return {};
    DWORD size = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    std::vector<BYTE> buf(size);
    if (!size || !GetTokenInformation(token, TokenUser, buf.data(), size, &size))
        buf.clear();
    CloseHandle(token);
    return buf;
}

bool same_user(const std::vector<BYTE>& a, const std::vector<BYTE>& b) {
    return !a.empty() && !b.empty()
        && EqualSid(reinterpret_cast<const TOKEN_USER*>(a.data())->User.Sid,
                    reinterpret_cast<const TOKEN_USER*>(b.data())->User.Sid);
}

}  // namespace

void idle_other_pikafish() {
    const DWORD             selfPid  = GetCurrentProcessId();
    const std::wstring      selfPath = image_path(GetCurrentProcess());
    const std::vector<BYTE> selfUser = token_user(GetCurrentProcess());
    if (selfPath.empty() || selfUser.empty())
        return;

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE)
        return;

    PROCESSENTRY32W e{};
    e.dwSize = sizeof(e);
    for (BOOL ok = Process32FirstW(snap, &e); ok; ok = Process32NextW(snap, &e))
    {
        // Other Pikafish / PikaJieQi engines on the box (not this process)
        const std::wstring exe = lower(e.szExeFile);
        if (e.th32ProcessID == selfPid
            || (exe.find(L"pikafish") == std::wstring::npos && exe.find(L"pikajieqi") == std::wstring::npos))
            continue;

        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_INFORMATION, FALSE,
                               e.th32ProcessID);
        if (!h)
            continue;  // other users / protected processes are never touched

        // Same user only; copies of this very executable (other instances of
        // this engine and its own watchers) are left alone.
        if (same_user(token_user(h), selfUser) && image_path(h) != selfPath
            && GetPriorityClass(h) != IDLE_PRIORITY_CLASS)
            SetPriorityClass(h, IDLE_PRIORITY_CLASS);
        CloseHandle(h);
    }
    CloseHandle(snap);
}

std::vector<int> plan_physical_core_pins(std::size_t threads) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
        return {};
    std::vector<char> buf(len);
    auto* base = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len))
        return {};

    // A process living in a single processor group (the usual case) can only
    // run on its affinity mask; on multi-group systems every group is usable.
    bool      filter = false;
    USHORT    procGroup = 0;
    DWORD_PTR procMask = 0, sysMask = 0;
    {
        USHORT groups[64];
        USHORT count = 64;
        if (GetProcessGroupAffinity(GetCurrentProcess(), &count, groups) && count == 1
            && GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask))
        {
            filter    = true;
            procGroup = groups[0];
        }
    }

    std::vector<PhysicalCore> cores;
    for (DWORD off = 0; off < len;)
    {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (info->Size == 0)
            break;
        off += info->Size;
        if (info->Relationship != RelationProcessorCore || info->Processor.GroupCount == 0)
            continue;

        const GROUP_AFFINITY& ga   = info->Processor.GroupMask[0];
        KAFFINITY             mask = ga.Mask;
        if (filter)
            mask = ga.Group == procGroup ? mask & KAFFINITY(procMask) : 0;
        if (!mask)
            continue;

        int bit = 0;
        while (!((mask >> bit) & 1))
            ++bit;
        cores.push_back({int(info->Processor.EfficiencyClass), int(ga.Group) * 64 + bit,
                         ga.Group == 0 && (ga.Mask & 1)});
    }

    return select_physical_cores(std::move(cores), threads);
}

bool pin_current_thread(int cpu) {
    if (cpu < 0)
        return false;
    GROUP_AFFINITY ga{};
    ga.Group = WORD(cpu / 64);
    ga.Mask  = KAFFINITY(1) << (cpu % 64);
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
}

}  // namespace jq::WinPerf

#endif  // _WIN32
