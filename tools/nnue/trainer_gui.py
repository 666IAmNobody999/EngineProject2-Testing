#!/usr/bin/env python3
"""
幻神揭棋引擎 NNUE 訓練器 (Windows GUI; also runs on Linux / macOS).

The GUI drives the scripts next to it (gendata.py, train.py, verify.py,
../jieqi_match.py) with an installed Python that has numpy + torch. The
Windows build (PyInstaller, see .github/workflows) bundles the scripts and the
universal engine, so only Python + "pip install numpy torch" are needed.

  trainer_gui.py                 start the GUI
  trainer_gui.py --selftest DIR  headless end-to-end test (data -> train -> verify)
"""

import json
import os
import re
import queue
import shutil
import subprocess
import sys
import threading
import time

APP_NAME = "幻神揭棋引擎 NNUE 訓練器"
RECORD_SIZE = 264
IS_WIN = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000 if IS_WIN else 0


# ---------------------------------------------------------------- helpers ---
def bundle_dir():
    return getattr(sys, "_MEIPASS", None)


def script_path(name):
    b = bundle_dir()
    if b:
        return os.path.join(b, "scripts", name)
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, name)
    return p if os.path.exists(p) else os.path.join(os.path.dirname(here), name)


def bundled_engine():
    b = bundle_dir()
    if b:
        d = os.path.join(b, "engine")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".exe") or not IS_WIN:
                    return os.path.join(d, f)
    return ""


def settings_file():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "PikaJieQiNNUE")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "settings.json")


def load_settings():
    try:
        with open(settings_file(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s):
    try:
        with open(settings_file(), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def child_env():
    e = dict(os.environ)
    e["PYTHONUNBUFFERED"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    # PyInstaller sets these for its own interpreter; the system Python must not see them
    for k in ("TCL_LIBRARY", "TK_LIBRARY", "PYTHONHOME", "PYTHONPATH"):
        e.pop(k, None)
    return e


def probe_python(cmd):
    """Return (sys.executable, has numpy+torch) of a working interpreter, or ('', False)."""
    code = ("import sys; print(sys.executable)\n"
            "try:\n import numpy, torch; print('OK')\nexcept Exception: print('NO')")
    try:
        r = subprocess.run(cmd + ["-c", code], capture_output=True,
                           text=True, timeout=60, creationflags=CREATE_NO_WINDOW, env=child_env())
        lines = r.stdout.strip().splitlines() if r.returncode == 0 else []
        if len(lines) < 2 or not os.path.exists(lines[-2]):
            return "", False
        return lines[-2], lines[-1] == "OK"
    except Exception:
        return "", False


def find_python(preferred=""):
    """Pick an interpreter, preferring one that already has numpy and torch."""
    cands = []
    if preferred:
        cands.append([preferred])
    if not bundle_dir():
        cands.append([sys.executable])
    if IS_WIN:
        cands += [["py", "-3"], ["python"], ["python3"]]
        try:  # every interpreter the py launcher knows about
            r = subprocess.run(["py", "-0p"], capture_output=True, text=True, timeout=20,
                               creationflags=CREATE_NO_WINDOW, env=child_env())
            for line in r.stdout.splitlines():
                m = re.search(r"([A-Za-z]:\\\S.*python\.exe)\s*$", line.strip(), re.I)
                if m:
                    cands.append([m.group(1)])
        except Exception:
            pass
    else:
        cands += [["python3"], ["python"]]
    first, seen = "", set()
    for c in cands:
        exe, ok = probe_python(c)
        if not exe or exe.lower() in seen:
            continue
        seen.add(exe.lower())
        if ok:
            return exe
        first = first or exe
    return first


def count_positions(folder):
    n = 0
    try:
        for f in os.listdir(folder):
            if f.endswith(".bin") and f.startswith("part-"):
                n += os.path.getsize(os.path.join(folder, f)) // RECORD_SIZE
    except Exception:
        pass
    return n


def data_files(folder):
    try:
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if f.startswith("part-") and f.endswith(".bin"))
    except Exception:
        return []


def kill_tree(p):
    if p is None or p.poll() is not None:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
    else:
        p.kill()


class Job:
    """A child process whose output lines go to a queue."""

    def __init__(self, cmd, cwd=None, stdin_text=None):
        self.q = queue.Queue()
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env(),
                                  creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace", bufsize=1)
        if stdin_text:
            self.p.stdin.write(stdin_text)
            self.p.stdin.close()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\r\n"))
        self.p.wait()
        self.q.put(None)

    def running(self):
        return self.p.poll() is None


def run_job(cmd, cwd=None, stdin_text=None, log=print):
    j = Job(cmd, cwd, stdin_text)
    while True:
        line = j.q.get()
        if line is None:
            return j.p.returncode
        log(line)


# --------------------------------------------------------------- selftest ---
def selftest(work):
    """Headless check on the build machine: data -> train -> export -> verify."""
    os.makedirs(work, exist_ok=True)
    logf = open(os.path.join(work, "selftest.log"), "w", encoding="utf-8")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
        print(s, flush=True)

    py = find_python()
    eng = bundled_engine() or os.environ.get("PIKAJIEQI_ENGINE", "")
    log(f"python: {py}\nengine: {eng}")
    if not py or not eng:
        log("FAIL: python or engine not found")
        return 1
    data, nets = os.path.join(work, "data"), os.path.join(work, "nets")
    rc = run_job([py, script_path("gendata.py"), "--engine", eng, "--out", data, "--positions", "3000",
                  "--workers", "2", "--depth", "3", "--seed", "11"], log=log)
    if rc or count_positions(data) < 3000:
        log("FAIL: data generation")
        return 1
    rc = run_job([py, script_path("train.py"), data, "--out", nets, "--batch", "512",
                  "--epochs", "1", "--epoch-size", "20480", "--save-every", "20"], log=log)
    net = os.path.join(nets, "pikajieqi-best.nnue")
    if rc or not os.path.exists(net):
        log("FAIL: training")
        return 1
    rc = run_job([eng], cwd=work, stdin_text="gennnue out=check.bin fen=check.fen positions=200 depth=3 seed=5\nquit\n",
                 log=lambda s: None)
    rc = run_job([py, script_path("verify.py"), "--engine", eng, "--net", net,
                  "--data", os.path.join(work, "check.bin"), "--fen", os.path.join(work, "check.fen"),
                  "--count", "200"], log=log)
    log("SELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# -------------------------------------------------------------------- GUI ---
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext

    if IS_WIN:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    S = load_settings()
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("980x720")
    if IS_WIN:
        import tkinter.font as tkfont
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family="Microsoft JhengHei UI", size=10)
            except Exception:
                pass

    cpu = os.cpu_count() or 4
    home = os.path.join(os.path.expanduser("~"), "PikaJieQiNNUE")

    def var(key, default):
        v = tk.StringVar(value=str(S.get(key, default)))
        v.trace_add("write", lambda *a: (S.__setitem__(key, v.get()), save_settings(S)))
        return v

    v_python = var("python", "")
    v_engine = var("engine", bundled_engine())
    if not os.path.exists(v_engine.get()) and bundled_engine():
        v_engine.set(bundled_engine())
    v_data = var("data", os.path.join(home, "data"))
    v_nets = var("nets", os.path.join(home, "nets"))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=6, pady=6)
    jobs = {}

    def make_log(parent):
        t = scrolledtext.ScrolledText(parent, height=16, wrap="none", font=("Consolas", 9))
        t.pack(fill="both", expand=True, padx=4, pady=4)
        return t

    def log_to(widget, line):
        widget.insert("end", line + "\n")
        if int(widget.index("end-1c").split(".")[0]) > 3000:
            widget.delete("1.0", "500.0")
        widget.see("end")

    def start(name, cmd, widget, cwd=None, stdin_text=None, on_line=None, on_exit=None):
        if name in jobs and jobs[name].running():
            messagebox.showinfo(APP_NAME, "這項工作仍在執行中。")
            return
        log_to(widget, "> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        try:
            j = Job(cmd, cwd, stdin_text)
        except Exception as e:
            log_to(widget, f"無法啟動：{e}")
            return
        jobs[name] = j

        def pump():
            try:
                while True:
                    line = j.q.get_nowait()
                    if line is None:
                        log_to(widget, f"[結束，代碼 {j.p.returncode}]")
                        if on_exit:
                            on_exit(j.p.returncode)
                        return
                    log_to(widget, line)
                    if on_line:
                        on_line(line)
            except queue.Empty:
                pass
            root.after(150, pump)
        pump()

    def python_or_warn():
        py = v_python.get().strip()
        if not py or not os.path.exists(py):
            py = find_python(py)
            if py:
                v_python.set(py)
        if not py:
            messagebox.showerror(APP_NAME, "找不到 Python。請先安裝 Python，或在「環境」分頁指定 python.exe。")
            nb.select(0)
        return py

    def row(parent, r, label, v, browse=None, width=60, hint=""):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=3)
        e = ttk.Entry(parent, textvariable=v, width=width)
        e.grid(row=r, column=1, sticky="we", padx=4, pady=3)
        if browse == "file":
            ttk.Button(parent, text="瀏覽…", command=lambda: v.set(filedialog.askopenfilename() or v.get())
                       ).grid(row=r, column=2, padx=4)
        elif browse == "dir":
            ttk.Button(parent, text="瀏覽…", command=lambda: v.set(filedialog.askdirectory() or v.get())
                       ).grid(row=r, column=2, padx=4)
        if hint:
            ttk.Label(parent, text=hint, foreground="#555").grid(row=r, column=3, sticky="w", padx=4)
        return e

    def open_folder(path):
        os.makedirs(path, exist_ok=True)
        if IS_WIN:
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])

    # ---- tab 1: environment ---------------------------------------------
    t1 = ttk.Frame(nb)
    nb.add(t1, text="1. 環境")
    f = ttk.LabelFrame(t1, text="Python 與 PyTorch")
    f.pack(fill="x", padx=6, pady=6)
    f.columnconfigure(1, weight=1)
    row(f, 0, "python.exe", v_python, "file", hint="留空會自動尋找")
    row(f, 1, "揭棋引擎", v_engine, "file", hint="預設使用內置 universal 引擎")
    env_log = None

    def check_env():
        py = python_or_warn()
        if not py:
            return
        start("env", [py, "-c", "import sys, numpy, torch; print('Python', sys.version.split()[0]);"
                      "print('numpy', numpy.__version__); print('torch', torch.__version__);"
                      "print('CUDA (NVIDIA GPU):', torch.cuda.is_available());"
                      "print('CPU threads:', torch.get_num_threads()); print('環境正常，可以開始。')"], env_log)

    def pip_install(cuda=False):
        py = python_or_warn()
        if not py:
            return
        cmd = [py, "-m", "pip", "install", "--upgrade", "numpy", "torch"]
        if cuda:
            cmd += ["--index-url", "https://download.pytorch.org/whl/cu128",
                    "--extra-index-url", "https://pypi.org/simple"]
        start("env", cmd, env_log)

    bf = ttk.Frame(t1)
    bf.pack(fill="x", padx=6)
    ttk.Button(bf, text="檢查環境", command=check_env).pack(side="left", padx=4)
    ttk.Button(bf, text="安裝 numpy + torch（CPU）", command=pip_install).pack(side="left", padx=4)
    ttk.Button(bf, text="安裝 torch（NVIDIA CUDA）", command=lambda: pip_install(True)).pack(side="left", padx=4)
    ttk.Label(t1, text=f"本機 CPU 執行緒：{cpu}。設定會自動保存。", foreground="#555").pack(anchor="w", padx=10)
    env_log = make_log(t1)

    # ---- tab 2: data generation -----------------------------------------
    t2 = ttk.Frame(nb)
    nb.add(t2, text="2. 產生資料")
    f = ttk.LabelFrame(t2, text="自我對局產生訓練資料（gennnue）")
    f.pack(fill="x", padx=6, pady=6)
    f.columnconfigure(1, weight=1)
    v_pos = var("gen_positions", 50000000)
    v_workers = var("gen_workers", max(1, cpu - 4))
    v_depth = var("gen_depth", 6)
    v_random = var("gen_random", 6)
    v_hash = var("gen_hash", 16)
    v_gnet = var("gen_net", "")
    v_book = var("gen_book", "")
    row(f, 0, "資料夾", v_data, "dir")
    row(f, 1, "局面總數", v_pos, width=16, hint="例如 50000000（約 13GB）")
    row(f, 2, "同時引擎數", v_workers, width=16, hint=f"建議 {max(1, cpu - 4)}（CPU 執行緒減 4）")
    row(f, 3, "搜尋深度", v_depth, width=16, hint="越深越準但越慢，建議 6–8")
    row(f, 4, "開局隨機步數", v_random, width=16)
    row(f, 5, "每引擎 Hash (MB)", v_hash, width=16)
    row(f, 6, "用 NNUE 網絡搜尋", v_gnet, "file", hint="可留空；第 1 代起填上一代網絡")
    row(f, 7, "開局 FEN 檔", v_book, "file", hint="可留空")
    gen_status = tk.StringVar(value="")
    ttk.Label(t2, textvariable=gen_status, font=("", 11, "bold")).pack(anchor="w", padx=10)
    gen_log = None

    def gen_start():
        py = python_or_warn()
        if not py:
            return
        if not os.path.exists(v_engine.get()):
            messagebox.showerror(APP_NAME, "找不到揭棋引擎，請在「環境」分頁指定。")
            return
        os.makedirs(v_data.get(), exist_ok=True)
        cmd = [py, script_path("gendata.py"), "--engine", v_engine.get(), "--out", v_data.get(),
               "--positions", v_pos.get(), "--workers", v_workers.get(), "--depth", v_depth.get(),
               "--random", v_random.get(), "--hash", v_hash.get(), "--seed", str(int(time.time()) % 1000000 * 100)]
        if v_gnet.get().strip():
            cmd += ["--net", v_gnet.get().strip()]
        if v_book.get().strip():
            cmd += ["--book", v_book.get().strip()]
        start("gen", cmd, gen_log)

    def gen_stop():
        if "gen" in jobs and jobs["gen"].running():
            open(os.path.join(v_data.get(), "STOP"), "w").close()
            log_to(gen_log, "正在停止（已完成的對局已保存）…")

    bf = ttk.Frame(t2)
    bf.pack(fill="x", padx=6)
    ttk.Button(bf, text="開始產生", command=gen_start).pack(side="left", padx=4)
    ttk.Button(bf, text="停止", command=gen_stop).pack(side="left", padx=4)
    ttk.Button(bf, text="開啟資料夾", command=lambda: open_folder(v_data.get())).pack(side="left", padx=4)
    gen_log = make_log(t2)

    # ---- tab 3: training -------------------------------------------------
    t3 = ttk.Frame(nb)
    nb.add(t3, text="3. 訓練")
    f = ttk.LabelFrame(t3, text="訓練 NNUE（停止時自動輸出權重）")
    f.pack(fill="x", padx=6, pady=6)
    f.columnconfigure(1, weight=1)
    v_l1 = var("tr_l1", 512)
    v_batch = var("tr_batch", 8192)
    v_lr = var("tr_lr", 0.001)
    v_lambda = var("tr_lambda", 1.0)
    v_esize = var("tr_epoch_size", 1000000)
    v_epochs = var("tr_epochs", 400)
    v_save = var("tr_save_every", 2000)
    v_threads = var("tr_threads", 0)
    v_resume = tk.BooleanVar(value=True)
    row(f, 0, "資料夾", v_data, "dir")
    row(f, 1, "輸出資料夾", v_nets, "dir")
    row(f, 2, "L1 寬度", v_l1, width=16, hint="內置引擎為 512；改動須重新編譯引擎 (NNUE_L1)")
    row(f, 3, "Batch", v_batch, width=16)
    row(f, 4, "學習率", v_lr, width=16)
    row(f, 5, "λ (lambda)", v_lambda, width=16, hint="初期 1.0；資料多時 0.75–0.9")
    row(f, 6, "每 epoch 局面數", v_esize, width=16)
    row(f, 7, "最多 epoch", v_epochs, width=16)
    row(f, 8, "每幾步保存", v_save, width=16)
    row(f, 9, "CPU 執行緒", v_threads, width=16, hint="0 = 自動")
    ttk.Checkbutton(f, text="如有 checkpoint.pt 就續訓", variable=v_resume).grid(row=10, column=1, sticky="w")
    tr_status = tk.StringVar(value="")
    ttk.Label(t3, textvariable=tr_status, font=("", 11, "bold")).pack(anchor="w", padx=10)
    tr_log = None
    best_seen = {"v": None}

    def tr_line(line):
        if line.startswith("step ") and "loss" in line and "validation" not in line:
            p = line.split()
            try:
                tr_status.set(f"第 {p[1]} 步   epoch {p[3]}   訓練損失 {p[5]}   速度 {p[9]} 局面/秒"
                              + (f"   最佳驗證損失 {best_seen['v']}" if best_seen["v"] else ""))
            except IndexError:
                pass
        if "new best validation loss" in line:
            best_seen["v"] = line.split("loss")[1].split()[0]

    def tr_start():
        py = python_or_warn()
        if not py:
            return
        files = data_files(v_data.get())
        if not files:
            messagebox.showerror(APP_NAME, "資料夾內沒有 part-*.bin 訓練資料。")
            return
        os.makedirs(v_nets.get(), exist_ok=True)
        cmd = [py, script_path("train.py"), v_data.get(), "--out", v_nets.get(), "--l1", v_l1.get(),
               "--batch", v_batch.get(), "--lr", v_lr.get(), "--lambda", v_lambda.get(),
               "--epoch-size", v_esize.get(), "--epochs", v_epochs.get(), "--save-every", v_save.get(),
               "--threads", v_threads.get()]
        ck = os.path.join(v_nets.get(), "checkpoint.pt")
        if v_resume.get() and os.path.exists(ck):
            cmd += ["--resume", ck]
        best_seen["v"] = None
        start("train", cmd, tr_log, on_line=tr_line,
              on_exit=lambda rc: tr_status.set(f"已停止。權重：{os.path.join(v_nets.get(), 'pikajieqi-best.nnue')}"
                                               if rc == 0 else f"訓練結束（代碼 {rc}），請查看記錄。"))

    def tr_stop():
        if "train" in jobs and jobs["train"].running():
            open(os.path.join(v_nets.get(), "STOP"), "w").close()
            log_to(tr_log, "正在停止：完成當前一步後保存 checkpoint 及輸出權重…")

    bf = ttk.Frame(t3)
    bf.pack(fill="x", padx=6)
    ttk.Button(bf, text="開始訓練", command=tr_start).pack(side="left", padx=4)
    ttk.Button(bf, text="停止並輸出權重", command=tr_stop).pack(side="left", padx=4)
    ttk.Button(bf, text="開啟輸出資料夾", command=lambda: open_folder(v_nets.get())).pack(side="left", padx=4)
    tr_log = make_log(t3)

    # ---- tab 4: test -----------------------------------------------------
    t4 = ttk.Frame(nb)
    nb.add(t4, text="4. 驗證及對局")
    f = ttk.LabelFrame(t4, text="測試新權重")
    f.pack(fill="x", padx=6, pady=6)
    f.columnconfigure(1, weight=1)
    v_tnet = var("test_net", "")
    v_games = var("test_games", 200)
    v_mt = var("test_movetime", 100)
    v_conc = var("test_concurrency", max(1, cpu // 3))
    row(f, 0, "網絡檔 (.nnue)", v_tnet, "file", hint="留空 = 輸出資料夾的 pikajieqi-best.nnue")
    row(f, 1, "對局數", v_games, width=16)
    row(f, 2, "每步毫秒", v_mt, width=16)
    row(f, 3, "同時對局數", v_conc, width=16)
    te_log = None

    def test_net():
        n = v_tnet.get().strip() or os.path.join(v_nets.get(), "pikajieqi-best.nnue")
        if not os.path.exists(n):
            messagebox.showerror(APP_NAME, f"找不到網絡檔：{n}")
            return ""
        return n

    def verify():
        py, net = python_or_warn(), test_net()
        if not py or not net:
            return
        work = os.path.join(v_nets.get(), "verify")
        os.makedirs(work, exist_ok=True)
        for f_ in ("check.bin", "check.fen"):
            if os.path.exists(os.path.join(work, f_)):
                os.remove(os.path.join(work, f_))

        def after_gen(rc):
            start("test", [py, script_path("verify.py"), "--engine", v_engine.get(), "--net", net,
                           "--data", os.path.join(work, "check.bin"), "--fen", os.path.join(work, "check.fen"),
                           "--count", "300"], te_log)
        start("test", [v_engine.get()], te_log, cwd=work,
              stdin_text="gennnue out=check.bin fen=check.fen positions=300 depth=4 seed=9\nquit\n",
              on_exit=after_gen)

    def match():
        py, net = python_or_warn(), test_net()
        if not py or not net:
            return
        e = v_engine.get()
        start("test", [py, script_path("jieqi_match.py"), "--e1", e, "--e2", e, "--games", v_games.get(),
                       "--movetime", v_mt.get(), "--concurrency", v_conc.get(), "--o1", "Hash=64",
                       "--o2", f"Hash=64,Use NNUE=true,EvalFile={net}"], te_log)
        log_to(te_log, "e1 = 手寫評估，e2 = NNUE。Elo 為 e1 的角度：負數代表 NNUE 較強。")

    def stop_test():
        if "test" in jobs:
            kill_tree(jobs["test"].p)

    def install_net():
        net = test_net()
        if not net:
            return
        dst = filedialog.askdirectory(title="選擇引擎所在資料夾（會複製為 pikajieqi.nnue）")
        if dst:
            shutil.copyfile(net, os.path.join(dst, "pikajieqi.nnue"))
            messagebox.showinfo(APP_NAME, "已複製。請在 GUI 設定 Use NNUE = true。")

    bf = ttk.Frame(t4)
    bf.pack(fill="x", padx=6)
    ttk.Button(bf, text="驗證計算一致", command=verify).pack(side="left", padx=4)
    ttk.Button(bf, text="NNUE 對 手寫評估", command=match).pack(side="left", padx=4)
    ttk.Button(bf, text="停止", command=stop_test).pack(side="left", padx=4)
    ttk.Button(bf, text="複製權重到引擎資料夾", command=install_net).pack(side="left", padx=4)
    te_log = make_log(t4)

    # ---- status refresh / exit ------------------------------------------
    def refresh():
        n = count_positions(v_data.get())
        gen_status.set(f"目前資料：{n:,} 個局面（{n * RECORD_SIZE / 1e9:.2f} GB）")
        root.after(5000, refresh)
    refresh()

    def on_close():
        busy = [k for k, j in jobs.items() if j.running()]
        if busy and not messagebox.askyesno(APP_NAME, "仍有工作執行中。要停止並離開嗎？\n（訓練會先保存權重）"):
            return
        if "train" in busy:
            open(os.path.join(v_nets.get(), "STOP"), "w").close()
            t0 = time.time()
            while jobs["train"].running() and time.time() - t0 < 120:
                root.update()
                time.sleep(0.2)
        if "gen" in busy:
            open(os.path.join(v_data.get(), "STOP"), "w").close()
            time.sleep(2)
        for j in jobs.values():
            kill_tree(j.p)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    if not v_python.get():
        root.after(300, lambda: v_python.set(find_python()))
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    run_gui()
