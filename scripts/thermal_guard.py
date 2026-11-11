"""Thermal guard: pauses training when the Mac gets too hot, resumes once it cools down.

Reads GPU/CPU temperature, fan speed and power from `macmon` (no sudo needed) and
macOS's own thermal state (0 nominal, 1 fair, 2 serious, 3 critical).

  PAUSE   when GPU >= --pause-at °C, or CPU >= --pause-at °C, or thermal state >= serious
  RESUME  when GPU and CPU <= --resume-at °C and thermal state <= fair

Pausing is SIGSTOP of train.py / evaluate.py (and their data-loader workers); resuming is SIGCONT.
Nothing is lost: the process simply freezes and continues where it was.
Log: logs_thermal.txt (one line per minute, plus every pause/resume).

usage: python scripts/thermal_guard.py [--pause-at 95] [--resume-at 82]
"""
import argparse, json, os, signal, subprocess, sys, time
from datetime import datetime

from Foundation import NSProcessInfo

STATES = ["nominal", "fair", "serious", "critical"]
PIDFILE = "logs_thermal.pid"
# anchored so it only matches the python processes, never shells that mention the script name
TARGETS = ["^[^ ]*python[^ ]* -I scripts/(train|evaluate)\\.py"]


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open("logs_thermal.txt", "a") as f:
        f.write(line + "\n")


def target_pids():
    pids = set()
    for t in TARGETS:
        out = subprocess.run(["pgrep", "-f", t], capture_output=True, text=True).stdout.split()
        for p in out:
            pids.add(int(p))
            kids = subprocess.run(["pgrep", "-P", p], capture_output=True, text=True).stdout.split()
            pids.update(int(k) for k in kids)
    pids.discard(os.getpid())
    return pids


def signal_all(sig):
    pids = target_pids()
    for p in pids:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass
    return pids


def already_running():
    try:
        pid = int(open(PIDFILE).read())
        os.kill(pid, 0)
        return pid != os.getpid()
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause-at", type=float, default=95.0)
    ap.add_argument("--resume-at", type=float, default=82.0)
    ap.add_argument("--idle-exit", type=int, default=30, help="exit after N minutes with no training running")
    ap.add_argument("--interval", type=int, default=10, help="seconds between sensor reads")
    a = ap.parse_args()

    if already_running():
        print("thermal guard already running", flush=True)
        return
    open(PIDFILE, "w").write(str(os.getpid()))

    paused = False

    def shutdown(*_):
        if paused:
            signal_all(signal.SIGCONT)
            log("guard stopped -> training resumed")
        try:
            os.remove(PIDFILE)
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    mon = subprocess.Popen(["macmon", "pipe", "-i", str(a.interval * 1000)],
                           stdout=subprocess.PIPE, text=True, bufsize=1)
    log(f"guard started: pause at {a.pause_at}°C or thermal state serious, resume at {a.resume_at}°C")
    last_log, hottest, paused_since, idle_since = 0.0, 0.0, None, None
    for line in mon.stdout:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        gpu, cpu = d["temp"]["gpu_temp_avg"], d["temp"]["cpu_temp_avg"]
        fans = "/".join(str(f["rpm"]) for f in d.get("fans", []))
        state = int(NSProcessInfo.processInfo().thermalState())
        hottest = max(hottest, gpu, cpu)
        info = (f"GPU {gpu:.0f}°C CPU {cpu:.0f}°C fans {fans} rpm "
                f"power {d['sys_power']:.0f}W state {STATES[state]}")

        too_hot = gpu >= a.pause_at or cpu >= a.pause_at or state >= 2
        cool = gpu <= a.resume_at and cpu <= a.resume_at and state <= 1
        if not paused and too_hot:
            pids = signal_all(signal.SIGSTOP)
            if pids:
                paused, paused_since = True, time.time()
                log(f"PAUSE (too hot) {info}")
        elif paused and cool:
            signal_all(signal.SIGCONT)
            log(f"RESUME after {(time.time() - paused_since) / 60:.1f} min {info}")
            paused = False
        elif paused:
            signal_all(signal.SIGSTOP)  # catch data-loader workers respawned meanwhile

        busy = bool(target_pids())
        idle_since = None if busy else (idle_since or time.time())
        if idle_since and time.time() - idle_since > a.idle_exit * 60:
            log(f"nothing to guard for {a.idle_exit} min -> exiting. hottest seen: {hottest:.0f}°C")
            mon.terminate(); shutdown()

        if time.time() - last_log >= 60:
            running = "paused" if paused else ("training" if busy else "idle")
            log(f"{running:8s} {info}  max so far {hottest:.0f}°C")
            last_log = time.time()


if __name__ == "__main__":
    main()
