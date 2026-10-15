#!/usr/bin/env python3
"""Phase 0b: joint-1-only CRI `CMD Move Joint` tests on the real ReBeL (operator at the e-stop).

  B1  J1 0 -> +30 -> 0 at vel 25/50/100 %, acc 40 %       (speed / acceleration mapping)
  B2  J1 -> +8 at vel 5 %, socket closed at 0.8 s          (does a dropped connection stop the move?)
  B3  J1 -> +8 at vel 5 %, second Move Joint to +4 at 0.8 s (replace / queue / reject?)

Joints 2..6 are always commanded to their start values. Requires the ROS driver STOPPED and every
joint within +/-10 deg of 0. Returns J1 to the start, then CMD Disable + QUIT.

    python3 tools/cri_characterize_move_joint_b.py --go
"""
import argparse, json, os, re, socket, sys, threading, time

p = argparse.ArgumentParser()
p.add_argument("--ip", default="192.168.3.11")
p.add_argument("--port", type=int, default=3920)
p.add_argument("--go", action="store_true")
p.add_argument("--out", default=os.path.expanduser(
    "~/igus_rebel_ros2/scan_results/cri_characterization_b_" + time.strftime("%Y%m%d_%H%M%S")))
a = p.parse_args()
if not a.go:
    print(__doc__); sys.exit(1)
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "raw_messages.log"), "w")
T0 = time.monotonic()
def now(): return time.monotonic() - T0
JRE = re.compile(r"POSJOINTCURRENT\s+((?:-?[\d.]+\s+){5}-?[\d.]+)")
QUIET = ("RUNSTATE", "CYCLESTAT", "GRIPPERSTATE", "VARIABLES", "OPINFO", "GSIG", "DIN", "DOUT", "INFO")
samples, events = [], []          # shared across connections: (t, q) / (t, kind, text)
state = {"q": None}


class Conn:
    def __init__(self):
        self.sock = socket.create_connection((a.ip, a.port), timeout=3.0)
        self.sock.settimeout(0.5)
        self.lock = threading.Lock(); self.cnt = 1
        self.stop = False; self.alive_on = True
        threading.Thread(target=self._reader, daemon=True).start()

    def send(self, cmd):
        with self.lock:
            c = self.cnt; self.cnt = self.cnt % 9999 + 1
            msg = f"CRISTART {c} {cmd} CRIEND\n"
            self.sock.sendall(msg.encode())
        LOG.write(f"{now():9.4f} >> {msg.strip()}\n")

    def _reader(self):
        buf = b""
        while not self.stop:
            try:
                d = self.sock.recv(65536)
                if not d: events.append((now(), "DISCONNECT", "")); break
                buf += d
            except socket.timeout:
                continue
            except OSError:
                break
            while True:
                s = buf.find(b"CRISTART"); e = buf.find(b"CRIEND", s + 1)
                if s < 0 or e < 0: break
                m = buf[s:e + 6].decode(errors="replace"); buf = buf[e + 6:]
                t = now(); parts = m.split(); kind = parts[2] if len(parts) > 2 else "?"
                if kind == "STATUS":
                    mm = JRE.search(m)
                    if mm:
                        q = [float(x) for x in mm.group(1).split()]
                        state["q"] = q; samples.append((t, q))
                elif kind not in QUIET:
                    LOG.write(f"{t:9.4f} << {m}\n")
                    if kind not in ("LOGMSG", "MESSAGE"):
                        events.append((t, kind, m[:160]))

    def start_alive(self):
        def run():
            while not self.stop:
                if self.alive_on:
                    try: self.send("ALIVEJOG 0 0 0 0 0 0 0 0 0")
                    except OSError: break
                time.sleep(0.05)
        threading.Thread(target=run, daemon=True).start()

    def close(self):
        self.stop = True
        try: self.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.sock.close()
        LOG.write(f"{now():9.4f} -- socket closed\n")


def wait_still(timeout=15.0):
    """Wait until J1 has not changed by > 0.02 deg for 0.5 s."""
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(0.1)
        rec = [q for t, q in samples if t > now() - 0.5]
        if len(rec) > 5 and max(r[0] for r in rec) - min(r[0] for r in rec) < 0.02:
            return True
    return False


def enable(c):
    for cmd in ("CMD SetActive true", "CMD Reset", "CMD Enable"):
        c.send(cmd); time.sleep(0.8)
    time.sleep(1.5)


c = Conn()
deadline = time.time() + 5
while state["q"] is None and time.time() < deadline: time.sleep(0.05)
if state["q"] is None:
    print("No STATUS within 5 s (is the ROS driver still connected?)"); sys.exit(2)
q0 = list(state["q"])
print("start joints (deg):", [round(x, 2) for x in q0])
if any(abs(x) > 10.0 for x in q0):
    print("REFUSED: every joint must be within +/-10 deg of 0 (arm upright)."); sys.exit(3)
c.start_alive(); enable(c)

def j1(dv):
    t = list(q0); t[0] = q0[0] + dv; return t

def cmd_move(t, vel, acc):
    c.send("CMD Move Joint " + " ".join(f"{x:.3f}" for x in t) + f" 0 0 0 {vel} {acc}")

def move(t, vel, acc, label, timeout=30.0, at=None):
    """at: (seconds, callable) fired once mid-move. Returns the record for this move."""
    ev0, s0, ts = len(events), len(samples), now()
    cmd_move(t, vel, acc); fired = False; ends = 0
    while now() - ts < timeout:
        time.sleep(0.02)
        if at and not fired and now() - ts > at[0]:
            at[1](); fired = True
        ends = sum(1 for _, k, _ in events[ev0:] if k.endswith("END") or k.endswith("ERROR"))
        need = 2 if (at and at[2]) else 1
        if ends >= need and now() - ts > 0.3:
            time.sleep(0.5); break
    return {"label": label, "vel": vel, "acc": acc, "target": t, "t_send": ts,
            "events": [(round(tt - ts, 3), k, x) for tt, k, x in events[ev0:]],
            "samples": [(round(tt - ts, 4), q) for tt, q in samples[s0:]], "final": state["q"]}

results = []
try:
    for vel in (25, 50, 100):
        results.append(move(j1(+30), vel, 40, f"B1 J1 +30 vel{vel} acc40"))
        results.append(move(j1(0), vel, 40, f"B1 J1 back vel{vel} acc40"))

    # B2: close the socket mid-move, reconnect after 1.0 s and watch where J1 ends up
    ev0, s0, ts = len(events), len(samples), now()
    cmd_move(j1(+8), 5, 20)
    while now() - ts < 0.8: time.sleep(0.01)
    q_close = list(state["q"]); t_close = now() - ts
    c.close(); time.sleep(1.0)
    c = Conn(); t_reconnect = now() - ts
    time.sleep(0.3); q_reconnect = list(state["q"])
    still = wait_still(15.0); q_settled = list(state["q"])
    results.append({"label": "B2 socket closed mid-move", "t_send": ts, "t_close": t_close,
                    "t_reconnect": t_reconnect, "q_close": q_close, "q_reconnect": q_reconnect,
                    "q_settled": q_settled, "settled": still, "target": j1(+8),
                    "events": [(round(tt - ts, 3), k, x) for tt, k, x in events[ev0:]],
                    "samples": [(round(tt - ts, 4), q) for tt, q in samples[s0:]], "final": state["q"]})
    if not still:
        raise RuntimeError("J1 still moving 15 s after reconnect")
    c.start_alive(); enable(c)
    results.append(move(j1(0), 10, 20, "B2b J1 back vel10"))

    # B3: second target mid-move (+8 then +4, both slow)
    results.append(move(j1(+8), 5, 20, "B3 +8 then +4 at 0.8 s",
                        at=(0.8, lambda: cmd_move(j1(+4), 5, 20), True)))
    wait_still(15.0)
    results.append(move(j1(0), 10, 20, "B3b J1 back vel10"))
finally:
    try:
        c.send("CMD Move Stop"); time.sleep(0.3)
        c.send("CMD Disable"); time.sleep(0.5); c.send("QUIT"); time.sleep(0.3)
    except OSError:
        pass
    c.stop = True; LOG.close()
    json.dump({"start": q0, "results": results}, open(os.path.join(a.out, "results.json"), "w"))

print("\nresults ->", a.out)
for r in results:
    s = r["samples"]
    v = [(s[i + 4][1][0] - s[i][1][0]) / (s[i + 4][0] - s[i][0]) for i in range(len(s) - 4) if s[i + 4][0] > s[i][0]]
    vpk = max((abs(x) for x in v), default=0.0)
    kinds = [k for _, k, _ in r["events"]]
    print(f"- {r['label']}: replies {kinds}; peak J1 {vpk:.1f} deg/s; final J1 {r['final'][0] - q0[0]:+.2f} deg "
          f"(target {r['target'][0] - q0[0]:+.1f})")
    if r["label"].startswith("B2 "):
        print(f"  J1 at close {r['q_close'][0]:+.2f}, at reconnect {r['q_reconnect'][0]:+.2f}, settled "
              f"{r['q_settled'][0]:+.2f} (t_close {r['t_close']:.2f}s, t_reconnect {r['t_reconnect']:.2f}s)")
