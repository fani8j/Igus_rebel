#!/usr/bin/env python3
"""Phase 0: characterise CRI `CMD Move Joint` on the real ReBeL (operator at the e-stop).

Answers: which replies arrive (CMDACK / EXECACK / EXECEND / MOVETOEXECEND / errors), real speed
for vel%/acc%, whether a multi-joint move is joint-space linear, whether `CMD Move Stop` stops a
running move, and what a short ALIVEJOG pause does during a move.

Safety: requires the ROS driver to be STOPPED (one CRI client), refuses unless every joint is
within +/-10 deg of 0 (arm upright), moves only joints 1/4/6 by <= 8 deg at 5-30 % speed,
relative to the start pose, and always returns there. Ends with CMD Disable + QUIT.

    python3 tools/cri_characterize_move_joint.py --go [--ip 192.168.3.11]
"""
import argparse, json, os, re, socket, sys, threading, time

p = argparse.ArgumentParser()
p.add_argument("--ip", default="192.168.3.11")
p.add_argument("--port", type=int, default=3920)
p.add_argument("--go", action="store_true", help="required: actually move the robot")
p.add_argument("--out", default=os.path.expanduser(
    "~/igus_rebel_ros2/scan_results/cri_characterization_" + time.strftime("%Y%m%d_%H%M%S")))
a = p.parse_args()
if not a.go:
    print(__doc__); sys.exit(1)
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "raw_messages.log"), "w")
T0 = time.monotonic()
lock = threading.Lock()
status = {"q": None, "t": 0.0}
events = []          # (t, kind, text)
samples = []         # (t, q[6])
alive = {"on": True, "stop": False}
cnt = [1]
sock = socket.create_connection((a.ip, a.port), timeout=3.0)
sock.settimeout(0.5)

def now(): return time.monotonic() - T0

def send(cmd):
    with lock:
        c = cnt[0]; cnt[0] = cnt[0] % 9999 + 1
        msg = f"CRISTART {c} {cmd} CRIEND\n"
        sock.sendall(msg.encode())
    LOG.write(f"{now():9.4f} >> {msg.strip()}\n")
    return c

JRE = re.compile(r"POSJOINTCURRENT\s+((?:-?[\d.]+\s+){5}-?[\d.]+)")
def reader():
    buf = b""
    while not alive["stop"]:
        try:
            d = sock.recv(65536)
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
            t = now(); parts = m.split()
            kind = parts[2] if len(parts) > 2 else "?"
            if kind == "STATUS":
                mm = JRE.search(m)
                if mm:
                    q = [float(x) for x in mm.group(1).split()]
                    status["q"] = q; status["t"] = t; samples.append((t, q))
            else:
                LOG.write(f"{t:9.4f} << {m}\n")
                if kind not in ("STATUS", "RUNSTATE", "CYCLESTAT", "GRIPPERSTATE", "VARIABLES", "OPINFO",
                                "GSIG", "DIN", "DOUT", "LOG", "INFO", "CMDACK"):
                    events.append((t, kind, m[:200]))
                elif kind == "CMDACK":
                    events.append((t, kind, m[:120]))

def alivejog():
    while not alive["stop"]:
        if alive["on"]:
            try: send("ALIVEJOG 0 0 0 0 0 0 0 0 0")
            except OSError: break
        time.sleep(0.05)

threading.Thread(target=reader, daemon=True).start()
deadline = time.time() + 5
while status["q"] is None and time.time() < deadline: time.sleep(0.05)
if status["q"] is None:
    print("No STATUS within 5 s (is the ROS driver still connected?)"); sys.exit(2)
q0 = list(status["q"])
print("start joints (deg):", [round(x, 2) for x in q0])
if any(abs(x) > 10.0 for x in q0):
    print("REFUSED: every joint must be within +/-10 deg of 0 (arm upright) before this test."); sys.exit(3)
threading.Thread(target=alivejog, daemon=True).start()
for c in ("CMD SetActive true", "CMD Reset", "CMD Enable"):
    send(c); time.sleep(0.8)
time.sleep(1.5)

def target(d):  # d: {joint_index: delta_deg}
    t = list(q0)
    for k, v in d.items(): t[k] += v
    return t

def move(t, vel, acc, timeout=25.0, label="", stop_after=None, pause_alive=None):
    """Send CMD Move Joint; wait for an EXEC*END/ERROR reply or arrival+standstill."""
    ev0 = len(events); s0 = len(samples); t_send = now()
    send("CMD Move Joint " + " ".join(f"{x:.3f}" for x in t) + f" 0 0 0 {vel} {acc}")
    end = now() + timeout; stopped = paused = False
    while now() < end:
        time.sleep(0.02)
        el = now() - t_send
        if stop_after is not None and not stopped and el > stop_after:
            send("CMD Move Stop"); stopped = True
        if pause_alive is not None and not paused and el > pause_alive[0]:
            alive["on"] = False; paused = True
        if pause_alive is not None and paused and not alive["on"] and el > pause_alive[0] + pause_alive[1]:
            alive["on"] = True
        if any(k.endswith("END") or k.endswith("ERROR") for _, k, _ in events[ev0:]) and el > 0.3:
            time.sleep(0.5); break
        if (stop_after is not None and stopped and el > stop_after + 2.5):
            break
    alive["on"] = True
    seg = samples[s0:]
    return {"label": label, "vel": vel, "acc": acc, "target": t, "t_send": t_send,
            "events": [(round(tt - t_send, 3), k, x) for tt, k, x in events[ev0:]],
            "samples": [(round(tt - t_send, 4), q) for tt, q in seg],
            "final": status["q"], "stopped_by_cmd": stopped}

results = []
try:
    results.append(move(target({0: +5}), 10, 20, label="T1 j1 +5 vel10 acc20"))
    results.append(move(q0, 30, 40, label="T2 j1 back vel30 acc40"))
    results.append(move(target({0: +6, 3: -6, 5: +8}), 20, 40, label="T3 multi j1+6 j4-6 j6+8 vel20"))
    results.append(move(q0, 20, 40, label="T4 back vel20"))
    results.append(move(target({0: +8}), 5, 20, label="T5 Move Stop after 0.8 s (vel5)", stop_after=0.8))
    results.append(move(q0, 10, 20, label="T5b back vel10"))
    results.append(move(target({0: +6}), 5, 20, label="T6 ALIVEJOG paused 0.6 s at 0.8 s (vel5)", pause_alive=(0.8, 0.6)))
    results.append(move(q0, 10, 20, label="T6b back vel10"))
finally:
    try:
        send("CMD Disable"); time.sleep(0.5); send("QUIT"); time.sleep(0.3)
    except OSError:
        pass
    alive["stop"] = True; LOG.close()
    json.dump({"start": q0, "results": results}, open(os.path.join(a.out, "results.json"), "w"))

# ---- quick analysis
def speed(seg, j):
    best = 0.0
    for (t1, a1), (t2, a2) in zip(seg, seg[3:]):
        if t2 > t1: best = max(best, abs(a2[j] - a1[j]) / (t2 - t1))
    return best
print("\nresults ->", a.out)
for r in results:
    seg = r["samples"]; fin = r["final"]
    err = max(abs(f - t) for f, t in zip(fin, r["target"])) if fin else float("nan")
    kinds = [k for _, k, _ in r["events"]]
    dur = next((tt for tt, k, _ in r["events"] if k.endswith("END")), None)
    print(f"- {r['label']}: replies {kinds}; end after {dur}s; final err {err:.2f} deg; "
          f"peak j1 {speed(seg, 0):.1f} deg/s")
    if r["label"].startswith("T3") and seg:
        d = [r["target"][j] - q0[j] for j in range(6)]
        devs = []
        for _, q in seg:
            prog = [(q[j] - q0[j]) / d[j] for j in (0, 3, 5) if abs(d[j]) > 1e-6]
            devs.append(max(prog) - min(prog))
        print(f"  T3 joint-space linearity: max progress spread between joints = {max(devs):.3f} (0 = perfectly linear)")
