"""Record-only tkinter + rclpy TCP pivot-calibration GUI.

Captures live ``parent -> tool`` TF poses, solves the ``solve_tcp`` pivot
calibration, and shows the appliable ``<joint>`` snippet for the operator to
apply by hand. This module publishes nothing and commands no motion: move the
arm with your existing teleop, then press Capture while the tip is stationary
on the pivot.
"""

import csv
import math
import threading
import tkinter as tk
from tkinter import filedialog

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

from rebel_tcp_gui.tcp_solver import solve_tcp

SAFETY_ITEMS = (
    "Tool is rigidly mounted (no flex or slip)",
    "Blade guarded and workspace clear",
    "Fixed pivot point installed",
    "Capture only while the tip is stationary on the pivot",
)

PASS_RMS_MM = 1.0
PASS_MAX_MM = 2.0


def quat_to_matrix(qx, qy, qz, qw):
    """Convert a ROS-xyzw quaternion to a 3x3 rotation matrix.

    The quaternion is normalized first. Raises ``ValueError`` for a
    zero-length quaternion.
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot convert zero-length quaternion to a matrix.")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ]
    )


class TcpGuiNode(Node):
    """rclpy node owning only the TF buffer (record-only, no publishers)."""

    def __init__(self):
        super().__init__("tcp_gui")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


class TcpGui:
    """Builds the tkinter widgets and wires them to a :class:`TcpGuiNode`."""

    def __init__(self, root, node):
        self.root = root
        self.node = node
        self._poses = []  # list of ((tx,ty,tz), (qx,qy,qz,qw))

        self.safety_vars = [tk.BooleanVar(value=False) for _ in SAFETY_ITEMS]
        self.parent_frame = tk.StringVar(value="base_link")
        self.tool_frame = tk.StringVar(value="igus_tool0")
        self.status_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="No solution yet.")

        self._build()
        self._refresh_gate()

    # -- construction ----------------------------------------------------
    def _build(self):
        self.root.title("Rebel TCP pivot calibration (record-only)")

        safety = tk.LabelFrame(self.root, text="Safety checklist")
        safety.pack(fill="x", padx=6, pady=4)
        for text, var in zip(SAFETY_ITEMS, self.safety_vars):
            tk.Checkbutton(
                safety, text=text, variable=var, command=self._refresh_gate,
                anchor="w", justify="left", wraplength=420,
            ).pack(fill="x", anchor="w")

        capture = tk.LabelFrame(self.root, text="Pose capture (record-only)")
        capture.pack(fill="x", padx=6, pady=4)
        tk.Label(
            capture,
            text="Move the arm with your existing teleop, hold the tip "
            "stationary on the pivot, then capture.",
            anchor="w", justify="left", wraplength=440,
        ).pack(fill="x")
        entry_row = tk.Frame(capture)
        entry_row.pack(fill="x")
        tk.Label(entry_row, text="Parent:").pack(side="left")
        tk.Entry(entry_row, textvariable=self.parent_frame, width=16).pack(side="left", padx=2)
        tk.Label(entry_row, text="Tool:").pack(side="left")
        tk.Entry(entry_row, textvariable=self.tool_frame, width=16).pack(side="left", padx=2)
        cap_row = tk.Frame(capture)
        cap_row.pack(fill="x", pady=2)
        self.capture_btn = tk.Button(cap_row, text="Capture pose", command=self._on_capture)
        self.capture_btn.pack(side="left", padx=2)
        tk.Button(cap_row, text="Delete selected", command=self._on_delete).pack(side="left", padx=2)
        self.pose_list = tk.Listbox(capture, width=64, height=8)
        self.pose_list.pack(fill="x", pady=2)

        solve = tk.LabelFrame(self.root, text="Solve TCP")
        solve.pack(fill="x", padx=6, pady=4)
        tk.Button(solve, text="Solve TCP", command=self._on_solve).pack(anchor="w", pady=2)
        self.result_label = tk.Label(
            solve, textvariable=self.result_var, anchor="w", justify="left",
            wraplength=440, bg="#f5f5f5", relief="sunken",
        )
        self.result_label.pack(fill="x", pady=2)
        tk.Label(solve, text="Joint snippet (apply by hand to the Xacro):").pack(anchor="w")
        self.snippet = tk.Text(solve, width=64, height=6, state="disabled", wrap="word")
        self.snippet.pack(fill="x", pady=2)
        tk.Button(solve, text="Save poses CSV", command=self._on_save_csv).pack(anchor="w", pady=2)

        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 justify="left", wraplength=460, fg="#b71c1c").pack(fill="x", padx=6, pady=2)

    # -- safety gate -----------------------------------------------------
    def _gate_open(self):
        return all(var.get() for var in self.safety_vars)

    def _refresh_gate(self):
        self.capture_btn.configure(state="normal" if self._gate_open() else "disabled")

    # -- callbacks -------------------------------------------------------
    def _on_capture(self):
        if not self._gate_open():
            self.status_var.set("Complete the safety checklist before capturing poses.")
            return
        parent = self.parent_frame.get().strip()
        tool = self.tool_frame.get().strip()
        if not parent or not tool:
            self.status_var.set("Parent and tool frames must both be non-empty.")
            return
        try:
            tf = self.node.tf_buffer.lookup_transform(
                parent, tool, Time(), timeout=Duration(seconds=2.0))
        except Exception as exc:
            self.status_var.set(f"TF lookup {parent} <- {tool} failed: {exc}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        pos = (t.x, t.y, t.z)
        quat = (q.x, q.y, q.z, q.w)
        self._poses.append((pos, quat))
        self.pose_list.insert(
            "end",
            f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
        )
        self.status_var.set(f"Captured pose {len(self._poses)} ({parent} <- {tool}).")

    def _on_delete(self):
        for idx in reversed(self.pose_list.curselection()):
            self.pose_list.delete(idx)
            del self._poses[idx]
        self.status_var.set(f"{len(self._poses)} pose(s) stored.")

    def _set_snippet(self, text):
        self.snippet.configure(state="normal")
        self.snippet.delete("1.0", "end")
        self.snippet.insert("1.0", text)
        self.snippet.configure(state="disabled")

    def _on_solve(self):
        try:
            poses = [
                (np.asarray(p, dtype=float), quat_to_matrix(*q))
                for p, q in self._poses
            ]
            result = solve_tcp(poses)
        except ValueError as exc:
            self.result_var.set(f"Cannot compute TCP: {exc}")
            self.result_label.configure(bg="#ffcdd2")
            return
        r = result["r_m"]
        c = result["pivot_m"]
        passed = result["rms_mm"] <= PASS_RMS_MM and result["max_mm"] <= PASS_MAX_MM
        verdict = (
            "Fit passes the suggested position-consistency threshold."
            if passed else
            "Fit needs review before it is applied."
        )
        residuals = ", ".join(f"{v:.3f}" for v in result["residuals_mm"])
        self.result_var.set(
            f"{verdict}\n"
            f"TCP r (igus_tool0): [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}] m\n"
            f"Pivot (base_link): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}] m\n"
            f"RMS: {result['rms_mm']:.3f} mm, max: {result['max_mm']:.3f} mm "
            f"(n={len(result['residuals_mm'])})\n"
            f"Residuals [mm]: {residuals}"
        )
        self.result_label.configure(bg="#c8e6c9" if passed else "#ffcdd2")
        self._set_snippet(
            f'<joint name="igus_tool0_to_xeg32_tool_tip" type="fixed">\n'
            f'  <parent link="igus_tool0"/>\n'
            f'  <child link="xeg32_tool_tip"/>\n'
            f'  <origin xyz="{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}" rpy="0 0 0"/>\n'
            f"</joint>"
        )
        self.status_var.set("")

    def _on_save_csv(self):
        if not self._poses:
            self.status_var.set("No poses to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save poses CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
                for pos, quat in self._poses:
                    writer.writerow([*pos, *quat])
        except OSError as exc:
            self.status_var.set(f"Could not save CSV: {exc}")
            return
        self.status_var.set(f"Saved {len(self._poses)} pose(s) to {path}.")


def main(args=None):
    """Entry point: spin rclpy in a daemon thread and run the tkinter GUI."""
    rclpy.init(args=args)
    node = TcpGuiNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    TcpGui(root, node)

    def on_close():
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()


if __name__ == "__main__":
    main()
