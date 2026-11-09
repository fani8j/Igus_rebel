"""Runs the page script under node with a stub DOM against real state JSON (skipped without node), plus static layout checks."""
import json
import os
import re
import shutil
import subprocess

import pytest

from dual_cut_ui import bridge as B
from helpers import Env, read_data, old_placement_limits

NODE = shutil.which("node")
STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dual_cut_ui", "static")
APP = os.path.join(STATIC, "app.js")
SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend_smoke.js")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_page(tmp_path, state, mode="", local_storage=None):
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    env = dict(os.environ, SMOKE_LOCALSTORAGE=json.dumps(local_storage or {}))
    r = subprocess.run([NODE, SMOKE, APP, str(p), mode], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def full_state(tmp_path, report="sequence_plan_report.json"):
    e = Env(tmp_path)
    try:
        e.bridge._plan_report = (read_data(report), e.clock())
        e.bridge._exec_reports = [(read_data("sequence_exec_report.json"), e.clock())]
        e.bridge.results["plan"] = (True, json.loads(read_data(report))["message"])
        e.act("detect_and_plan")
        e.wait_job()
        e.bridge.set_ready()
        st = e.core.state("op")
        st["cameras"] = {"scene": {}, "left_gripper": {}, "left_wrist": {}}
        return st
    finally:
        e.close()


@needs_node
@pytest.mark.parametrize("report", ["sequence_plan_report.json", "side_first_report.json", "top_first_report.json"])
def test_page_renders_a_full_state_without_errors(tmp_path, report):
    out = run_page(tmp_path, full_state(tmp_path, report))
    assert out["pill"] == "PLANNED" and "planned" in out["pillClass"]
    assert "Ctrl L ✓" in out["chips"] and "Execution ON" in out["chips"] and "snapshot" in out["chips"]
    assert out["cutDisabled"] is False and out["enable"] == "true" and out["readback"] == "ENABLED"
    assert out["op"] == "You are the operator" and out["takeHidden"] is True
    assert out["calls"][0].startswith("GET /api/state")
    assert "Makespan" in out["result"] and "<svg" in out["result"] and "Placement (info)" in out["result"] and "Cut permission" in out["result"]
    assert "Snapshot age" in out["result"] and "Top cut L / R" in out["result"] and "Min cross" in out["result"]
    assert any('"software_stop"' in b for b in out["afterStop"]) and out["stopBanner"] == "sent"
    assert out["cutDialogOpen"] is True and "#3" in out["cutSummary"]
    assert set(out["tabs"]) == {"summary", "detection", "phases", "plan", "log"}


@needs_node
def test_step_rows_are_short_and_the_plan_message_is_shown_once(tmp_path):
    st = full_state(tmp_path, "side_first_report.json")
    out = run_page(tmp_path, st)
    assert "Plan OK - makespan 91 s est." in out["steps"]
    msg = st["plan"]["message"]
    first = msg[:60]
    # the long coordinator message appears in the Plan tab only (not in the steps, summary, phases or log)
    assert first not in out["steps"] and first not in out["tabs"]["summary"] and first not in out["tabs"]["phases"]
    assert msg not in out["tabs"]["log"]
    assert out["tabs"]["plan"].count(first) == 1
    assert "Parallel phases take the longer arm, serial phases the sum" in out["tabs"]["phases"]
    assert "50.3 s" in out["tabs"]["phases"] or "50.2 s" in out["tabs"]["phases"]          # serial phase B: 23.0 + 27.3


@needs_node
def test_a_long_refusal_is_shortened_in_the_step_row(tmp_path):
    st = full_state(tmp_path)
    st["job"]["state"] = "failed"
    st["job"]["steps"][1].update(state="failed", detail="refused: " + "z" * 300)
    out = run_page(tmp_path, st)
    assert "z" * 150 not in out["steps"] and "…" in out["steps"]


@needs_node
def test_action_reply_stays_after_later_renders(tmp_path):
    out = run_page(tmp_path, full_state(tmp_path))
    assert out["lastAfterRearm"] == "Rearm: rearmed (plan cleared)" and "ok" in out["lastClass"]
    assert out["lastAfterRender"] == "Rearm: rearmed (plan cleared)"
    assert out["lastAfterStop"].startswith("Software stop: ")                      # replaced only by the NEXT action


@needs_node
def test_plan_not_retained_is_flagged_and_dimmed(tmp_path):
    st = full_state(tmp_path)
    st["coordinator"].update(plan_retained=False, plan_seq=0, state=B.STATE_IDLE, state_name="IDLE")
    out = run_page(tmp_path, st)
    assert "Plan not retained any more (rearm / new plan needed)" in out["tabs"]["summary"]
    assert 'class="dim"' in out["tabs"]["summary"]
    assert "Plan not retained any more" in out["tabs"]["phases"] and "Plan not retained any more" in out["tabs"]["plan"]
    st["coordinator"].update(plan_retained=True, state=B.STATE_PLANNED, state_name="PLANNED")
    out = run_page(tmp_path, st)
    assert "not retained" not in out["tabs"]["summary"] and 'class="dim"' not in out["tabs"]["summary"]


@needs_node
def test_failed_plan_report_is_flagged(tmp_path):
    st = full_state(tmp_path)
    st["plan"] = dict(st["plan"], ok=False, message="top seam coverage refused: GAP between the cuts")
    out = run_page(tmp_path, st)
    assert "The last plan attempt failed: top seam coverage refused" in out["tabs"]["summary"]


@needs_node
def test_stale_values_are_shown_as_unknown_never_as_good(tmp_path):
    st = full_state(tmp_path)
    st["coordinator"].update(online=False, state_name="COORDINATOR OFFLINE")
    st["gates"]["all_enabled"] = False
    out = run_page(tmp_path, st)
    assert "Execution ?" in out["chips"] and "Execution off" not in out["chips"]
    assert out["readback"] == "unknown" and "Close execution gates" in out["enableText"] and out["enable"] == "false"
    assert "joints ?" in out["chips"]
    st = full_state(tmp_path)                                              # coordinator alive, read-backs stale
    st["gates"]["fresh"] = False
    st["health"]["fresh"] = False
    out = run_page(tmp_path, st)
    assert "Execution ?" in out["chips"] and "Ctrl L ?" in out["chips"] and "HW R ?" in out["chips"]
    assert "Ctrl L ✓" not in out["chips"] and "Execution ON" not in out["chips"]


@needs_node
def test_page_renders_offline_stopped_and_read_only(tmp_path):
    st = full_state(tmp_path)
    st["coordinator"].update(online=False, state_name="COORDINATOR OFFLINE")
    st["operator"]["is_operator"] = False
    st["cut"] = {"can_cut": False, "blockers": ["read-only: you are not the operator", "a <b>tag</b>"]}
    st["plan"], st["svg"], st["job"], st["gates"], st["health"] = None, None, None, None, None
    st["snapshot"] = {"available": False, "error": "cannot read snapshot x"}
    out = run_page(tmp_path, st)
    assert out["pill"] == "COORDINATOR OFFLINE" and "offline" in out["pillClass"] and out["cutDialogOpen"] is False
    assert out["cutDisabled"] is True and "read-only: you are not the operator (+1 more" in out["why"]
    assert "cannot read snapshot x" in out["tabs"]["detection"] and out["op"] == "Read-only" and out["takeHidden"] is False
    assert "&lt;b&gt;tag&lt;/b&gt;" in out["tabs"]["plan"] and "<b>" not in out["tabs"]["plan"]
    st["coordinator"].update(online=True, state_name="STOPPED", stop_reason="software stop")
    out = run_page(tmp_path, st)
    assert out["pill"] == "STOPPED: software stop" and "stopped" in out["pillClass"]


@needs_node
def test_execution_progress_is_shown_while_running(tmp_path):
    st = full_state(tmp_path, "side_first_report.json")
    st["coordinator"].update(state=B.STATE_EXECUTING, state_name="EXECUTING", phase="B")
    st["execution"] = {"active": True, "phase": "B", "run_elapsed_s": 60.0, "phase_elapsed_s": 12.0}
    out = run_page(tmp_path, st)
    assert "Running phase B: 12 s of 50 s planned" in out["execLine"] and out["execLineHidden"] is False
    pres = run_page(tmp_path, st, "present")
    assert "Executing - phase B" in pres["result"] and "12 s of 50 s planned" in pres["result"] and "Run: 60 s of 91 s est." in pres["result"]


@needs_node
def test_presenter_view_shows_kpis_carton_and_progress_without_tabs(tmp_path):
    out = run_page(tmp_path, full_state(tmp_path), "present")
    assert out["op"] == "Presenter view" and out["takeHidden"] is True
    for needle in ("Makespan", "Top cut L / R", "Min cross", "Snapshot age", "Cut permission", "Placement (info)", "<svg", "Last run: phase A completed"):
        assert needle in out["result"] or needle in out["tabs"]["summary"], needle
    assert "lastAfterRearm" not in out                                       # no controls were exercised


def test_presenter_css_hides_every_control_but_software_stop():
    css = open(os.path.join(STATIC, "style.css")).read()
    assert re.search(r"body\.present \.ctl, body\.present \.opctl, body\.present #tabbar \{ display: none !important", css)
    html = open(os.path.join(STATIC, "index.html")).read()
    # every button in the page is either inside .ctl, marked opctl, a tab, a dialog button, the banner dismiss or the stop button
    for m in re.finditer(r"<button([^>]*)>", html):
        attrs = m.group(1)
        ok = ('opctl' in attrs or 'id="stop"' in attrs or 'class="tab"' in attrs or 'dlg-' in attrs or 'banner-' in attrs
              or 'id="det-toggle"' in attrs or 'id="cam-menu-btn"' in attrs)          # view-only: change what THIS page shows
        assert ok, attrs
    ctl = html[html.index('class="card ctl opctl"'):html.index('id="result"')]
    assert 'id="btn-dp"' in ctl and 'id="cut-btn"' in ctl and 'id="btn-enable"' in ctl


def test_layout_is_a_fixed_viewport_grid_without_horizontal_scroll():
    css = open(os.path.join(STATIC, "style.css")).read()
    assert re.search(r"body \{[^}]*height: 100vh[^}]*overflow: hidden", css)
    assert re.search(r"\* \{[^}]*overflow-wrap: anywhere", css)
    assert re.search(r"\.tabbody \{[^}]*overflow-y: auto; overflow-x: hidden", css)
    assert "pre.log { margin: 0; white-space: pre-wrap" in css
    assert "font-size: clamp(14px, 1.6vh, 20px)" in css                      # scales between 1366x768 and 1920x1080
    assert "#tile-large { flex: 1 1 0; min-height: 0; }" in css
    assert re.search(r"main \{[^}]*grid-template-rows: minmax\(0, 1fr\)", css)     # the row is capped to the viewport, so children cannot push it out
    html = open(os.path.join(STATIC, "index.html")).read()
    # the operator's controls sit in the first (non-scrolling) card of the right column, ahead of the scrolling result card
    assert html.index('id="btn-dp"') < html.index('id="btn-enable"') < html.index('id="cut-btn"') < html.index('id="result-body"')
    assert 'id="tabbar"' in html and html.count('data-tab=') == 5


def test_every_element_id_used_by_the_script_exists_in_the_page():
    js = open(APP).read()
    html = open(os.path.join(STATIC, "index.html")).read()
    used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", js)) | set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", js))
    ids = set(re.findall(r'id="([^"]+)"', html))
    assert used and not (used - ids), sorted(used - ids)
    assert 'id="stop"' in html and ">Software stop<" in html and "E-stop" not in html
    assert "/static/app.js" in html and "/static/style.css" in html


def test_camera_polling_rates_in_the_page():
    js = open(APP).read()
    assert "large: 125, small: 500" in js                                    # 8 Hz / 2 Hz
    assert "const others = names.filter((n) => n !== bigCam)" in js          # the big camera has no duplicate small tile
    assert "if (document.hidden || !S) return;" in js                        # no fetching while the tab is hidden


# ---- round 3 -------------------------------------------------------------------------------------------------------------
def test_dialogs_are_never_modal_and_the_top_bar_stays_above_the_backdrop():
    js = open(APP).read()
    css = open(os.path.join(STATIC, "style.css")).read()
    html = open(os.path.join(STATIC, "index.html")).read()
    assert "showModal" not in re.sub(r"//.*", "", js).replace("never showModal()", "")       # only the explanatory comment may mention it
    assert ".show()" in js and "openDlg('dlg-confirm')" in js and "openDlg('dlg-cut')" in js and "openDlg('dlg-image')" in js
    z = lambda sel: int(re.search(re.escape(sel) + r" \{[^}]*z-index: (\d+)", css).group(1))      # noqa: E731
    assert z(".topbar") > z("dialog") > z(".backdrop") > 0                                      # stop button above dialogs and backdrop
    assert re.search(r"\.topbar \{ position: relative; z-index", css)                          # z-index only works on a positioned element
    assert "dialog::backdrop" not in css                                                       # no top-layer backdrop
    assert 'id="backdrop"' in html
    # the stop button is in the top bar, not inside any dialog or the backdrop
    assert html.index('id="stop"') < html.index('id="backdrop"') < html.index('<dialog')
    assert not re.search(r"<dialog[^>]*>(?:(?!</dialog>).)*id=\"stop\"", html, re.S)


@needs_node
def test_software_stop_works_with_every_dialog_open(tmp_path):
    out = run_page(tmp_path, full_state(tmp_path))
    d = out["dialogs"]
    assert d["cut"] == {"open": True, "backdropShown": True, "stop": True, "stillOpen": True, "backdropGone": True}
    assert d["image"]["open"] and d["image"]["backdropShown"] and d["image"]["stop"] and d["image"]["closedByEsc"]
    assert d["confirm"]["open"] and d["confirm"]["backdropShown"] and d["confirm"]["stop"] and d["confirm"]["cancelledByBackdrop"]


def test_the_zoom_dialog_cannot_reach_the_top_bar():
    css = open(os.path.join(STATIC, "style.css")).read()
    assert "max-height: 78vh" in css and "top: 50%" in css and "dialog#dlg-image img { max-height: 60vh" in css


@needs_node
def test_last_action_shows_the_job_result_when_the_job_ends(tmp_path):
    st = full_state(tmp_path, "side_first_report.json")
    out = run_page(tmp_path, st)
    assert out["lastAfterJob"] == "Detect & plan: done - Plan OK - makespan 91 s est." and "ok" in out["lastAfterJobClass"]
    # a failed plan
    st["job"].update(state="failed", message="refused: gap")
    st["job"]["steps"][1].update(state="failed", detail="top seam coverage refused: GAP between the cuts")
    out = run_page(tmp_path, st)
    assert out["lastAfterJob"].startswith("Detect & plan: failed at Plan - top seam coverage refused") and "bad" in out["lastAfterJobClass"]
    # a failed detect
    st["job"]["steps"][0].update(state="failed", detail="prerequisite or recorder timeout: is the scene camera running?")
    st["job"]["steps"][1].update(state="skipped", detail="")
    out = run_page(tmp_path, st)
    assert "failed at Detect - prerequisite or recorder timeout" in out["lastAfterJob"]
    st["job"].update(state="cancelled")
    assert run_page(tmp_path, st)["lastAfterJob"] == "Detect & plan: cancelled"


def test_the_job_line_is_not_overwritten_by_a_newer_action():
    js = open(APP).read()
    assert "if (w.seq !== lastSeq) return;" in js and "jobWatch = { id: r.job" in js


@needs_node
def test_kpi_coverage_value_is_short_and_nowrap(tmp_path):
    out = run_page(tmp_path, full_state(tmp_path), "present")
    assert "0-85 / 82-100 %" in out["result"] and "0-85 % / 82-100 %" not in out["result"]
    css = open(os.path.join(STATIC, "style.css")).read()
    assert re.search(r"\.kpi \.v \{[^}]*white-space: nowrap", css)


def test_presenter_images_fill_the_column():
    css = open(os.path.join(STATIC, "style.css")).read()
    assert re.search(r"\.pview \{[^}]*grid-template-rows: minmax\(0, 1fr\) minmax\(0, 1fr\)", css)           # SVG and overlay: half each
    assert re.search(r"\.pview \.topdown svg, \.pview img\.thumb \{[^}]*width: 100%; height: 100%", css)


def test_top_bar_shows_the_lab_and_compal_logos():
    html = open(os.path.join(STATIC, "index.html")).read()
    brand = html[html.index('<div class="brand">'):html.index('id="state-pill"')]
    for src, alt in (("/static/lab_logo.png", "Hucenrotia Lab"), ("/static/compal_logo.svg", "Compal")):
        assert f'src="{src}"' in brand and f'alt="{alt}"' in brand
        assert os.path.isfile(os.path.join(STATIC, os.path.basename(src)))      # flat static/: installed by setup.py's glob
    svg = open(os.path.join(STATIC, "compal_logo.svg")).read().lower()
    assert svg.startswith("<svg") and not any(t in svg for t in ("<script", "onload", "javascript:", "foreignobject"))
    assert "Dual-arm cutting" in brand


def cam_state(tmp_path):
    st = full_state(tmp_path)
    st["cameras"] = {"scene": {}, "knife": {}, "left_wrist": {}, "left_gripper": {}}
    st["cameras_default_off"] = ["knife"]
    return st


@needs_node
@pytest.mark.parametrize("mode", ["", "present"])
def test_camera_menu_switches_cameras_on_and_off(tmp_path, mode):
    out = run_page(tmp_path, cam_state(tmp_path), mode)
    c = out["cams"]
    for n in ("scene", "knife", "left_wrist", "left_gripper"):                 # every camera has a checkbox, in every view
        assert f'data-cam="{n}"' in c["menu"]
    assert 'data-cam="knife"> knife' in c["menu"] and 'data-cam="scene" checked' in c["menu"]   # knife starts off (server default)
    assert 'data-cam="knife"' not in c["small"] and 'data-cam="left_wrist"' in c["small"] and c["large"].startswith("scene")
    assert c["afterSceneOff"]["large"] == "left wrist" and 'data-cam="left_wrist"' not in c["afterSceneOff"]["small"]
    assert sorted(json.loads(c["afterSceneOff"]["saved"])) == ["knife", "scene"]              # remembered in this browser
    assert c["allOff"]["large"] == "" and c["allOff"]["small"] == "" and "switched off" in c["allOff"]["nosig"]
    assert c["allOff"]["nosigHidden"] is False
    assert 'data-cam="knife"' in c["allOn"]["small"] and json.loads(c["allOn"]["saved"]) == []
    assert c["menuOpen"] and c["menuClosedByEsc"]


@needs_node
def test_the_browser_choice_wins_over_the_server_default(tmp_path):
    out = run_page(tmp_path, cam_state(tmp_path), local_storage={"dual_cut_ui.cameras_off": "[]"})
    assert 'data-cam="knife"' in out["cams"]["small"] and 'data-cam="knife" checked' in out["cams"]["menu"]
    out = run_page(tmp_path, cam_state(tmp_path), local_storage={"dual_cut_ui.cameras_off": "not json"})
    assert 'data-cam="knife"' not in out["cams"]["small"]                      # unreadable -> the server default


def test_camera_menu_is_not_an_operator_control():
    html = open(os.path.join(STATIC, "index.html")).read()
    btn = html[html.index('id="cam-menu-btn"') - 60:html.index('id="cam-menu-btn"') + 120]
    assert "opctl" not in btn                                                # viewers and the presenter view may use it


# ---------------------------------------------------------------- cut permission, placement notes, blocked-detection switch
def blocked_state(tmp_path, allowed_on=False, blocked=True):
    from helpers import write_snapshot
    e = Env(tmp_path, placement_limits=old_placement_limits())
    try:
        if blocked:
            write_snapshot(e.snap, mutate=lambda d: d.__setitem__("cut_permission", {
                "status": "blocked", "reasons": [{"code": "generic_seam_unconfirmed"}], "flags": []}))
        e.bridge.set_ready()
        if allowed_on:
            e.bridge._allow_blocked = (True, e.clock())
        st = e.core.state("op")
        st["cameras"] = {"scene": {}, "left_wrist": {}}
        return st
    finally:
        e.close()


@needs_node
def test_blocked_detection_is_prominent_and_the_switch_asks_first(tmp_path):
    out = run_page(tmp_path, blocked_state(tmp_path))
    assert out["perm"]["hidden"] is False and out["perm"]["cls"] == "permline bad"
    assert "BLOCKED" in out["perm"]["text"] and "generic_seam_unconfirmed" in out["perm"]["text"] and "Cut is refused" in out["perm"]["text"]
    assert out["allow"]["hidden"] is False and out["allow"]["text"] == "Allow blocked detection" and out["allow"]["checked"] == "false"
    assert out["allow"]["readback"] == "off" and out["allow"]["disabled"] is False
    assert out["cutDisabled"] is True and "BLOCKED" in out["why"]
    assert out["allowDialogTitle"] == "Cut a BLOCKED detection?" and "generic_seam_unconfirmed" in out["allowDialogText"]
    assert len(out["allowPosts"]) == 1 and json.loads(out["allowPosts"][0]) == {"action": "set_allow_blocked_detection", "args": {"enabled": True}}


@needs_node
def test_switch_off_needs_no_dialog(tmp_path):
    out = run_page(tmp_path, blocked_state(tmp_path, allowed_on=True))
    assert out["allow"]["checked"] == "true" and out["allow"]["readback"] == "ALLOWED" and "click to switch off" in out["allow"]["text"]
    assert "cut allowed by the switch" in out["perm"]["text"] and out["cutDisabled"] is False
    assert [json.loads(b)["args"] for b in out["allowPosts"]] == [{"enabled": False}]
    assert out["allowDialogTitle"] != "Cut a BLOCKED detection?"


@needs_node
def test_an_allowed_detection_shows_no_switch(tmp_path):
    out = run_page(tmp_path, blocked_state(tmp_path, blocked=False))
    assert out["perm"]["cls"] == "permline ok" and out["perm"]["text"] == "Detection: cut permission ALLOWED"
    assert out["allow"]["hidden"] is True


@needs_node
def test_placement_warning_is_shown_with_the_hint_and_says_it_blocks_nothing(tmp_path):
    out = run_page(tmp_path, blocked_state(tmp_path, blocked=False))          # the recorded snapshot: centre x 0.453
    assert out["place"]["hidden"] is False
    h = out["place"]["html"]
    assert "placement warning" in h and "Left face probably out of reach" in h and "toward -x" in h and "Information only: nothing is blocked" in h
    assert "Placement (info)" in out["result"] and "warning" in out["result"]
    det = out["tabs"]["detection"]
    assert "Left face reach" in det and "probably out of reach" in det and "Tested region" in det and "Right holder / table" in det
    assert out["cutDisabled"] is False                                          # the warning does not block Cut


@needs_node
def test_no_placement_warning_for_a_good_placement(tmp_path):
    from helpers import write_snapshot, Env as E2, old_placement_limits
    fix = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "placement_20261022.json")))["detections"]
    e = E2(tmp_path, placement_limits=old_placement_limits())
    try:
        write_snapshot(e.snap, mutate=lambda d: d.update(fix["t062904"]["snapshot"]))
        e.bridge.set_ready()
        st = e.core.state("op")
        st["cameras"] = {"scene": {}}
    finally:
        e.close()
    out = run_page(tmp_path, st)
    assert out["place"]["hidden"] is True and "in tested area" in out["result"]
