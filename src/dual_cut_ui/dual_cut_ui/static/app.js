/* Dual-arm cutting UI. Plain vanilla JS, no dependencies, works offline. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- helpers
  const $ = (id) => document.getElementById(id);
  const DASH = '-';
  const HOLD_MS = 1500;
  const POLL_MS = 500, POLL_HIDDEN_MS = 2000;
  const PERIOD = { large: 125, small: 500 };      // large tile ~8 Hz, small tiles 2 Hz
  const FAIL_PERIOD_MS = 1000;
  const view = new URLSearchParams(window.location.search).get('view') || '';
  const present = view === 'present';
  if (present) document.body.classList.add('present');

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function num(v, digits, unit) {
    return (typeof v === 'number' && isFinite(v)) ? v.toFixed(digits) + (unit || '') : DASH;
  }
  function any(v) { return (v == null || v === '') ? DASH : String(v); }
  function short(s, n) { s = String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); return s.length <= n ? s : s.slice(0, n - 1) + '…'; }
  function setHTML(el, html) { if (el && el._h !== html) { el.innerHTML = html; el._h = html; } }
  function setText(el, t) { if (el && el.textContent !== t) el.textContent = t; }
  function at(o, ...keys) { let c = o; for (const k of keys) { if (c == null || typeof c !== 'object') return undefined; c = c[k]; } return c; }
  function n0(x) { const s = x.toFixed(0); return s === '-0' ? '0' : s; }      // no '-0' for tiny negative values
  function pct(iv) { return iv ? n0(iv[0] * 100) + '-' + n0(iv[1] * 100) : DASH; }
  function mm(iv) { return iv ? n0(iv[0]) + ' to ' + n0(iv[1]) + ' mm' : DASH; }

  // ---------------------------------------------------------- token / page id
  const pageId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
    : 'p-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  let token = '';
  try {
    const url = new URL(window.location.href);
    const t = url.searchParams.get('token');
    if (t) {
      sessionStorage.setItem('dual_cut_token', t);
      url.searchParams.delete('token');
      history.replaceState(null, '', url.pathname + (url.search || '') + url.hash);
    }
    token = sessionStorage.getItem('dual_cut_token') || '';
  } catch (e) { /* storage blocked: the backend will answer 401 */ }

  let S = null, failCount = 0, authBad = false;
  const pending = new Set();
  let tab = 'summary';

  async function api(method, path, body) {
    const headers = { 'X-Token': token, 'X-Page-Id': pageId };
    const opts = { method, headers, cache: 'no-store' };
    if (body !== undefined) { headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    let resp;
    try { resp = await fetch(path, opts); }
    catch (e) { return { ok: false, status: 0, message: 'request failed: ' + (e && e.message ? e.message : e) }; }
    if (resp.status === 401) { authBad = true; banners(); } else if (authBad && resp.status < 400) { authBad = false; banners(); }
    let data = null;
    try { data = await resp.json(); } catch (e) { /* not JSON */ }
    const out = (data && typeof data === 'object' && !Array.isArray(data)) ? data : {};
    out.status = resp.status;
    if (resp.status >= 400) { out.ok = false; if (!out.message) out.message = 'HTTP ' + resp.status; }
    if (typeof out.ok !== 'boolean') out.ok = false;
    return out;
  }
  const action = (name, args) => api('POST', '/api/action', { action: name, args: args || {} });

  // ----------------------------------------------------------------- banners and the "last action" line
  let errTimer = null;
  function banners() {
    $('banner-backend').hidden = failCount < 3;
    $('banner-auth').hidden = !authBad;
  }
  function showError(msg) {
    setText($('banner-error-text'), msg);
    $('banner-error').hidden = false;
    clearTimeout(errTimer);
    errTimer = setTimeout(() => { $('banner-error').hidden = true; }, 20000);
  }
  $('banner-error-close').addEventListener('click', () => { $('banner-error').hidden = true; });
  $('banner-stop-close').addEventListener('click', () => { $('banner-stop').hidden = true; });

  // The reply of the latest action stays here until the next action; rendering never touches it.
  const LABELS = { detect_and_plan: 'Detect & plan', detect: 'Detect', plan: 'Plan', cancel_job: 'Cancel', rearm: 'Rearm',
    set_execution_enabled: 'Execution', set_allow_blocked_detection: 'Blocked detection', execute: 'Cut', take_control: 'Take control', release_control: 'Release control', software_stop: 'Software stop' };
  let lastSeq = 0, jobWatch = null;
  function setLast(cls, text) {
    lastSeq++;
    const el = $('last-action');
    el.className = 'lastaction ' + cls;
    setText(el, text);
  }

  // ------------------------------------------------------------ software stop
  $('stop').addEventListener('click', async () => {
    setLast('info', 'Software stop: sending...');
    const r = await action('software_stop');
    const b = $('banner-stop');
    b.className = 'banner ' + (r.ok ? 'ok' : 'bad');
    const text = r.message || (r.ok ? 'Software stop sent.' : 'Software stop failed.');
    setText($('banner-stop-text'), text);
    b.hidden = false;
    setLast(r.ok ? 'ok' : 'bad', 'Software stop: ' + short(text, 160));
    pollNow();
  });

  // ------------------------------------------------------------------ dialogs
  // SAFETY: dialogs are NON-modal (dialog.show(), never showModal()) with our own backdrop below the top bar (CSS z-index), so the
  // Software stop button stays clickable while any dialog is open.  Esc or a click on the backdrop cancels the open dialog.
  const DIALOGS = ['dlg-confirm', 'dlg-cut', 'dlg-image'];
  function openDlg(id) { $(id).show(); $('backdrop').hidden = false; }
  function cancelDialogs() {
    if ($('dlg-confirm').open && $('dlg-confirm-no').onclick) $('dlg-confirm-no').onclick();
    if ($('dlg-cut').open) { stopHold(); $('dlg-cut').close(); }
    if ($('dlg-image').open) $('dlg-image').close();
  }
  for (const id of DIALOGS) {
    $(id).addEventListener('close', () => { if (!DIALOGS.some((d) => $(d).open)) $('backdrop').hidden = true; });
  }
  $('backdrop').addEventListener('click', cancelDialogs);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') cancelDialogs(); });
  function confirmDialog(title, text, yesLabel) {
    return new Promise((resolve) => {
      const dlg = $('dlg-confirm');
      setText($('dlg-confirm-title'), title);
      setText($('dlg-confirm-text'), text);
      setText($('dlg-confirm-yes'), yesLabel || 'Yes');
      const clear = () => { $('dlg-confirm-yes').onclick = null; $('dlg-confirm-no').onclick = null; };
      $('dlg-confirm-yes').onclick = () => { dlg.close(); clear(); resolve(true); };
      $('dlg-confirm-no').onclick = () => { dlg.close(); clear(); resolve(false); };
      openDlg('dlg-confirm');
    });
  }

  // -------------------------------------------------------------- state poll
  let pollTimer = null;
  async function poll() {
    const r = await api('GET', '/api/state' + (view ? '?view=' + encodeURIComponent(view) : ''));
    if (r.ok) { S = r; failCount = 0; banners(); render(); }
    else { failCount++; banners(); if (r.status === 0 || r.status >= 500) renderOffline(); }
    pollTimer = setTimeout(poll, document.hidden ? POLL_HIDDEN_MS : POLL_MS);
  }
  function pollNow() { clearTimeout(pollTimer); poll(); }
  function renderOffline() {
    const p = $('state-pill');
    p.className = 'pill offline';
    setText(p, 'UI SERVER NOT ANSWERING');
  }

  // ------------------------------------------------------------------ top bar
  const STATE_CLASS = { IDLE: 'idle', PLANNING: 'planning', PLANNED: 'planned', WAITING_BARRIER: 'waiting', EXECUTING: 'executing', STOPPED: 'stopped' };
  function chip(cls, text, title) { return '<span class="chip ' + cls + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>' + esc(text) + '</span>'; }
  const gatesKnown = () => !!(S && S.coordinator && S.coordinator.online && S.gates && S.gates.fresh);
  function renderTop() {
    const c = S.coordinator || {};
    const pill = $('state-pill');
    let text, cls;
    if (!c.online) { text = 'COORDINATOR OFFLINE'; cls = 'offline'; }
    else {
      cls = STATE_CLASS[c.state_name] || 'idle';
      text = c.state_name === 'WAITING_BARRIER' ? 'WAITING (barrier)' : c.state_name;
      if (c.state_name === 'STOPPED') text = 'STOPPED' + (c.stop_reason ? ': ' + c.stop_reason : '');
      if (c.phase && (c.state_name === 'EXECUTING' || c.state_name === 'WAITING_BARRIER')) text += ' - phase ' + c.phase;
    }
    pill.className = 'pill ' + cls;
    setText(pill, text);

    const h = S.health, g = S.gates, sn = S.snapshot || {};
    const hOk = !!(h && h.fresh);
    const chips = [];
    // A value that is stale (or whose source is offline) is shown as "?" in grey, never as good news.
    const act = (kind, arm, label) => {
      if (!hOk) return chip('unk', label + ' ?', 'no fresh controller_manager answer');
      return h[kind] && h[kind][arm] ? chip('ok', label + ' ✓') : chip('bad', label + ' NOT active');
    };
    chips.push(act('controllers', 'left', 'Ctrl L'), act('controllers', 'right', 'Ctrl R'), act('hardware', 'left', 'HW L'), act('hardware', 'right', 'HW R'));
    if (!gatesKnown()) chips.push(chip('unk', 'Execution ?', 'no fresh read-back from the coordinator'));
    else chips.push(g.all_enabled ? chip('warn', 'Execution ON') : chip('ok', 'Execution off'));
    const ja = c.joint_states_age_s;
    chips.push(c.online && typeof ja === 'number' ? chip(ja < 0.5 ? 'ok' : 'bad', 'joints ' + ja.toFixed(2) + ' s') : chip('unk', 'joints ?'));
    const mc = c.min_cross_distance_m;
    chips.push(c.online && typeof mc === 'number' ? chip(mc < 0.05 ? 'bad' : 'unk', 'cross ' + (mc * 1000).toFixed(0) + ' mm') : chip('unk', 'cross ' + DASH));
    if (!sn.available) chips.push(chip('bad', 'snapshot ' + DASH, sn.error || ''));
    else if (sn.age_s == null) chips.push(chip('bad', 'snapshot unusable'));
    else chips.push(chip(sn.fresh ? (sn.remaining_s < 60 ? 'warn' : 'ok') : 'bad',
      'snapshot ' + sn.age_s.toFixed(0) + '/' + sn.max_age_s.toFixed(0) + ' s' + (sn.fresh ? ' (' + sn.remaining_s.toFixed(0) + ' left)' : ' STALE')));
    setHTML($('chips'), chips.join(''));

    const op = S.operator || {};
    const lbl = $('op-label');
    if (present) { lbl.className = 'who ro'; setText(lbl, 'Presenter view'); }
    else if (op.is_operator) { lbl.className = 'who me'; setText(lbl, 'You are the operator'); }
    else { lbl.className = 'who ro'; setText(lbl, 'Read-only'); }
    $('take-control').hidden = present || !!op.is_operator;
  }

  // -------------------------------------------------------- control card
  function stepDetail(job, s) {
    if (s.name === 'Plan' && s.state === 'ok') {
      const p = S.plan;
      const ms = p && p.ok !== false && typeof p.makespan_s === 'number' ? ' - makespan ' + p.makespan_s.toFixed(0) + ' s' + (p.makespan_kind === 'est.' ? ' est.' : '') : '';
      return 'Plan OK' + ms;
    }
    return short(s.detail, 120);
  }
  // When the job this page started has ended, the last-action line shows its result instead of "... started" (unless another
  // action has replaced the line meanwhile).
  function watchJob() {
    const job = S.job;
    if (!jobWatch || !job || job.id !== jobWatch.id || job.state === 'running') return;
    const w = jobWatch;
    jobWatch = null;
    if (w.seq !== lastSeq) return;
    const failed = job.steps.find((x) => x.state === 'failed');
    const parts = job.steps.filter((x) => x.state === 'ok').map((x) => stepDetail(job, x));
    if (job.state === 'done') setLast('ok', w.label + ': done - ' + (job.steps.some((x) => x.name === 'Plan') ? stepDetail(job, job.steps.find((x) => x.name === 'Plan')) : parts.join('; ')));
    else if (job.state === 'cancelled') setLast('bad', w.label + ': cancelled');
    else setLast('bad', w.label + ': failed' + (failed ? ' at ' + failed.name : '') + ' - ' + short((failed && failed.detail) || job.message, 140));
  }
  function renderSteps() {
    const job = S.job;
    const ul = $('steps');
    if (!job) { setHTML(ul, '<li><span class="mark">.</span><span class="muted">Press Detect &amp; plan.</span></li>'); return; }
    const marks = { ok: '✓', failed: '✗', running: '…', pending: '•', skipped: '-' };
    setHTML(ul, job.steps.map((s) => '<li class="' + esc(s.state) + '"><span class="mark">' + (marks[s.state] || '?') + '</span><strong>' +
      esc(s.name) + '</strong><span class="detail">' + esc(stepDetail(job, s)) + '</span></li>').join(''));
  }

  function plannedFor(phase) {
    const p = S.plan;
    if (!p) return null;
    if (p.schedule_mode === 'sequence' && p.sequence && !p.sequence.fallback_to_phases) return p.makespan_s;
    const ph = (p.phases || []).find((x) => x.name === phase);
    return ph ? ph.duration_s : null;
  }
  function execText() {
    const ex = S.execution || {};
    if (!ex.active) return '';
    const planned = plannedFor(ex.phase);
    if (S.coordinator.state_name === 'WAITING_BARRIER') return 'Waiting at the barrier' + (ex.phase ? ' (phase ' + ex.phase + ')' : '') + (S.coordinator.barrier_message ? ': ' + short(S.coordinator.barrier_message, 90) : '');
    return 'Running' + (ex.phase ? ' phase ' + ex.phase : '') + ': ' + num(ex.phase_elapsed_s, 0) + ' s' + (planned != null ? ' of ' + planned.toFixed(0) + ' s planned' : '');
  }

  function renderControls() {
    const op = !!at(S, 'operator', 'is_operator');
    const c = S.coordinator || {};
    const job = S.job, running = !!(job && job.state === 'running');
    const busyState = !c.online || ['PLANNING', 'WAITING_BARRIER', 'EXECUTING'].includes(c.state_name);
    const planOk = op && !running && !busyState;
    $('btn-dp').disabled = !planOk || pending.has('dp');
    $('btn-plan').disabled = !planOk;
    $('btn-detect').disabled = !op || running;
    $('btn-rearm').disabled = !op || running;
    const detecting = running && job.steps.some((s) => s.name === 'Detect' && s.state === 'running');
    $('btn-cancel-job').hidden = !detecting;
    $('btn-cancel-job').disabled = !op;

    const known = gatesKnown();
    const on = known && !!S.gates.all_enabled;
    const rb = $('gates-readback');
    rb.className = 'readback ' + (!known ? 'unk' : (on ? 'on' : 'off'));
    setText(rb, !known ? 'unknown' : (on ? 'ENABLED' : 'off'));
    const sw = $('btn-enable');
    sw.setAttribute('aria-checked', on ? 'true' : 'false');
    setText(sw, !known ? 'Close execution gates (state unknown)' : (on ? 'Execution enabled - click to disable' : 'Enable execution'));
    sw.disabled = !op || (known && !on && !!c.stop_latched) || pending.has('enable');

    renderPermission(op, c);
    const cut = S.cut || { can_cut: false, blockers: [] };
    $('cut-btn').disabled = !cut.can_cut;
    const bl = cut.blockers || [];
    setText($('why'), cut.can_cut || !bl.length ? '' : 'Cut blocked: ' + short(bl[0], 110) + (bl.length > 1 ? ' (+' + (bl.length - 1) + ' more, see Plan tab)' : ''));
    const et = execText();
    $('exec-line').hidden = !et;
    setText($('exec-line'), et);
  }

  // cut permission of the detection (prominent), placement pre-check (information only) and the blocked-detection switch
  function renderPermission(op, c) {
    const sn = S.snapshot || {}, cp = sn.cut_permission, ab = S.allow_blocked || {};
    const pl = $('perm-line');
    pl.hidden = !sn.available;
    if (sn.available) {
      if (!cp) { pl.className = 'permline unk'; setText(pl, 'Detection: cut permission not reported'); }
      else if (cp.status === 'allowed') { pl.className = 'permline ok'; setText(pl, 'Detection: cut permission ALLOWED'); }
      else {
        const codes = (cp.reasons || []).map((r) => r.code).join(', ');
        pl.className = 'permline bad';
        setText(pl, 'Detection: cut permission ' + String(cp.status || '?').toUpperCase() + (codes ? ' - ' + codes : '') +
          (ab.value === true ? ' (cut allowed by the switch below)' : ' - Cut is refused'));
      }
    }
    const notes = $('place-notes'), p = sn.placement, ws = (p && p.warnings) || [];
    notes.hidden = !ws.length;
    setHTML(notes, ws.map((w) => '<div class="pnote ' + esc(w.level) + '"><span class="tag">' + (w.level === 'warn' ? 'placement warning' : 'placement note') + '</span><strong>' + esc(w.title) +
      '.</strong><span class="hint">' + esc(w.hint) + '</span></div>').join('') +
      (ws.length ? '<div class="muted" style="font-size:.75rem">Information only: nothing is blocked. Details: Detection tab.</div>' : ''));
    const show = !!(ab.detection_blocked || ab.value === true);
    $('allow-row').hidden = !show;
    const on = ab.value === true;
    const rb = $('allow-readback');
    rb.className = 'readback ' + (ab.value == null ? 'unk' : (on ? 'on' : 'off'));
    setText(rb, ab.value == null ? 'unknown' : (on ? 'ALLOWED' : 'off'));
    const sw = $('btn-allow');
    sw.setAttribute('aria-checked', on ? 'true' : 'false');
    setText(sw, on ? 'Blocked detection ALLOWED - click to switch off' : 'Allow blocked detection');
    sw.disabled = !op || (!on && !!c.stop_latched) || pending.has('allow');
  }

  // ---------------------------------------------------------------- result
  function kpi(label, value, sub, cls) {
    return '<div class="kpi ' + (cls || '') + '"><div class="k">' + esc(label) + '</div><div class="v">' + esc(value) + '</div><div class="s">' + esc(sub || '') + '</div></div>';
  }
  function planState() {
    const p = S.plan, c = S.coordinator || {};
    if (!p || p.parsed === false) return { kind: 'none' };
    if (p.ok === false) return { kind: 'failed' };
    if (!c.online) return { kind: 'unknown' };
    if (!c.plan_retained && c.state_name !== 'PLANNING') return { kind: 'gone' };
    return { kind: 'ok' };
  }
  function noticeHTML(ps) {
    const p = S.plan;
    if (ps.kind === 'gone') return '<div class="notice">Plan not retained any more (rearm / new plan needed).</div>';
    if (ps.kind === 'failed') return '<div class="notice bad">The last plan attempt failed: ' + esc(short(p.message, 160)) + '</div>';
    if (ps.kind === 'unknown') return '<div class="notice">Coordinator offline: the plan state is unknown.</div>';
    return '';
  }
  function kpiTiles() {
    const sn = S.snapshot || {}, plan = S.plan, ps = planState();
    const planOk = ps.kind === 'ok';
    const cov = plan && plan.top_coverage;
    const plancls = planOk ? '' : 'dim';
    const t = [];
    t.push('<div class="' + plancls + '">' + kpi('Makespan', plan && plan.makespan_s != null ? num(plan.makespan_s, 0, ' s') : DASH, plan && plan.makespan_s != null ? (plan.makespan_kind === 'est.' ? 'estimate' : plan.makespan_kind) : '') + '</div>');
    t.push('<div class="' + plancls + '">' + kpi('Top cut L / R', cov ? pct(cov.left_fraction) + ' / ' + pct(cov.right_fraction) + ' %' : DASH, cov ? 'overlap ' + num(cov.overlap_mm, 0, ' mm') : '', plan && plan.top_coverage_ok === false ? 'bad' : '') + '</div>');
    t.push('<div class="' + plancls + '">' + kpi('Min cross', plan && plan.min_cross_m != null ? num(plan.min_cross_m * 1000, 0, ' mm') : DASH, 'planned') + '</div>');
    const rem = sn.remaining_s;
    t.push(kpi('Snapshot age', sn.age_s != null ? num(sn.age_s, 0, ' s') : DASH, sn.fresh ? num(rem, 0, ' s left') : (sn.available ? 'STALE' : 'none'), sn.fresh ? (rem < 60 ? 'warn' : 'ok') : 'bad'));
    const cp = sn.cut_permission;
    t.push(kpi('Cut permission', cp ? any(cp.status) : DASH, cp && cp.reasons && cp.reasons.length ? cp.reasons.map((r) => r.code).join(', ') : (cp && cp.override ? 'override ' + cp.override : ''),
      cp && cp.status === 'blocked' ? 'bad' : (cp && cp.status === 'allowed' ? 'ok' : '')));
    const pl = sn.placement, nw = pl && pl.warnings ? pl.warnings.length : 0;
    const hardWarn = pl && (pl.warnings || []).some((w) => w.level === 'warn');
    t.push(kpi('Placement (info)', pl ? (nw ? nw + (hardWarn ? ' warning' : ' note') + (nw > 1 ? 's' : '') : 'in tested area') : DASH,
      pl ? 'yaw ' + num(pl.yaw_deg, 1, '°') + ', x ' + num(pl.centre_x_m, 3) + ' y ' + num(pl.centre_y_m, 3) : '', hardWarn ? 'warn' : ''));
    return '<div class="kpis">' + t.join('') + '</div>';
  }
  function execReportsHTML() {
    const ex = (S.exec_reports || []).map((e) => {
      if (e.parsed === false) return '';
      return '<p>Phase ' + esc(any(e.phase)) + ': ' + (e.completed ? '<strong class="msg-ok">completed</strong>' : '<strong class="msg-bad">NOT completed</strong>') +
        ' (' + esc(any(e.mode)) + '), wall ' + num(e.wall_s, 1, ' s') + ', min cross ' + num(e.min_cross_distance_during_execution_m != null ? e.min_cross_distance_during_execution_m * 1000 : null, 0, ' mm') +
        ', tracking error ' + num(e.max_tracking_error_rad, 3, ' rad') + (e.stop_reason ? ', stop: ' + esc(e.stop_reason) : '') + '</p>';
    }).join('');
    return ex ? '<h3>Last execution</h3>' + ex : '';
  }
  function summaryHTML() {
    const ps = planState();
    let h = noticeHTML(ps) + kpiTiles();
    if (S.svg) h += '<div class="topdown">' + S.svg + '</div>';
    return h + execReportsHTML();
  }
  function permissionText(cp) {
    if (!cp) return DASH;
    const codes = (cp.reasons || []).map((r) => r.code + (r.detail ? ' (' + r.detail + ')' : ''));
    let t = String(cp.status || DASH);
    if (codes.length) t += ': ' + codes.join('; ');
    if (cp.override) t += '  [override: ' + cp.override + ']';
    return t;
  }
  function detectionHTML() {
    const sn = S.snapshot || {};
    let h = detUrl ? '<img class="thumb" src="' + detUrl + '" alt="detection overlay (click to enlarge)" title="click to enlarge">' : '<p class="muted">No detection overlay for the current snapshot.</p>';
    if (!sn.available) return h + '<p class="msg-bad">' + esc(sn.error || 'no snapshot') + '</p>';
    h += '<dl class="facts"><dt>Snapshot</dt><dd>' + esc(any(sn.snapshot_id)) + '</dd><dt>Detector</dt><dd>' + esc(any(sn.detector_method)) +
      '</dd><dt>Cut permission</dt><dd>' + esc(permissionText(sn.cut_permission)) + '</dd><dt>Age</dt><dd>' + num(sn.age_s, 0, ' s') + ' of ' + num(sn.max_age_s, 0, ' s') +
      (sn.fresh ? '' : ' (stale)') + '</dd><dt>Plane rms</dt><dd>' + num(sn.plane_rms_m != null ? sn.plane_rms_m * 1000 : null, 1, ' mm') + '</dd>';
    const p = sn.placement;
    if (p) {
      const r = p.region || {};
      h += '<dt>Placement</dt><dd>seam yaw ' + num(p.yaw_deg, 1, ' deg') + ', centre x ' + num(p.centre_x_m, 3) + ' y ' + num(p.centre_y_m, 3) + ' m - information only, nothing is blocked</dd>' +
        '<dt>Tested region</dt><dd>' + (r.inside ? 'inside' : '<strong>outside</strong>') + ' the box where today\'s detections planned (x ' + (r.x_m || []).map((v) => num(v, 3)).join(' .. ') +
        ', y ' + (r.y_m || []).map((v) => num(v, 3)).join(' .. ') + ' m, yaw ' + (r.yaw_deg || []).map((v) => num(v, 1)).join(' .. ') + ' deg)</dd>';
      const tr = p.two_arm_reach;
      if (tr) h += '<dt>Two-arm reach</dt><dd>model margin ' + num(tr.margin_mm, 0, ' mm') + ' (' + esc(String(tr.binding || '').replace('_', ' ')) + '): ' +
        (tr.status === 'fail' ? '<strong>probably cannot be cut by both arms</strong>' : (tr.status === 'marginal' ? '<strong>marginal</strong>' : 'ok')) + ' - estimate, the planner decides</dd>';
      const lr = p.left_reach;
      if (lr) h += '<dt>Left face reach</dt><dd>wrist centre ' + num(lr.wrist_radius_mm, 0, ' mm') + ' from the shoulder at the back of the tape band (flag above ' + num(lr.flag_radius_mm, 0, ' mm') +
        ', arm limit ' + num(lr.envelope_mm, 0, ' mm') + '): ' + (lr.flag ? '<strong>probably out of reach</strong>' : 'ok') + '</dd>';
      const rt = p.right_table;
      if (rt) h += '<dt>Right holder / table</dt><dd>' + (rt.flag ? '<strong>may be refused</strong>' : 'no known problem') + ' (refused for x &le; ' + num(rt.x_max_m, 4) + ' and yaw &le; ' + num(rt.yaw_max_deg, 1) + ')</dd>';
    }
    h += '</dl>';
    for (const w of (p && p.warnings) || []) {
      h += '<div class="pnote ' + esc(w.level) + '"><span class="tag">' + (w.level === 'warn' ? 'placement warning' : 'placement note') + '</span><strong>' + esc(w.title) + '.</strong> ' + esc(w.text) +
        '<span class="hint">' + esc(w.hint) + '</span></div>';
    }
    return h;
  }
  function phasesHTML() {
    const plan = S.plan;
    if (!plan || plan.parsed === false) return '<p class="muted">No plan report yet.</p>';
    let h = noticeHTML(planState());
    const seq = plan.sequence;
    if (plan.schedule_mode === 'sequence' && seq && !seq.fallback_to_phases) {
      h += '<p>Sequence schedule: ' + (seq.delayed_arm ? esc(seq.delayed_arm) + ' arm waits ' + num(seq.delay_s, 1, ' s') : 'no delay') +
        '; left ' + num(seq.duration_left_s, 1, ' s') + ', right ' + num(seq.duration_right_s, 1, ' s') + ' before the delay; makespan ' + num(plan.makespan_s, 1, ' s') + '.</p>';
    }
    if (plan.phases && plan.phases.length) {
      h += '<table class="phases"><tr><th>Phase</th><th>Mode</th><th>Left</th><th>Right</th><th>Duration</th><th>Min cross</th></tr>' +
        plan.phases.map((p) => '<tr><td>' + esc(p.name) + (p.ok === false ? ' (failed)' : '') + '</td><td>' + esc(any(p.mode)) + (p.serial_first ? ' (' + esc(p.serial_first) + ' first)' : '') +
          '</td><td>' + num(p.duration_left_s, 1, ' s') + '</td><td>' + num(p.duration_right_s, 1, ' s') + '</td><td>' + num(p.duration_s, 1, ' s') +
          (p.duration_how && p.duration_how !== 'parallel' && p.duration_how !== 'serial' ? ' (assumed: sum)' : '') + '</td><td>' + num(p.min_cross_effective_m != null ? p.min_cross_effective_m * 1000 : null, 0, ' mm') + '</td></tr>').join('') + '</table>' +
        '<p class="muted">Parallel phases take the longer arm, serial phases the sum of both arms. Makespan ' + num(plan.makespan_s, 1, ' s') + (plan.makespan_kind === 'est.' ? ' (est.)' : '') + '.</p>';
    } else if (!(plan.schedule_mode === 'sequence')) h += '<p class="muted">No phase data in this report.</p>';
    return h;
  }
  function planHTML() {
    const plan = S.plan, ps = planState();
    let h = noticeHTML(ps);
    if (!plan || plan.parsed === false) h += '<p class="muted">' + esc(plan ? plan.message : 'No plan report yet.') + '</p>';
    else {
      h += '<p class="' + (plan.ok ? 'msg-ok' : 'msg-bad') + '">' + (plan.ok ? 'Plan OK' : 'Plan refused / failed') + '</p><p>' + esc(any(plan.message)) + '</p>' +
        '<dl class="facts"><dt>Schedule</dt><dd>' + esc(any(plan.schedule_mode)) + '</dd><dt>Phase order</dt><dd>' + esc(any(plan.phase_order)) +
        '</dd><dt>Top split</dt><dd>' + esc(any(plan.top_split_mode)) + '</dd><dt>Top, left arm</dt><dd>' + esc(mm(plan.top_coverage && plan.top_coverage.left_mm)) +
        '</dd><dt>Top, right arm</dt><dd>' + esc(mm(plan.top_coverage && plan.top_coverage.right_mm)) + '</dd></dl>';
    }
    const cut = S.cut || {};
    if (!cut.can_cut && (cut.blockers || []).length) h += '<h3>Why Cut is blocked</h3><ul class="blockers">' + cut.blockers.map((b) => '<li>' + esc(b) + '</li>').join('') + '</ul>';
    return h + execReportsHTML();
  }
  function logHTML() {
    const job = S.job;
    if (!job) return '<p class="muted">No job yet.</p>';
    return '<p class="' + (job.state === 'done' ? 'msg-ok' : (job.state === 'running' ? 'msg-info' : 'msg-bad')) + '">' + esc(job.kind) + ' ' + esc(job.state) +
      (job.message ? ': ' + esc(short(job.message, 160)) : '') + '</p><pre class="log">' + esc((job.log || []).join('\n')) + '</pre>';
  }
  function progressHTML() {
    const ex = S.execution || {}, c = S.coordinator || {};
    if (ex.active) {
      const planned = plannedFor(ex.phase), total = at(S, 'plan', 'makespan_s');
      const frac = planned ? Math.min(1, ex.phase_elapsed_s / planned) : 0;
      const waiting = c.state_name === 'WAITING_BARRIER';
      return '<div class="progress"><div class="ptitle">' + (waiting ? 'Waiting at the barrier' : 'Executing') + (ex.phase ? ' - phase ' + esc(ex.phase) : '') + '</div>' +
        '<div>' + (waiting ? esc(short(c.barrier_message, 120)) : 'Phase: ' + num(ex.phase_elapsed_s, 0, ' s') + (planned != null ? ' of ' + planned.toFixed(0) + ' s planned' : '') +
          ' &nbsp; Run: ' + num(ex.run_elapsed_s, 0, ' s') + (total != null ? ' of ' + total.toFixed(0) + ' s est.' : '')) + '</div>' +
        '<div class="pbar"><div style="width:' + (frac * 100).toFixed(0) + '%"></div></div></div>';
    }
    const last = (S.exec_reports || []).filter((e) => e.parsed !== false).slice(-1)[0];
    return '<div class="progress"><div class="ptitle">' + (c.state_name === 'STOPPED' ? 'Stopped' : (last ? 'Last run: phase ' + esc(any(last.phase)) + (last.completed ? ' completed' : ' NOT completed') : 'No run in progress')) +
      '</div>' + (last ? '<div>wall ' + num(last.wall_s, 0, ' s') + ', min cross ' + num(last.min_cross_distance_during_execution_m != null ? last.min_cross_distance_during_execution_m * 1000 : null, 0, ' mm') + '</div>' : '') + '</div>';
  }
  function presenterHTML() {
    return '<div class="pgrid">' + progressHTML() + '<div>' + noticeHTML(planState()) + '</div>' + kpiTiles() +
      '<div class="pview"><div class="topdown">' + (S.svg || '<span class="muted">no carton data</span>') + '</div><div>' +
      (detUrl ? '<img class="thumb" src="' + detUrl + '" alt="detection overlay">' : '<span class="muted">no detection overlay</span>') + '</div></div></div>';
  }
  function renderTabs() {
    for (const b of document.querySelectorAll('.tab')) b.classList.toggle('active', b.dataset.tab === tab);
  }
  function renderResult() {
    const body = $('result-body');
    let html;
    if (present) html = presenterHTML();
    else if (tab === 'detection') html = detectionHTML();
    else if (tab === 'phases') html = phasesHTML();
    else if (tab === 'plan') html = planHTML();
    else if (tab === 'log') html = logHTML();
    else html = summaryHTML();
    setHTML(body, html);
    if (tab === 'log' && !present && body._atBottom !== false) body.scrollTop = body.scrollHeight;
  }

  // detection overlay (large JPEG, fetched when needed and when the snapshot changed)
  let detUrl = null, detKey = null, detBusy = false;
  async function refreshOverlay() {
    const d = S.detection || {};
    if (!d.available) { detUrl = null; detKey = null; return; }
    if (!(present || tab === 'detection') || detKey === d.mtime || detBusy) return;
    detBusy = true;
    try {
      const r = await fetch('/api/detection.jpg?size=large', { headers: { 'X-Token': token }, cache: 'no-store' });
      if (r.ok) { detUrl = URL.createObjectURL(await r.blob()); detKey = d.mtime; renderResult(); }
    } catch (e) { /* retried at the next poll */ }
    detBusy = false;
  }

  // a stop the SERVER issued on its own (plan changed under the Cut confirmation) must reach the operator
  let lastAlert = 0;
  function checkAlert() {
    const a = S.alert;
    if (!a || a.id <= lastAlert) return;
    lastAlert = a.id;
    if (S.server_time - a.at < 120) { showError(a.message); setLast('bad', short(a.message, 200)); }
  }

  function render() {
    checkAlert();
    renderTop();
    renderSteps();
    watchJob();
    renderControls();
    renderTabs();
    refreshOverlay().then(() => {});
    renderResult();
    renderCamList();
  }
  for (const b of document.querySelectorAll('.tab')) {
    b.addEventListener('click', () => { tab = b.dataset.tab; if (S) render(); });
  }

  // --------------------------------------------------------------- buttons
  async function runAction(key, name, args) {
    pending.add(key);
    if (S) renderControls();
    setLast('info', (LABELS[name] || name) + '...');
    const r = await action(name, args);
    pending.delete(key);
    setLast(r.ok ? 'ok' : 'bad', (LABELS[name] || name) + ': ' + short(r.message || (r.ok ? 'ok' : 'failed'), 200));
    if (r.ok && r.job) jobWatch = { id: r.job, label: LABELS[name] || name, seq: lastSeq };      // replace the "started" line when the job ends
    if (!r.ok && r.status !== 409) showError((LABELS[name] || name) + ': ' + (r.message || 'failed'));
    pollNow();
    return r;
  }
  $('btn-dp').addEventListener('click', () => runAction('dp', 'detect_and_plan', {}));
  $('btn-detect').addEventListener('click', () => runAction('detect', 'detect', {}));
  $('btn-plan').addEventListener('click', () => runAction('plan', 'plan', {}));
  $('btn-cancel-job').addEventListener('click', () => runAction('cancel', 'cancel_job', {}));
  $('btn-more').addEventListener('click', () => {
    const row = $('adv-row');
    row.hidden = !row.hidden;
    $('btn-more').setAttribute('aria-expanded', row.hidden ? 'false' : 'true');
  });
  $('btn-rearm').addEventListener('click', async () => {
    if (await confirmDialog('Rearm?', 'Rearm clears the stop and the retained plan. The arms must be at rest. You will have to plan again.', 'Rearm')) {
      runAction('rearm', 'rearm', {});
    }
  });
  $('btn-enable').addEventListener('click', async () => {
    const on = gatesKnown() && !!S.gates.all_enabled;
    if (on || !gatesKnown()) { runAction('enable', 'set_execution_enabled', { enabled: false }); return; }
    if (await confirmDialog('Arm the cell for motion?', 'This lets the coordinator execute the retained plan when you press Cut. It switches off again after the run, on Software stop, or when this UI closes.', 'Enable')) {
      runAction('enable', 'set_execution_enabled', { enabled: true });
    }
  });
  // Blocked-detection switch: sets the coordinator parameter allow_blocked_detection.  Off again by itself after the run, on Software
  // stop, at the next Detect and when the UI closes (the server does that); the page only shows the READ-BACK.
  $('btn-allow').addEventListener('click', async () => {
    const ab = (S && S.allow_blocked) || {};
    if (ab.value === true) { runAction('allow', 'set_allow_blocked_detection', { enabled: false }); return; }
    const codes = (ab.reason_codes || []).join(', ');
    if (await confirmDialog('Cut a BLOCKED detection?', 'The detector marked this carton cut_permission BLOCKED' + (codes ? ' (' + codes + ')' : '') +
      '. The coordinator refuses to execute its plan unless you allow it here. This switches off again after the run, on Software stop, at the next Detect, or when this UI closes.', 'Allow')) {
      runAction('allow', 'set_allow_blocked_detection', { enabled: true });
    }
  });
  $('take-control').addEventListener('click', async () => {
    setLast('info', 'Take control...');
    let r = await action('take_control', {});
    if (!r.ok && r.needs_force) {
      if (await confirmDialog('Take over?', 'Another screen is the operator. Take over from the other screen?', 'Take over')) {
        r = await action('take_control', { force: true });
      } else { setLast('info', 'Take control: cancelled'); pollNow(); return; }
    }
    setLast(r.ok ? 'ok' : 'bad', 'Take control: ' + short(r.message || (r.ok ? 'ok' : 'failed'), 160));
    pollNow();
  });

  // ------------------------------------------------------ cut + hold to confirm
  let cutSeq = null, holdStart = null, holdRaf = null;
  $('cut-btn').addEventListener('click', () => {
    if (!S || !S.cut || !S.cut.can_cut) return;
    cutSeq = at(S, 'coordinator', 'plan_seq');
    const plan = S.plan || {}, sn = S.snapshot || {}, cov = plan.top_coverage || {};
    const rows = [['Plan', '#' + cutSeq], ['Makespan', num(plan.makespan_s, 1, ' s') + (plan.makespan_kind === 'est.' ? ' (est.)' : '')],
      ['Top, left arm', mm(cov.left_mm)], ['Top, right arm', mm(cov.right_mm)],
      ['Min cross', num(plan.min_cross_m != null ? plan.min_cross_m * 1000 : null, 0, ' mm')], ['Snapshot age', num(sn.age_s, 0, ' s') + ' of ' + num(sn.max_age_s, 0, ' s')]];
    setHTML($('dlg-cut-summary'), rows.map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>').join(''));
    $('hold-bar').style.width = '0';
    openDlg('dlg-cut');
  });
  $('dlg-cut-no').addEventListener('click', () => { stopHold(); $('dlg-cut').close(); });
  $('dlg-cut').addEventListener('close', stopHold);
  function stopHold() {
    holdStart = null;
    if (holdRaf) cancelAnimationFrame(holdRaf);
    holdRaf = null;
    $('hold-bar').style.width = '0';
  }
  function holdStep() {
    if (holdStart == null) return;
    // the plan must not change underneath the dialog
    if (!S || !S.cut || !S.cut.can_cut || at(S, 'coordinator', 'plan_seq') !== cutSeq) {
      stopHold(); $('dlg-cut').close(); showError('The plan or the cell state changed: look at the result and press Cut again.'); return;
    }
    const p = Math.min(1, (performance.now() - holdStart) / HOLD_MS);
    $('hold-bar').style.width = (p * 100).toFixed(0) + '%';
    if (p >= 1) { const seq = cutSeq; stopHold(); $('dlg-cut').close(); runAction('execute', 'execute', { plan_seq: seq }); return; }
    holdRaf = requestAnimationFrame(holdStep);
  }
  function startHold() { if (holdStart == null) { holdStart = performance.now(); holdRaf = requestAnimationFrame(holdStep); } }
  const hb = $('dlg-cut-hold');
  hb.addEventListener('pointerdown', (e) => { e.preventDefault(); startHold(); });
  for (const ev of ['pointerup', 'pointercancel', 'pointerleave', 'blur']) hb.addEventListener(ev, stopHold);
  hb.addEventListener('keydown', (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); startHold(); } });
  hb.addEventListener('keyup', (e) => { if (e.key === ' ' || e.key === 'Enter') stopHold(); });

  // ------------------------------------------------------------------ cameras
  let bigCam = 'scene', showDet = false, camKey = '';
  const tiles = new Map();         // id -> {name, size, img, nosig, last, inflight, fails}
  function tileHTML(name) {
    return '<div class="tile small" data-cam="' + esc(name) + '"><img alt="' + esc(name) + '"><span class="cap">' + esc(name.replace(/_/g, ' ')) + '</span><span class="nosig">no signal</span></div>';
  }
  // cameras switched on/off in this browser (localStorage; first visit: the server's cameras_default_off).  An off camera
  // is never fetched, so the server drops its subscription after its idle time and it costs no CPU.
  const CAM_OFF_KEY = 'dual_cut_ui.cameras_off';
  let camOff = null, camMenuKey = '';
  function loadCamOff() {
    try {
      const v = JSON.parse(localStorage.getItem(CAM_OFF_KEY));
      if (Array.isArray(v)) return new Set(v.filter((n) => typeof n === 'string'));
    } catch (e) { /* unreadable: use the default */ }
    return new Set(S.cameras_default_off || []);
  }
  function saveCamOff() {
    try { localStorage.setItem(CAM_OFF_KEY, JSON.stringify([...camOff])); } catch (e) { /* private mode: not remembered */ }
  }
  function renderCamMenu(all) {
    const key = all.join(',') + '|' + [...camOff].join(',');
    if (key === camMenuKey) return;
    camMenuKey = key;
    $('cam-menu').innerHTML = all.map((n) => '<label><input type="checkbox" data-cam="' + esc(n) + '"' + (camOff.has(n) ? '' : ' checked') +
      '> ' + esc(n.replace(/_/g, ' ')) + '</label>').join('') + '<span class="hint">Off = not streamed. Saved in this browser.</span>';
  }
  $('cam-menu').addEventListener('change', (e) => {
    const n = e.target.dataset && e.target.dataset.cam;
    if (!n || !camOff) return;
    if (e.target.checked) camOff.delete(n); else camOff.add(n);
    saveCamOff();
    camKey = '';
    renderCamList();
  });
  function setCamMenu(open) {
    $('cam-menu').hidden = !open;
    $('cam-menu-btn').setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  setCamMenu(false);
  $('cam-menu-btn').addEventListener('click', () => setCamMenu($('cam-menu').hidden));
  document.addEventListener('click', (e) => { if (!e.target.closest('.cam-menu-wrap')) setCamMenu(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setCamMenu(false); });

  function renderCamList() {
    const all = Object.keys(S.cameras || {});
    if (!camOff) camOff = loadCamOff();
    renderCamMenu(all);
    const names = all.filter((n) => !camOff.has(n));
    const key = all.join(',') + '|' + [...camOff].join(',') + '|' + bigCam + '|' + showDet;
    if (key === camKey) return;
    camKey = key;
    if (!names.includes(bigCam)) { bigCam = names[0] || ''; if (bigCam !== 'scene') showDet = false; }
    const small = $('small-tiles');
    const others = names.filter((n) => n !== bigCam);          // the big camera's own small tile is never built, so never polled
    small.style.gridTemplateColumns = 'repeat(' + Math.max(1, others.length) + ', minmax(0, 1fr))';
    small.innerHTML = others.map(tileHTML).join('');
    tiles.clear();
    tiles.set('large', { name: bigCam, size: 'large', img: $('img-large'), nosig: $('nosig-large'), last: 0, inflight: false, fails: 0, det: showDet && bigCam === 'scene' });
    setText($('cap-large'), (bigCam || '').replace(/_/g, ' ') + (showDet && bigCam === 'scene' ? ' - detection' : ''));
    $('img-large').removeAttribute('src');
    $('nosig-large').textContent = bigCam ? 'no signal' : 'All cameras are switched off (Cameras on/off)';
    if (!bigCam) $('nosig-large').hidden = false;
    for (const el of small.querySelectorAll('.tile')) {
      const n = el.dataset.cam;
      tiles.set('s:' + n, { name: n, size: 'small', img: el.querySelector('img'), nosig: el.querySelector('.nosig'), last: 0, inflight: false, fails: 0 });
      el.addEventListener('click', () => { bigCam = n; if (n !== 'scene') showDet = false; camKey = ''; renderCamList(); });
    }
    const dt = $('det-toggle');
    dt.hidden = bigCam !== 'scene';
    dt.setAttribute('aria-pressed', showDet ? 'true' : 'false');
    setText(dt, showDet ? 'Show live camera' : 'Show detection');
  }
  $('det-toggle').addEventListener('click', () => { showDet = !showDet; camKey = ''; if (S) renderCamList(); });

  async function fetchTile(rec) {
    rec.inflight = true;
    const url = rec.det ? '/api/detection.jpg?size=large' : '/api/camera/' + encodeURIComponent(rec.name) + '.jpg?size=' + rec.size;
    try {
      const r = await fetch(url, { headers: { 'X-Token': token }, cache: 'no-store' });
      if (r.ok) {
        const u = URL.createObjectURL(await r.blob());
        const old = rec.img.dataset.u;
        rec.img.src = u;
        rec.img.dataset.u = u;
        if (old) URL.revokeObjectURL(old);
        rec.nosig.hidden = true;
        rec.fails = 0;
      } else {
        rec.fails++;
        rec.nosig.textContent = rec.det ? 'no detection overlay' : 'no signal';
        rec.nosig.hidden = false;
        if (rec.fails >= 2) rec.img.removeAttribute('src');
      }
    } catch (e) {
      rec.fails++;
      rec.nosig.hidden = false;
    } finally {
      rec.inflight = false;
      rec.last = performance.now();
    }
  }
  setInterval(() => {
    if (document.hidden || !S) return;                      // nothing is fetched while the tab is hidden
    const now = performance.now();
    for (const rec of tiles.values()) {
      if (rec.inflight || !rec.name) continue;
      const period = rec.det ? 2000 : (rec.fails > 2 ? FAIL_PERIOD_MS : PERIOD[rec.size]);
      if (now - rec.last >= period) fetchTile(rec);
    }
  }, 60);

  // zoom the detection overlay
  $('result-body').addEventListener('click', (e) => {
    if (!e.target.classList || !e.target.classList.contains('thumb') || !detUrl) return;
    $('img-zoom').src = detUrl;
    openDlg('dlg-image');
  });
  $('dlg-image-close').addEventListener('click', () => $('dlg-image').close());
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollNow(); });

  poll();
})();
