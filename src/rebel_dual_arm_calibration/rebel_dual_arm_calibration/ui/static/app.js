/* Dual-arm calibration UI. Plain vanilla JS, no dependencies, works offline. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- helpers
  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const MINUS = '−';
  const POLL_MS = 500;
  const IMG_PERIOD_MS = 200; // ~5 Hz
  const JOINT_AGE_OK_S = 2.0;

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }
  function at(o /*, ...keys */) {
    let cur = o;
    for (let i = 1; i < arguments.length; i++) {
      if (cur == null || typeof cur !== 'object') return undefined;
      cur = cur[arguments[i]];
    }
    return cur;
  }
  function fmtNum(v, digits, unit) {
    return (typeof v === 'number' && isFinite(v)) ? v.toFixed(digits) + (unit || '') : '—';
  }
  function fmtAny(v) {
    if (v == null || v === '') return '—';
    if (typeof v === 'object') { try { return JSON.stringify(v); } catch (e) { return '?'; } }
    return String(v);
  }
  function perArm(o, arm) {
    if (o == null) return undefined;
    if (typeof o !== 'object') return o;
    return arm in o ? o[arm] : undefined;
  }
  function truthyFlag(v) { // bool, or object carrying ok/stationary
    if (typeof v === 'boolean') return v;
    if (isObj(v)) {
      for (const k of ['stationary', 'ok', 'still']) if (typeof v[k] === 'boolean') return v[k];
    }
    return null;
  }
  function setHTML(el, html) {
    if (!el) return;
    if (el._h !== html) { el.innerHTML = html; el._h = html; }
  }
  function setText(el, txt) { if (el && el.textContent !== txt) el.textContent = txt; }
  function armLabel(a) { return String(a).toUpperCase(); }

  // ---------------------------------------------------------- token / page id
  const pageId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
    : 'p-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  let token = '';
  try {
    const url = new URL(window.location.href);
    const t = url.searchParams.get('token');
    if (t) {
      sessionStorage.setItem('calib_token', t);
      url.searchParams.delete('token');
      history.replaceState(null, '', url.pathname + (url.search || '') + url.hash);
    }
    token = sessionStorage.getItem('calib_token') || '';
  } catch (e) { /* storage blocked: token stays empty, backend will answer 401 */ }

  // ------------------------------------------------------------------- state
  let S = null;            // last good /api/state
  let failCount = 0;       // consecutive failed polls
  let authBad = false;
  let tab = 'setup';
  let selArm = null;
  let selTouch = null;
  let stopNote = null;     // {t, ok, text}
  const pending = new Set(); // names of in-flight non-jog actions

  // --------------------------------------------------------------------- api
  async function api(method, path, body) {
    const headers = { 'X-Token': token, 'X-Page-Id': pageId };
    const opts = { method, headers, cache: 'no-store' };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(path, opts);
    } catch (e) {
      return { ok: false, status: 0, message: 'request failed: ' + (e && e.message ? e.message : e) };
    }
    if (resp.status === 401) { authBad = true; updateBanners(); }
    else if (authBad && resp.status < 400) { authBad = false; updateBanners(); }
    let data = null;
    try { data = await resp.json(); } catch (e) { /* not JSON */ }
    const out = isObj(data) ? data : {};
    out.status = resp.status;
    if (resp.status >= 400) {
      out.ok = false;
      if (!out.message) out.message = resp.status === 401
        ? 'token missing/invalid: open the URL printed by the calibration_ui node' : ('HTTP ' + resp.status);
    }
    if (typeof out.ok !== 'boolean') out.ok = false;
    if (!out.message && !out.ok) out.message = 'request failed';
    return out;
  }

  // ----------------------------------------------------------------- banners
  let errTimer = null;
  function showError(msg) {
    setText($('banner-error-text'), msg);
    $('banner-error').hidden = false;
    clearTimeout(errTimer);
    errTimer = setTimeout(() => { $('banner-error').hidden = true; }, 15000);
  }
  function updateBanners() {
    $('banner-backend').hidden = failCount < 3;
    $('banner-auth').hidden = !authBad;
  }
  $('banner-error-close').addEventListener('click', () => { $('banner-error').hidden = true; });

  function setResult(el, cls, text) {
    if (!el) return;
    el.className = 'result ' + cls;
    el.textContent = text;
  }
  // Report an action response in a page result line (and the global banner on failure).
  function report(el, res, what) {
    const msg = (res && res.message) ? res.message : (res && res.ok ? 'ok' : 'failed');
    if (res && res.ok) $('banner-error').hidden = true;   // a later success supersedes an old error
    if (res && res.ok && /^NO ARM SAMPLE/i.test(msg)) {
      setResult(el, 'warn', msg);           // the service ran, but only a table-board image was stored
    } else if (res && res.ok) {
      setResult(el, 'ok', msg);
    } else {
      setResult(el, 'bad', msg);
      showError((what ? what + ': ' : '') + msg);
    }
  }
  async function runAction(name, btn, resultEl, fn, what) {
    if (pending.has(name)) return;
    pending.add(name);
    if (btn) btn.disabled = true;
    setResult(resultEl, 'pend', 'working…');
    let res;
    try { res = await fn(); } catch (e) { res = { ok: false, message: String(e) }; }
    pending.delete(name);
    if (btn) btn.disabled = false;
    report(resultEl, res, what);
    renderAll();
    return res;
  }

  // -------------------------------------------------------------------- arms
  function activeArms() {
    const a = at(S, 'recorder', 'active_arms');
    if (Array.isArray(a) && a.length) return a.map(String);
    const j = at(S, 'jog', 'arms');
    if (Array.isArray(j) && j.length) return j.map(String);
    return ['left', 'right'];
  }
  function allArms() {
    const set = [];
    for (const a of [].concat(activeArms(), at(S, 'jog', 'arms') || [])) if (set.indexOf(a) < 0) set.push(a);
    return set;
  }
  function ensureSelArm() {
    const arms = activeArms();
    if (!selArm || arms.indexOf(selArm) < 0) selArm = arms[0];
  }
  function hwOf(arm) {
    const h = at(S, 'jog', 'hardware', arm);
    return isObj(h) ? h : null;
  }
  function hwKind(arm) { // 'real' | 'mock' | 'unknown'
    const h = hwOf(arm);
    if (!h || typeof h.real !== 'boolean') return 'unknown';
    return h.real ? 'real' : 'mock';
  }

  function renderArmSelectors() {
    const arms = activeArms();
    const html = arms.map((a) =>
      '<button type="button" class="armbtn" role="radio" aria-checked="' + (a === selArm) + '" data-arm="' + esc(a) + '">' +
      esc(armLabel(a)) + '</button>').join('');
    $$('.armsel').forEach((el) => setHTML(el, html));
  }
  document.addEventListener('click', (e) => {
    const b = e.target.closest && e.target.closest('.armbtn');
    if (!b) return;
    selArm = b.getAttribute('data-arm');
    renderAll();
  });

  // ------------------------------------------------------------------ header
  function renderHeader() {
    setText($('mode-label'), 'mode: ' + fmtAny(at(S, 'mode') != null ? S.mode : at(S, 'recorder', 'mode')));
    // hardware banners
    let hw = '';
    for (const a of allArms()) {
      const k = hwKind(a);
      const label = k === 'real' ? 'REAL ROBOT' : (k === 'mock' ? 'MOCK' : 'HW ?');
      hw += '<span class="hw ' + k + '">' + esc(armLabel(a)) + ': ' + label + '</span>';
    }
    setHTML($('hw-banners'), S ? hw : '<span class="hw unknown">no state</span>');
    // chips
    const img = at(S, 'health', 'image');
    const imgOk = isObj(img) ? img.ok === true : null;
    const age = isObj(img) ? img.age_s : null;
    const chips = [
      chip('Image' + (age != null ? ' ' + fmtNum(age, 1, ' s') : ''), imgOk),
      chip('camera_info', typeof at(S, 'health', 'camera_info') === 'boolean' ? S.health.camera_info : null),
      chip('Recorder', typeof at(S, 'health', 'recorder') === 'boolean' ? S.health.recorder : null),
      chip('Backend', failCount >= 3 ? false : (S ? true : null)),
    ];
    setHTML($('chips'), chips.join(''));
    renderJogLast();
  }
  function chip(label, ok) {
    const cls = ok === true ? 'ok' : (ok === false ? 'bad' : 'unk');
    const mark = ok === true ? '✓ ' : (ok === false ? '✗ ' : '? ');
    return '<span class="chip ' + cls + '">' + mark + esc(label) + '</span>';
  }
  function renderJogLast() {
    const el = $('jog-last');
    const last = at(S, 'jog', 'last');
    let cls = 'neutral';
    let text = 'Last jog: —';
    if (isObj(last)) {
      const ph = String(last.phase || '');
      if (ph === 'done' && last.ok !== false) cls = 'ok';
      else if (ph === 'planning' || ph === 'checking' || ph === 'executing') cls = 'warn';
      else if (ph === 'refused' || ph === 'failed' || ph === 'stopped' || last.ok === false) cls = 'bad';
      text = 'Last jog: ' + (ph || '—').toUpperCase() + (last.message ? ' — ' + last.message : '');
    }
    if (at(S, 'jog', 'busy') === true) text += '   [busy' + (S.jog.busy_arm ? ': ' + S.jog.busy_arm : '') + ']';
    if (stopNote && Date.now() - stopNote.t < 8000) {
      text = 'STOP: ' + stopNote.text + '   |   ' + text;
      if (!stopNote.ok) cls = 'bad';
    }
    el.className = 'banner ' + cls;
    setText(el, text);
    $('banner-fault').hidden = at(S, 'jog', 'fault') !== true;
  }

  // -------------------------------------------------------------------- STOP
  async function doStop() {
    if (typeof wrOk === 'function' && wrOk() && wrRunning()) api('POST', '/api/wrist/stop', {}); // also ends a wrist sequence
    const res = await api('POST', '/api/jog/stop', {});
    stopNote = { t: Date.now(), ok: !!res.ok, text: res.message || (res.ok ? 'sent' : 'failed') };
    if (!res.ok) showError('STOP: ' + (res.message || 'failed'));
    renderJogLast();
    setTimeout(renderJogLast, 8100);
  }
  $('stop').addEventListener('click', doStop);
  $('clear-fault').addEventListener('click', async () => {
    const res = await api('POST', '/api/jog/clear_fault', {});
    stopNote = { t: Date.now(), ok: !!res.ok, text: res.message || (res.ok ? 'fault cleared' : 'failed') };
    if (!res.ok) showError('Clear fault: ' + (res.message || 'failed'));
    renderJogLast();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !e.repeat) doStop();
  });

  // ------------------------------------------------------------------- setup
  let formDirty = false;
  let formSig = null;
  $('session-form').addEventListener('input', () => { formDirty = true; });

  function renderSetup() {
    const r = at(S, 'recorder');
    const facts = [
      ['Mode', at(S, 'mode') != null ? S.mode : at(r, 'mode')],
      ['Session dir', at(r, 'session_dir')],
      ['Active arms', Array.isArray(at(r, 'active_arms')) ? r.active_arms.join(', ') : null],
      ['Captures', at(r, 'capture_count')],
      ['Table board obs.', at(r, 'n_table_obs')],
    ];
    setHTML($('session-facts'), facts.map((f) => '<dt>' + esc(f[0]) + '</dt><dd>' + esc(fmtAny(f[1])) + '</dd>').join('') +
      (r ? '' : '<dt>Recorder</dt><dd>not running</dd>'));

    // prefill the form from session_info unless the operator is editing
    const si = at(r, 'session_info');
    const sig = JSON.stringify(si == null ? null : si);
    if (sig !== formSig && !formDirty) {
      formSig = sig;
      const g = (k) => (isObj(si) && si[k] != null ? String(si[k]) : '');
      $('f-temp').value = g('ambient_temp_c');
      $('f-rh').value = g('ambient_rh_pct');
      $('f-warm').value = g('warmup_min');
      $('f-note').value = g('operator_note');
    }

    // checklist
    const items = [];
    const img = at(S, 'health', 'image');
    items.push(['Camera image', isObj(img) ? img.ok === true : null,
      isObj(img) ? ('age ' + fmtNum(img.age_s, 2, ' s') + (Array.isArray(img.size) ? ', ' + img.size.join('×') : '')) : '—']);
    items.push(['camera_info', typeof at(S, 'health', 'camera_info') === 'boolean' ? S.health.camera_info : null, '']);
    items.push(['Recorder node', typeof at(S, 'health', 'recorder') === 'boolean' ? S.health.recorder : null, '']);
    for (const a of allArms()) {
      const L = armLabel(a);
      const age = at(S, 'jog', 'joint_states_age_s', a);
      items.push([L + ' joint states', typeof age === 'number' ? age < JOINT_AGE_OK_S : null, 'age ' + fmtNum(age, 2, ' s')]);
      // recorder status_json: tf = {left: {ok, reason}, ...}
      const tfv = perArm(at(r, 'tf'), a);
      const tf = truthyFlag(tfv);
      items.push([L + ' TF', tf, tf === null ? '—' : (tf ? '' : fmtAny(isObj(tfv) ? tfv.reason : ''))]);
      const h = hwOf(a);
      let hok = null;
      if (h) hok = (h.state === 'active') && (h.controller_state === 'active');
      items.push([L + ' controller / hardware', hok,
        h ? (fmtAny(h.component) + ': ' + fmtAny(h.state) + ' | ' + fmtAny(h.controller) + ': ' + fmtAny(h.controller_state)) : '—']);
    }
    setHTML($('checklist'), items.map((it) => {
      const cls = it[1] === true ? 'ok' : (it[1] === false ? 'bad' : 'unk');
      const mark = it[1] === true ? '✓' : (it[1] === false ? '✗' : '–');
      return '<li class="' + cls + '"><span class="mark" aria-hidden="true">' + mark + '</span><span>' + esc(it[0]) +
        '</span><span class="detail">' + esc(it[2]) + '</span></li>';
    }).join(''));
  }
  $('session-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const num = (id) => { const v = $(id).value.trim(); if (v === '') return null; const n = Number(v); return isFinite(n) ? n : null; };
    const body = {
      ambient_temp_c: num('f-temp'),
      ambient_rh_pct: num('f-rh'),
      warmup_min: num('f-warm'),
      operator_note: $('f-note').value,
    };
    const res = await runAction('session', $('session-save'), $('session-result'),
      () => api('POST', '/api/session', body), 'Session');
    if (res && res.ok) { formDirty = false; formSig = null; }
  });

  // -------------------------------------------------------------- intrinsics
  function renderIntrinsics() {
    const n = at(S, 'recorder', 'n_intrinsics');
    setText($('intr-count'), typeof n === 'number' ? String(n) : '—');
    const tgt = at(S, 'recorder', 'intrinsics_target');
    setText($('intr-target'), '(aim 25–40' + (tgt != null ? '; target: ' + fmtAny(tgt) : '') + ')');
    const jobs = isObj(at(S, 'jobs')) ? Object.values(S.jobs) : [];
    const mine = jobs.filter((j) => isObj(j) && /intrinsic/i.test(String(j.kind || '')));
    const job = mine.length ? mine[mine.length - 1] : null;
    if (!job) {
      setHTML($('intr-job'), '<span class="muted">no job yet</span>');
      $('intr-calibrate').disabled = pending.has('calibrate');
      return;
    }
    let tail = job.output_tail;
    if (Array.isArray(tail)) tail = tail.join('\n');
    const st = String(job.state || '');
    setHTML($('intr-job'),
      '<div><span class="state ' + esc(st) + '">' + esc(st.toUpperCase() || '—') + '</span>' +
      (job.returncode != null ? ' (return code ' + esc(job.returncode) + ')' : '') + '</div>' +
      (job.result_file ? '<div>Result file: <code>' + esc(job.result_file) + '</code></div>' : '') +
      '<pre aria-label="Job output">' + esc(tail || '') + '</pre>');
    $('intr-calibrate').disabled = st === 'running' || pending.has('calibrate');
  }
  $('intr-capture').addEventListener('click', () =>
    runAction('icap', $('intr-capture'), $('intr-result'), () => api('POST', '/api/intrinsics/capture', {}), 'Capture view'));
  $('intr-calibrate').addEventListener('click', () =>
    runAction('calibrate', $('intr-calibrate'), $('intr-result'), () => api('POST', '/api/intrinsics/calibrate', {}), 'Calibrate'));

  // ------------------------------------------------------------------ record
  function detOf(key) {
    const d = at(S, 'detections', key);
    return isObj(d) ? d : null;
  }
  function badge(label, d) {
    if (!d) return '<span class="badge unk">' + esc(label) + ': —</span>';
    let ok = null;
    for (const k of ['ok', 'detected', 'visible']) if (typeof d[k] === 'boolean') { ok = d[k]; break; }
    const n = typeof d.n === 'number' ? ' (' + d.n + ' pts)' : '';
    return '<span class="badge ' + (ok === true ? 'ok' : (ok === false ? 'bad' : 'unk')) + '">' +
      (ok === true ? '✓ ' : (ok === false ? '✗ ' : '')) + esc(label) + n + '</span>';
  }
  function renderRecord() {
    const arms = activeArms();
    let b = '';
    for (const a of arms) b += badge(armLabel(a) + ' board', detOf(a));
    b += badge('Table board', detOf('table'));
    setHTML($('det-record'), b);

    let c = '';
    for (const a of arms) {
      const n = at(S, 'recorder', 'counts', a);
      const st = truthyFlag(perArm(at(S, 'recorder', 'stationary'), a));
      const cls = typeof n === 'number' && n >= 30 ? 'ok' : 'unk';
      c += '<div class="count' + (a === selArm ? ' sel' : '') + '"><span>' + esc(armLabel(a)) + ' poses <span class="muted">(aim 30–40)</span></span>' +
        '<span><b class="' + cls + '">' + (typeof n === 'number' ? n : '—') + '</b>' +
        (st === null ? '' : ' <span class="chip ' + (st ? 'ok' : 'bad') + '">' + (st ? 'still' : 'moving') + '</span>') + '</span></div>';
    }
    const cc = at(S, 'recorder', 'capture_count');
    c += '<div class="count"><span>Captures (all)</span><b>' + (typeof cc === 'number' ? cc : '—') + '</b></div>';
    setHTML($('rec-counts'), c);
    renderSuggest();
    $('capture').disabled = pending.has('capture') || !selArm || !S || !S.recorder || autoMine();
    renderAuto();
  }

  // ------------------------------------------------------------ auto capture
  // Server-side (autocapture.py): it never moves the robot. This page owns a run it started; the run ends when this
  // page stops polling /api/state, when it is stopped here, on leaving the Record arm tab, or when the page closes.
  let autoSeq = null;        // newest auto-capture event already shown (null: not seen the first state yet)
  let autoWasMine = false;   // this page started a run that has not been seen to end yet
  function autoOn() { return !!(S && S.auto_capture && S.auto_capture.on === true); }
  function autoMine() { return autoOn() && S.auto_capture.owner_is_you === true; }
  function showAutoEvents(A) {
    const ev = Array.isArray(A.events) ? A.events : [];
    const top = ev.reduce((m, e) => Math.max(m, e.seq || 0), 0);
    if (autoSeq === null) { autoSeq = top; return; }     // do not replay the history of an earlier page
    for (const e of ev) {
      if (!(e.seq > autoSeq)) continue;
      // same result line as a manual capture; a failure of an automatic attempt is not an error banner
      if (autoWasMine || A.owner_is_you) setResult($('capture-result'), e.level === 'bad' ? 'bad' : (e.level === 'warn' ? 'warn' : 'ok'), 'Auto: ' + e.message);
    }
    autoSeq = Math.max(autoSeq, top);
  }
  function renderAuto() {
    const A = S && S.auto_capture;
    if (!A) { setText($('auto-status'), ''); return; }
    showAutoEvents(A);
    const mine = autoMine(), other = autoOn() && !mine;
    if (!A.on) autoWasMine = false;
    const btn = $('auto-toggle');
    btn.setAttribute('aria-pressed', String(mine));
    setText(btn, mine ? 'Auto capture: ON' : 'Auto capture: OFF');
    btn.disabled = pending.has('auto') || !S.recorder || other;
    $('auto-change').disabled = autoOn();
    if (autoOn()) $('auto-change').value = String(Math.round(A.min_change_deg || 2)); // else keep the operator's choice
    const st = $('auto-status');
    let txt = '';
    if (other) txt = 'Auto capture is running on another page';
    else if (mine) txt = (A.message || '') + '  [' + A.captured + ' captured, ' + A.skipped + ' skipped, ' + A.gave_up + ' gave up]';
    else if (A.message) txt = A.message;
    setText(st, txt);
    st.className = 'autostatus' + (mine ? ' on' : '');
  }
  $('auto-toggle').addEventListener('click', async () => {
    if (pending.has('auto')) return;
    const mine = autoMine();
    pending.add('auto');
    renderAll();
    let res;
    try {
      res = mine ? await api('POST', '/api/auto_capture/stop', {})
        : await api('POST', '/api/auto_capture/start', { min_change_deg: Number($('auto-change').value) });
    } catch (e) { res = { ok: false, message: String(e) }; }
    pending.delete('auto');
    if (res.ok) {
      autoWasMine = !mine;
      if (res.auto_capture && S) S.auto_capture = res.auto_capture;
      $('banner-error').hidden = true;
    } else {
      showError('Auto capture: ' + (res.message || 'failed'));
    }
    renderAll();
  });
  function stopAutoQuietly() { // leaving the tab / closing the page: stop at once instead of waiting for the 5 s heartbeat
    if (!autoMine()) return;
    try {
      fetch('/api/auto_capture/stop', { method: 'POST', keepalive: true, cache: 'no-store', body: '{}',
        headers: { 'X-Token': token, 'X-Page-Id': pageId, 'Content-Type': 'application/json' } }).catch(() => {});
    } catch (e) { /* the heartbeat timeout stops it anyway */ }
    S.auto_capture.on = false;
    S.auto_capture.owner_is_you = false;
  }
  window.addEventListener('pagehide', stopAutoQuietly);
  window.addEventListener('beforeunload', stopAutoQuietly);
  $('capture').addEventListener('click', () => {
    if (!selArm) return;
    runAction('capture', $('capture'), $('capture-result'),
      () => api('POST', '/api/capture', { arm: selArm }), 'Capture');
  });

  // ----------------------------------------------------------- suggested pose
  // Backend: suggest.py.  Nothing here moves the arm except "Move there", which needs a stored plan for the selected
  // suggestion, the jog's own enable on REAL hardware, and the confirm dialog.  STOP / Esc (global) cancels a move.
  let sgLast = null;          // last seen suggest.state, to report a finished search once
  let sgMove = null;          // {t, sawBusy} while a move started from this panel is being watched
  function sgS() { const g = at(S, 'suggest'); return isObj(g) ? g : null; }
  function sgItems() { const g = sgS(); return g && Array.isArray(g.items) ? g.items : []; }
  function sgSel() { const g = sgS(); return g && typeof g.selected === 'number' ? g.selected : null; }
  function sgFreshPlan() { // the jog's stored plan, only if it belongs to the selected suggestion
    const g = sgS(), mp = at(S, 'jog', 'move_plan');
    if (!g || !isObj(g.plan) || !isObj(mp)) return null;
    if (g.plan.plan_id !== mp.plan_id || g.plan.index !== g.selected) return null;
    return mp;
  }
  function yesNo(v) { return v === true ? '<span class="yes">yes</span>' : (v === false ? '<span class="no">no</span>' : '—'); }
  function renderSuggest() {
    const g = sgS();
    const items = sgItems(), sel = sgSel();
    const st = g ? String(g.state || 'idle') : 'idle';
    const supported = !!g && g.supported !== false;
    const computing = st === 'computing';
    const busy = at(S, 'jog', 'busy') === true || at(S, 'jog', 'stopping') === true || at(S, 'jog', 'fault') === true;
    const have = st === 'ready' && items.length > 0 && sel != null && sel < items.length;
    const plan = have ? sgFreshPlan() : null;
    const js = jogState();
    // status line
    let status = '';
    if (!g) status = 'Waiting for the backend.';
    else if (!supported) status = 'Suggestions are available in single-arm modes only.';
    else if (computing) status = 'Searching for a pose (IK + collision checks), this can take a minute…';
    else if (st === 'error') status = g.message || 'No suggestion.';
    else if (have) status = 'Suggestion ' + (sel + 1) + ' of ' + items.length + ' (shown in the live image).';
    else status = 'Press "Suggest pose" (needs at least 6 recorded samples of this arm).';
    setText($('sg-status'), status);
    setText($('sg-armnote'), supported && g && g.arm ? '(' + armLabel(g.arm) + ' arm)' : '');
    // detail
    let html = '';
    if (have) {
      const it = items[sel], gn = isObj(it.gains) ? it.gains : {};
      const before = gn.rot_spread2_before, after = gn.rot_spread2_after;
      html += '<div class="sg-title">#' + esc(it.rank != null ? it.rank : sel + 1) + ': ' + esc(it.description || '') + '</div><dl>' +
        '<dt>Rotation spread</dt><dd>' + fmtNum(before, 3) + ' → ' + fmtNum(after, 3) + '</dd>' +
        '<dt>Information gain</dt><dd>' + fmtNum(gn.info_gain, 3) + '</dd>' +
        '<dt>New image area</dt><dd>' + yesNo(typeof gn.empty_cell === 'boolean' ? gn.empty_cell : (gn.empty_cell == null ? null : !!gn.empty_cell)) + '</dd>' +
        '<dt>Max joint change</dt><dd>' + fmtNum(gn.max_joint_change_deg, 1, '°') + '</dd>' +
        '<dt>Distance to camera</dt><dd>' + fmtNum(gn.distance_m, 2, ' m') + '</dd>' +
        '<dt>View angle</dt><dd>' + fmtNum(gn.view_angle_deg, 1, '°') + '</dd>' +
        (Array.isArray(it.q_deg) ? '<dt>Joints</dt><dd>' + esc(it.q_deg.map((v) => fmtNum(v, 1)).join(', ')) + '°</dd>' : '') +
        '</dl>';
      if (typeof g.based_on_samples === 'number') {
        html += '<div' + (g.stale ? ' class="stale"' : ' class="muted"') + '>based on ' + esc(g.based_on_samples) + ' samples' +
          (typeof g.session_samples === 'number' ? ', session now has ' + esc(g.session_samples) : '') +
          (g.stale ? ' — suggestions are stale, press "Suggest pose" again' : '') + '</div>';
      }
      if (plan) {
        html += '<div class="yes">Plan stored: ~' + esc(fmtNum(plan.duration_s, 1)) + ' s, max joint change ' +
          esc(fmtNum(plan.max_joint_change_deg, 1)) + '° (age ' + esc(fmtNum(plan.age_s, 0)) + ' s)</div>';
      }
    }
    setHTML($('sg-detail'), html);
    // buttons
    const free = !pending.has('sg') && !computing && !!S && failCount < 3 && !authBad;
    $('sg-compute').disabled = !free || !supported || !S.recorder || busy;
    $('sg-next').disabled = !free || !have || items.length < 2;
    $('sg-rviz').disabled = !free || !have;
    $('sg-plan').disabled = !free || !have || busy || !js.can;      // the jog applies its enable rules to planning too
    $('sg-plan').title = !have ? '' : (js.can ? '' : js.reason);
    $('sg-move').disabled = !free || !have || !plan || !js.can;
    $('sg-move').title = !have ? '' : (!plan ? 'Press "Plan preview" first' : (js.can ? '' : js.reason));
    // a finished search is reported once in the result line
    // (an error is already the status line above: reporting it again made the card show the text twice)
    if (g && sgLast === 'computing' && st !== 'computing' && st !== 'error') {
      setResult($('sg-result'), 'ok', g.message || st);
    }
    const rl = $('sg-result');           // never show the same text in the status line and the result line
    if (rl && st === 'error' && rl.textContent && rl.textContent === status) setResult(rl, '', '');
    sgLast = g ? st : null;
    // a move started here: report how it ended
    if (sgMove) {
      if (busy) sgMove.sawBusy = true;
      if (!busy && (sgMove.sawBusy || Date.now() - sgMove.t > 4000)) {
        const last = at(S, 'jog', 'last');
        const ok = isObj(last) && last.phase === 'done' && last.ok !== false;
        setResult($('sg-result'), ok ? 'ok' : 'bad', 'Move ' + (isObj(last) ? String(last.phase || 'ended') : 'ended') +
          (isObj(last) && last.message ? ': ' + last.message : '') +
          (ok ? '. Capture this pose (Auto capture takes it when ON), then press "Suggest pose" again.' : ''));
        sgMove = null;
      }
    }
  }
  async function sgAction(btn, fn, what) {
    const res = await runAction('sg', btn, $('sg-result'), fn, what);
    await pollOnce();
    return res;
  }
  $('sg-compute').addEventListener('click', async () => {
    const res = await sgAction($('sg-compute'), () => api('POST', '/api/suggest/compute', { arm: selArm }), 'Suggest pose');
    if (res && res.ok) sgLast = 'computing';    // so the finished search is reported even if it ended before the next poll
  });
  $('sg-next').addEventListener('click', () => {
    const n = sgItems().length, sel = sgSel();
    if (n < 2 || sel == null) return;
    sgAction($('sg-next'), () => api('POST', '/api/suggest/select', { index: (sel + 1) % n }), 'Next suggestion');
  });
  $('sg-rviz').addEventListener('click', () => {
    const sel = sgSel();
    if (sel == null) return;
    sgAction($('sg-rviz'), () => api('POST', '/api/suggest/send_rviz', { index: sel }), 'Send to RViz');
  });
  $('sg-plan').addEventListener('click', () => {
    const sel = sgSel();
    if (sel == null) return;
    sgAction($('sg-plan'), () => api('POST', '/api/suggest/plan', { index: sel }), 'Plan preview');
  });
  const mdlg = $('move-dialog');
  async function sgDoMove(planId) {
    const res = await sgAction($('sg-move'), () => api('POST', '/api/suggest/move', { plan_id: planId }), 'Move there');
    if (res && res.ok) sgMove = { t: Date.now(), sawBusy: false };
  }
  const KNIFE_WARNING = 'The knife has no collision model: keep the blade covered and the path clear.';
  // The plan the operator is shown in the REAL confirm dialog.  OK sends exactly this plan_id (never a re-fetched one):
  // if the stored plan was replaced meanwhile the server refuses the id and nothing moves.
  let moveShown = null;               // {plan_id, dur} captured when the dialog OPENS
  $('sg-move').addEventListener('click', () => {
    const plan = sgFreshPlan();
    if (!plan || !jogState().can) return;
    const dur = fmtNum(plan.duration_s, 0);
    if (hwKind(selArm) !== 'real') { sgDoMove(plan.plan_id); return; }
    moveShown = { plan_id: plan.plan_id, dur: dur };
    setText($('move-text'), 'Plan ' + plan.plan_id + ', duration ~' + dur + ' s');
    if (typeof mdlg.showModal === 'function') mdlg.showModal();
    else if (window.confirm('Move the REAL arm to the suggested pose? Plan ' + moveShown.plan_id + ', duration ~' + dur +
                            ' s. ' + KNIFE_WARNING)) { const shown = moveShown; moveShown = null; sgDoMove(shown.plan_id); }
    else moveShown = null;
  });
  $('move-cancel').addEventListener('click', () => { moveShown = null; mdlg.close(); });
  mdlg.addEventListener('cancel', () => { moveShown = null; });      // Esc
  $('move-ok').addEventListener('click', () => {
    const shown = moveShown;          // exactly what was displayed when the dialog opened
    moveShown = null;
    mdlg.close();
    if (shown && jogState().can) sgDoMove(shown.plan_id);
  });

  // ----------------------------------------------------------------- touches
  const KIND_TITLES = { board: 'Table board points', fit: 'FIT stickers', check: 'CHECK stickers (held out, never in a fit)',
    hover: 'Hover points the recorder does not list (hover only: add them to touch_stickers in the targets file to capture them)' };
  // hover points of the targets file: one group per tile ("tile A: 105, 106"), with the tile's own name for every chip
  function hoverGroups(items) {
    const by = new Map();
    for (const it of items) {
      const key = typeof it.tile === 'string' && it.tile ? it.tile : '';
      if (!by.has(key)) by.set(key, []);
      by.get(key).push(it);
    }
    return Array.from(by.entries()).map(([tile, its]) => ({
      title: tile ? tile + ': ' + its.map((it) => it.id).join(', ') + ' (hover only: add the ids to touch_stickers in the targets file to capture them)'
                  : KIND_TITLES.hover,
      items: its }));
  }
  function touchCount(arm, id) {
    const tc = at(S, 'recorder', 'touch_counts');
    if (!isObj(tc)) return null;
    let v = isObj(tc[arm]) ? tc[arm][String(id)] : undefined;
    if (v === undefined && typeof tc[String(id)] === 'number') v = tc[String(id)];
    if (v === undefined && isObj(tc[String(id)])) v = tc[String(id)][arm];
    return typeof v === 'number' ? v : null;
  }
  function touchDone(item, arm) {
    if (isObj(item.done) && typeof item.done[arm] === 'boolean') return item.done[arm];
    const ids = at(S, 'recorder', 'touched_ids', arm);
    return Array.isArray(ids) ? ids.indexOf(item.id) >= 0 : false;
  }
  function renderTouches() {
    const list = Array.isArray(at(S, 'touch_list')) ? S.touch_list.filter(isObj) : [];
    const arms = activeArms();
    if (selTouch == null && list.length) {
      const first = list.find((it) => !touchDone(it, selArm)) || list[0];
      selTouch = first.id;
    }
    let html = '';
    for (const kind of ['board', 'fit', 'check', 'hover']) {
      const items = list.filter((it) => it.kind === kind);
      if (!items.length) continue;
      const groups = kind === 'hover' ? hoverGroups(items) : [{ title: KIND_TITLES[kind], items }];
      for (const grp of groups) {
        html += '<div class="touch-group"><h3>' + esc(grp.title) + '</h3><div class="touch-items">';
        for (const it of grp.items) {
          const sel = it.id === selTouch;
          const marks = arms.map((a) => {
            const cnt = touchCount(a, it.id);
            const done = touchDone(it, a);
            return esc(armLabel(a).charAt(0)) + ' ' + (done ? '<span class="y">✓' + (cnt ? '×' + cnt : '') + '</span>' : '–');
          }).join(' &nbsp; ');
          const tname = typeof it.tile === 'string' && it.tile ? '<span class="ttile">' + esc(it.tile) + (it.tile_name ? ' · ' + esc(it.tile_name) : '') + '</span>' : '';
          html += '<button type="button" class="touch-item' + (sel ? ' sel' : '') + (kind === 'check' || kind === 'hover' ? ' checkk' : '') +
            '" data-tid="' + esc(it.id) + '" aria-pressed="' + sel + '" aria-label="Touch point ' + esc(it.id) + (it.tile ? ' (' + esc(it.tile) + ')' : '') + '">' +
            '<span class="tid">' + esc(it.id) + '</span>' + tname + '<span class="tdone">' + marks + '</span></button>';
        }
        html += '</div></div>';
      }
    }
    setHTML($('touch-list'), list.length ? html : '<p class="muted">no touch list yet (recorder not running)</p>');
    setText($('touch-sel'), selTouch != null ? 'point ' + selTouch + ', arm ' + (selArm ? armLabel(selArm) : '—') : 'select a point');
    const selItem = list.find((it) => it.id === selTouch);
    const notCapturable = !!selItem && (selItem.capturable === false || selItem.kind === 'hover');
    $('touch-capture').disabled = pending.has('touch') || selTouch == null || !selArm || !S || !S.recorder || notCapturable;
    $('touch-capture').title = notCapturable ? 'Not a touch point of the recorder (add it to touch_stickers in the targets file)' : '';
  }
  $('touch-list').addEventListener('click', (e) => {
    const b = e.target.closest('.touch-item');
    if (!b) return;
    const raw = b.getAttribute('data-tid');
    const n = Number(raw);
    selTouch = isFinite(n) ? n : raw;
    renderTouches();
  });
  $('touch-capture').addEventListener('click', () => {
    if (selTouch == null || !selArm) return;
    runAction('touch', $('touch-capture'), $('touch-result'),
      () => api('POST', '/api/touch', { arm: selArm, point_id: selTouch }), 'Touch');
  });

  // ------------------------------------------------- approach assist (hover)
  // Backend: hover_ui.py.  Detect points predicts every printed touch mark in the arm base frame from the arm's own
  // calibration fit.  Hover = IK pose 60 mm above the mark -> preview in RViz -> Move (REAL: confirm dialog bound to the
  // plan id; needs allow_real_move) or Send to RViz.  Descend to 10 mm = a straight Cartesian move along the paper normal
  // (like a jog step: REAL needs the page enable only).  STOP / Esc (global) stops both like a jog step.
  let hvSel = null;           // selected point id (click on a crosshair in the image or on a row; highlighted in both)
  let hvHeight = 20;          // final hover height above the mark in mm (10 / 20 / 30); the operator fine-tunes by teleop
  const HV_PICK_PX = 25;      // a click within this many displayed pixels of a predicted crosshair selects it
  const HV_TELEOP_HINT = 'Fine-tune with teleop (or the jog panel), release the keys, then Capture touch / Record offset.';
  let hvWatch = null;         // {t, sawBusy, what, tab} while a Move / Descend started here is being watched
  let hvMoveShown = null;     // {plan_id, dur, tab}: what the REAL confirm dialog showed when it OPENED
  const hvDlg = $('hover-dialog');
  function hvS() { const h = at(S, 'hover'); return isObj(h) ? h : null; }
  function hvPoints() { const h = hvS(); return h && Array.isArray(h.points) ? h.points.filter(isObj) : []; }
  function hvReady() { const h = hvS(); return !!h && h.supported !== false && h.state === 'ready'; }
  function hvSelId() {
    if (tab === 'touches' && selTouch != null && hvPoints().some((p) => p.id === selTouch && p.detected)) return selTouch;
    if (hvSel != null) return hvSel;
    const h = hvS();
    return h && typeof h.selected === 'number' ? h.selected : null;
  }
  function jogBusyNow() { return at(S, 'jog', 'busy') === true || at(S, 'jog', 'stopping') === true || at(S, 'jog', 'fault') === true; }
  function hvFreshPlan() { // the jog's stored plan, only if it belongs to the prepared point
    const h = hvS(), mp = at(S, 'jog', 'move_plan');
    if (!h || !isObj(h.plan) || !isObj(mp) || !isObj(h.prepared)) return null;
    if (h.plan.plan_id !== mp.plan_id || h.plan.point_id !== h.prepared.point_id) return null;
    return mp;
  }
  function hvQualityText(q) {
    if (!isObj(q)) return '';
    const out = [];
    for (const k of Object.keys(q)) {
      const v = q[k];
      if (typeof v === 'number' && isFinite(v)) out.push(k.replace(/_/g, ' ') + ' ' + (Number.isInteger(v) ? v : v.toFixed(2)));
      else if (typeof v === 'string' && v.length <= 24) out.push(k.replace(/_/g, ' ') + ' ' + v);
      if (out.length >= 3) break;
    }
    return out.join(', ');
  }
  function hvPosText(mm) {
    return Array.isArray(mm) && mm.length === 3 ? 'x ' + fmtNum(mm[0], 1) + '  y ' + fmtNum(mm[1], 1) + '  z ' + fmtNum(mm[2], 1) + ' mm (base)' : '';
  }
  // opt-in stopgap (hover_correction_<arm>_mm launch argument): a constant base-frame shift added to every prediction; shown
  // only while it is non-zero: "Prediction correction ACTIVE (left): -5.3 / +8.1 / +0.0 mm (base frame)"
  function hvCorrectionText(g) {
    if (!g || g.supported === false || !isObj(g.correction_mm) || typeof g.correction_arm !== 'string') return '';
    const v = g.correction_mm[g.correction_arm];
    if (!Array.isArray(v) || v.length !== 3 || !v.every((c) => typeof c === 'number' && isFinite(c)) || v.every((c) => c === 0)) return '';
    return 'Prediction correction ACTIVE (' + g.correction_arm + '): ' +
      v.map((c) => (c < 0 || Object.is(c, -0) ? '' : '+') + c.toFixed(1)).join(' / ') + ' mm (base frame)';
  }
  function hvBuild(root, mode) {
    root.innerHTML =
      '<h2>' + (mode === 'test' ? 'Hover test' : 'Approach assist') + ' <span class="muted hv-arm"></span></h2>' +
      '<div class="hv-corr" role="status" hidden></div>' +
      '<div class="row"><button type="button" class="primary" data-hv="compute">Detect points</button>' +
        '<label class="autochg">Hover height <select class="hv-height" aria-label="Final hover height above the mark">' +
        '<option value="10">10 mm</option><option value="20">20 mm</option><option value="30">30 mm</option></select></label>' +
        '<span class="autostatus hv-status" aria-live="polite"></span></div>' +
      '<div class="hv-rows"></div>' +
      '<div class="hv-plan"></div>' +
      '<p class="result" data-hvres aria-live="polite"></p>' +
      (mode === 'test' ? '<div class="hv-summary"></div>' : '') +
      '<p class="muted">' + (mode === 'test'
        ? 'Hover test: click a crosshair in the image (or a row), <b>Plan hover</b> (60 mm above the mark, preview in RViz), <b>Move</b>, ' +
          '<b>Descend</b> to the chosen hover height, then fine-tune the tip onto the mark by teleop (or the jog panel). ' +
          'Jog the tip onto the mark first, then <b>Record offset</b> (arm still): each press logs predicted vs actual to ' +
          '<code>hover_test.jsonl</code> in the session.'
        : 'Approach assist: <b>Detect points</b> predicts the marks from the arm\'s own calibration; click a crosshair in the image ' +
          'or a row to select it. <b>Plan hover</b> plans a pose 60 mm above the mark (preview in RViz), <b>Move</b> goes there, ' +
          '<b>Descend</b> goes straight down the paper normal to the hover height. Then fine-tune by teleop (or the jog panel), ' +
          'release the keys and <b>Capture touch</b>: the predicted vs touched offset is shown and logged ' +
          '(<code>hover_log.jsonl</code>). Nothing moves until you press Plan hover and then Move; STOP / Esc stops it.') + '</p>';
    root._built = mode;
  }
  function renderHover(root, mode) {
    if (!root) return;
    if (root._built !== mode) hvBuild(root, mode);
    const q = (sel) => root.querySelector(sel);
    const g = hvS();
    const supported = !!g && g.supported !== false;
    const st = g ? String(g.state || 'idle') : 'idle';
    const computing = st === 'computing';
    setText(q('.hv-arm'), supported && g && g.arm ? '(' + armLabel(g.arm) + ' arm, ' + fmtAny(g.calib_samples) + ' calibration samples)' : '');
    // status line
    let status = '';
    if (!g) status = 'Waiting for the backend.';
    else if (!supported) status = 'Approach assist is available in single-arm modes only.';
    else if (computing) status = 'Detecting points …';
    else if (st === 'error') status = g.message || 'No points.';
    else if (st === 'ready') {
      status = (g.message || 'Points detected.') + ' (' + fmtNum(g.age_s, 0, ' s') + ' ago)' +
        (g.stale ? ' — new calibration samples since: press "Detect points" again' : '');
    } else status = 'Press "Detect points" with the hover tiles / table board in view (needs at least 6 recorded samples of this arm).';
    setText(q('.hv-status'), status);
    const corrEl = q('.hv-corr'), corrTxt = hvCorrectionText(g);
    setText(corrEl, corrTxt);
    corrEl.hidden = !corrTxt;
    const free = !pending.has('hv') && !computing && !!S && failCount < 3 && !authBad;
    q('[data-hv="compute"]').disabled = !free || !supported || !S || !S.recorder;
    const hsel = q('.hv-height');
    if (hsel && hsel.value !== String(hvHeight)) hsel.value = String(hvHeight);
    if (hsel) hsel.disabled = !supported || pending.has('hv');
    // rows
    const pts = hvPoints();            // every point the targets file defines (same list in both tabs)
    const js = jogState();
    const busy = jogBusyNow();
    let rows = '';
    const srcIds = {};                 // source ("table", "tile A", ...) -> its point ids, for the group headers
    for (const p of pts) { const k = String(p.source || ''); (srcIds[k] = srcIds[k] || []).push(p.id); }
    const srcStatus = g && isObj(g.status) ? g.status : {};
    let lastSrc = null;
    for (const p of pts) {
      const srcName = String(p.source || '');
      if (srcName !== lastSrc) {         // one header per tile / board: "tile A: 101, 102, 103" + what the camera found
        lastSrc = srcName;
        rows += '<div class="hv-group"><b>' + esc(srcName || 'points') + '</b>: ' + esc(srcIds[srcName].join(', ')) +
          (srcStatus[srcName] && st !== 'idle' ? ' <span class="muted hv-gstatus">' + esc(srcStatus[srcName]) + '</span>' : '') + '</div>';
      }
      const det = p.detected === true;
      const sel = p.id === hvSelId();
      const canMove = free && det && js.can && !busy;
      const why = !det ? (p.reason || 'not detected') : (js.can ? '' : js.reason);
      // Descend: only from the pre-hover pose that Plan hover -> Move went to (for THIS point), and on the REAL robot only when
      // the UI was started with allow_real_move:=true (otherwise fine-tune with teleop)
      const realBlocked = hwKind(selArm) === 'real' && at(S, 'jog', 'allow_real_move') !== true;
      const reached = !!g && g.reached_point === p.id;
      const dWhy = !det ? why : (realBlocked ? 'Descend is disabled on the REAL robot (start the UI with allow_real_move:=true); use teleop instead'
        : (!reached ? 'Plan hover -> Move to the pre-hover pose first' : why));
      const canDescend = canMove && !realBlocked && reached;
      rows += '<div class="hv-row' + (det ? '' : ' off') + (sel ? ' sel' : '') + '" data-pid="' + esc(p.id) + '">' +
        '<span class="hv-id">' + esc(p.id) + '</span>' +
        '<span class="hv-name"><b>' + esc(p.name || '') + '</b> <span class="muted">' + esc(p.source || '') + '</span></span>' +
        '<span class="hv-btns">' +
          '<button type="button" class="small" data-hv="prepare" data-pid="' + esc(p.id) + '"' + (canMove ? '' : ' disabled') +
            ' title="' + esc(why || 'IK pose 60 mm above the mark + collision-checked plan (preview in RViz)') + '">Plan hover</button>' +
          '<button type="button" class="small" data-hv="descend" data-pid="' + esc(p.id) + '"' + (canDescend ? '' : ' disabled') +
            ' title="' + esc(dWhy || 'straight down the paper normal to ' + hvHeight + ' mm above the mark') + '">Descend to ' + hvHeight + ' mm</button>' +
          (mode === 'test' ? '<button type="button" class="small primary" data-hv="record" data-pid="' + esc(p.id) + '"' +
            ((free && det && !busy) ? '' : ' disabled') + ' title="' + esc(det ? 'tip on the mark, arm still' : why) + '">Record offset</button>' : '') +
        '</span>' +
        '<span class="hv-sub">' + (det ? esc(hvPosText(p.pred_base_mm)) + (hvQualityText(p.quality) ? ' · ' + esc(hvQualityText(p.quality)) : '')
                                       : (st === 'error' ? '' : 'not detected: ') + esc(p.reason || '')) + '</span>' +
        (isObj(p.last_offset) ? '<span class="hv-off">' + (p.last_offset.kind === 'touch' ? 'touch: ' : 'hover test: ') +
          esc(p.last_offset.text || '') + '</span>' : '') +
        '</div>';
    }
    setHTML(q('.hv-rows'), rows || '<p class="muted">' + (supported ? 'no points yet' : '') + '</p>');
    // prepared pose / plan / where the arm is now
    let plan = '';
    const prep = g && isObj(g.prepared) ? g.prepared : null;
    const dsc = g && isObj(g.descended) ? g.descended : null;
    const nameOf = (id) => (hvPoints().find((p) => p.id === id) || {}).name || '';
    if (supported && dsc) {
      const where = esc(nameOf(dsc.point_id)) + ' (' + esc(dsc.point_id) + ')';
      const ph = String(dsc.phase || '');
      if (ph === 'done') plan = '<div class="hv-prep"><span class="yes">At ' + esc(fmtNum(dsc.height_mm, 0)) + ' mm above ' + where + '.</span> ' + esc(HV_TELEOP_HINT) + '</div>';
      else if (['planning', 'checking', 'executing', 'stopping'].indexOf(ph) >= 0) plan = '<div class="hv-prep">Descending to ' + esc(fmtNum(dsc.height_mm, 0)) + ' mm above ' + where + ' …</div>';
      else plan = '<div class="hv-prep">The descend above ' + where + ' ended as "' + esc(ph) + '".</div>';
    } else if (prep && supported) {
      const fresh = hvFreshPlan();
      const where = esc(nameOf(prep.point_id)) + ' (' + esc(prep.point_id) + ')';
      if (g.reached_point === prep.point_id) {
        plan = '<div class="hv-prep"><span class="yes">At the pre-hover pose (' + esc(fmtNum(prep.height_mm, 0)) + ' mm above ' + where + ').</span> Descend to ' +
          esc(hvHeight) + ' mm, or fine-tune from here.</div>';
      } else if (String(g.interlock || '').indexOf('still running') >= 0) {
        plan = '<div class="hv-prep">Moving to the pre-hover pose above ' + where + ' …</div>';
      } else {
        plan = '<div class="hv-prep">Prepared: ' + where + ', ' + esc(fmtNum(prep.height_mm, 0)) +
          ' mm above the mark, IK pose ' + esc(prep.candidate + 1) + ' of ' + esc(prep.n_candidates) + ', max joint change ' +
          esc(fmtNum(prep.max_joint_change_deg, 1)) + '°' +
          (fresh ? ' — <span class="yes">plan stored: ~' + esc(fmtNum(fresh.duration_s, 1)) + ' s</span> (age ' + esc(fmtNum(fresh.age_s, 0)) + ' s)'
                 : ' — no stored plan (RViz only, or plan again)') + '</div>' +
          '<div class="row"><button type="button" class="danger" data-hv="move"' + ((free && fresh && js.can) ? '' : ' disabled') +
            ' title="' + esc(fresh ? (js.can ? '' : js.reason) : 'Press Plan hover first') + '">Move</button>' +
          '<button type="button" data-hv="send"' + (free ? '' : ' disabled') + '>Send to RViz</button></div>';
      }
    }
    setHTML(q('.hv-plan'), plan);
    // hover-test summary
    const sm = q('.hv-summary');
    if (sm) {
      const ts = g && isObj(g.test_summary) ? g.test_summary : null;
      let t = '';
      if (ts && ts.n > 0) {
        t = '<table><tr><th>Point</th><th>n</th><th>last dx / dy / dn (mm)</th><th>mean lateral</th><th>max lateral</th></tr>';
        for (const r of (Array.isArray(ts.points) ? ts.points : [])) {
          const l = isObj(r.last) ? r.last : {};
          t += '<tr><td>' + esc(r.point_id) + ' ' + esc(r.name || '') + '</td><td>' + esc(r.n) + '</td><td>' +
            esc(fmtNum(l.dx_mm, 1)) + ' / ' + esc(fmtNum(l.dy_mm, 1)) + ' / ' + esc(fmtNum(l.dn_mm, 1)) + '</td><td>' +
            esc(fmtNum(r.mean_lateral_mm, 2, ' mm')) + '</td><td>' + esc(fmtNum(r.max_lateral_mm, 2, ' mm')) + '</td></tr>';
        }
        t += '<tr class="total"><td>All</td><td>' + esc(ts.n) + '</td><td></td><td>' + esc(fmtNum(ts.mean_lateral_mm, 2, ' mm')) +
          '</td><td>' + esc(fmtNum(ts.max_lateral_mm, 2, ' mm')) + '</td></tr></table>';
      } else t = '<p class="muted">no offsets recorded in this session yet</p>';
      setHTML(sm, t);
    }
    hvWatchTick(root, mode);
  }
  function hvWatchTick(root, mode) {
    if (!hvWatch || hvWatch.tab !== mode) return;
    const busy = jogBusyNow();
    if (busy) hvWatch.sawBusy = true;
    if (!busy && (hvWatch.sawBusy || Date.now() - hvWatch.t > 4000)) {
      const last = at(S, 'jog', 'last');
      const ok = isObj(last) && last.phase === 'done' && last.ok !== false;
      const phase = isObj(last) ? String(last.phase || 'ended') : 'ended';
      const lm = isObj(last) && last.message ? String(last.message).replace(new RegExp('^' + phase + ':\\s*', 'i'), '') : '';
      setResult(root.querySelector('[data-hvres]'), ok ? 'ok' : 'bad', hvWatch.what + ' ' + phase + (lm ? ': ' + lm : '') +
        (ok && hvWatch.what === 'Descend' ? '. ' + HV_TELEOP_HINT : ''));
      hvWatch = null;
    }
  }
  async function hvAct(btn, kind) {
    const root = btn.closest('.hover-card');
    const mode = root.getAttribute('data-hvmode');
    const resEl = root.querySelector('[data-hvres]');
    const pid = Number(btn.getAttribute('data-pid'));
    const g = hvS();
    if (kind !== 'compute' && kind !== 'move' && kind !== 'send') hvSel = pid;
    let res;
    if (kind === 'compute') {
      res = await runAction('hv', btn, resEl, () => api('POST', '/api/hover/compute', { arm: selArm }), 'Detect points');
    } else if (kind === 'prepare') {
      res = await runAction('hv', btn, resEl, () => api('POST', '/api/hover/prepare', { point_id: pid, stage: 'prehover' }), 'Plan hover');
    } else if (kind === 'descend') {
      res = await runAction('hv', btn, resEl, () => api('POST', '/api/hover/descend', { point_id: pid, height_mm: hvHeight }), 'Descend');
      if (res && res.ok) hvWatch = { t: Date.now(), sawBusy: false, what: 'Descend', tab: mode };
    } else if (kind === 'record') {
      res = await runAction('hv', btn, resEl, () => api('POST', '/api/hover/record_offset', { point_id: pid }), 'Record offset');
    } else if (kind === 'send') {
      if (!g || !isObj(g.prepared)) return;
      hvSel = g.prepared.point_id;
      res = await runAction('hv', btn, resEl, () => api('POST', '/api/hover/send_rviz', { point_id: g.prepared.point_id }), 'Send to RViz');
    } else if (kind === 'move') {
      const plan = hvFreshPlan();
      if (!plan || !jogState().can) return;
      if (hwKind(selArm) !== 'real') { await hvDoMove(btn, plan.plan_id, mode); return; }
      // REAL: the dialog shows this plan; OK sends exactly this plan_id (never a re-fetched one), so a plan that was
      // replaced meanwhile is refused by the server and nothing moves
      hvMoveShown = { plan_id: plan.plan_id, dur: fmtNum(plan.duration_s, 0), tab: mode };
      setText($('hover-text'), 'Plan ' + plan.plan_id + ', duration ~' + hvMoveShown.dur + ' s');
      if (typeof hvDlg.showModal === 'function') hvDlg.showModal();
      else if (window.confirm('Move the REAL arm 60 mm above the mark? Plan ' + hvMoveShown.plan_id + ', duration ~' + hvMoveShown.dur +
                              ' s. ' + KNIFE_WARNING)) { const shown = hvMoveShown; hvMoveShown = null; await hvDoMove(btn, shown.plan_id, mode); }
      else hvMoveShown = null;
      return;
    }
    await pollOnce();
  }
  async function hvDoMove(btn, planId, mode) {
    const root = document.querySelector('.hover-card[data-hvmode="' + mode + '"]');
    const res = await runAction('hv', btn, root.querySelector('[data-hvres]'), () => api('POST', '/api/hover/move', { plan_id: planId }), 'Move');
    if (res && res.ok) hvWatch = { t: Date.now(), sawBusy: false, what: 'Move', tab: mode };
    await pollOnce();
  }
  $$('.hover-card').forEach((root) => {
    root.addEventListener('click', (e) => {
      const b = e.target.closest('[data-hv]');
      if (b && !b.disabled && root.contains(b)) { hvAct(b, b.getAttribute('data-hv')); return; }
      const row = e.target.closest('.hv-row');
      if (row && root.contains(row)) {
        const n = Number(row.getAttribute('data-pid'));
        hvSel = n;
        if (root.getAttribute('data-hvmode') === 'touches') selTouch = n;
        renderAll();
      }
    });
  });
  $$('.hover-card').forEach((root) => {     // (the selects are built later: delegate)
    root.addEventListener('change', (e) => {
      if (e.target.classList && e.target.classList.contains('hv-height')) {
        hvHeight = Number(e.target.value) || 20;          // ONE setting for both tabs: every selector follows at once
        $$('.hv-height').forEach((el) => { el.value = String(hvHeight); });
        renderAll();
      }
    });
  });
  function hvImageClick(img, e) {     // click near a predicted crosshair (displayed px, scaled to camera px) selects the point
    const pts = hvPoints().filter((p) => p.detected === true && Array.isArray(p.uv) && p.uv.length === 2);
    const size = at(S, 'health', 'image', 'size');
    if (!pts.length || !Array.isArray(size) || !(size[0] > 0) || !img.naturalWidth) return;
    const r = img.getBoundingClientRect();
    const nw = img.naturalWidth, nh = img.naturalHeight;
    const sc = Math.min(r.width / nw, r.height / nh);                  // object-fit: contain
    const ox = (r.width - nw * sc) / 2, oy = (r.height - nh * sc) / 2;
    const k = sc * nw / size[0];                                       // camera pixel -> displayed pixel
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    let best = null, bd = HV_PICK_PX;
    for (const p of pts) {
      const d = Math.hypot(ox + p.uv[0] * k - cx, oy + p.uv[1] * k - cy);
      if (d <= bd) { bd = d; best = p; }
    }
    if (!best) return;
    hvSel = best.id;
    if (tab === 'touches') selTouch = best.id;
    renderAll();
  }
  ['img-touches', 'img-hovertest'].forEach((id) => {
    const im = $(id);
    im.classList.add('pick');
    im.addEventListener('click', (e) => hvImageClick(im, e));
  });
  $('hover-cancel').addEventListener('click', () => { hvMoveShown = null; hvDlg.close(); });
  hvDlg.addEventListener('cancel', () => { hvMoveShown = null; });      // Esc
  $('hover-ok').addEventListener('click', () => {
    const shown = hvMoveShown;          // exactly what was displayed when the dialog opened
    hvMoveShown = null;
    hvDlg.close();
    if (shown && jogState().can) {
      const root = document.querySelector('.hover-card[data-hvmode="' + shown.tab + '"]');
      hvDoMove(null, shown.plan_id, shown.tab);
    }
  });

  // --------------------------------------------------------------- jog panel
  const jog = {
    mode: 'cart',            // cart | rot | joint
    frame: 'world',
    sizeIdx: { touches: {}, record: {} }, // per tab: {mode: index}
    inflight: false,
    cooldownUntil: 0,
    builtMode: null,
    builtSteps: null,
    pendingEnable: false,
  };
  const STEP_KEYS = { cart: 'cart_mm', rot: 'rot_deg', joint: 'joint_deg' };
  const STEP_UNITS = { cart: ' mm', rot: '°', joint: '°' };
  const panel = document.createElement('div');
  panel.className = 'jog';
  panel.setAttribute('role', 'group');
  panel.setAttribute('aria-label', 'Jog panel');
  panel.innerHTML =
    '<h2><span>Jog</span><span class="hwtag unknown" data-j="hwtag">—</span></h2>' +
    '<div class="seg" role="radiogroup" aria-label="Jog mode" data-j="mode">' +
      '<span class="lbl">Mode</span>' +
      '<button type="button" role="radio" data-mode="cart">Cartesian</button>' +
      '<button type="button" role="radio" data-mode="rot">Rotation</button>' +
      '<button type="button" role="radio" data-mode="joint">Joint</button></div>' +
    '<div class="seg" role="radiogroup" aria-label="Step size" data-j="steps"></div>' +
    '<div class="seg" role="radiogroup" aria-label="Frame" data-j="frame">' +
      '<span class="lbl">Frame</span>' +
      '<button type="button" role="radio" data-frame="world">World</button>' +
      '<button type="button" role="radio" data-frame="tool">Tool</button></div>' +
    '<div class="axes" data-j="axes"></div>' +
    '<div class="realbox" data-j="realbox" hidden>' +
      '<label><input type="checkbox" data-j="enable" aria-label="Enable motion on REAL robot"> Enable motion on REAL robot</label>' +
      '<div class="sub"><span data-j="remaining"></span><button type="button" class="small" data-j="lock">Lock</button></div></div>' +
    '<div class="jogstat" data-j="stat" aria-live="polite"></div>' +
    '<div class="limits" data-j="limits"></div>';
  const J = (name) => panel.querySelector('[data-j="' + name + '"]');

  function jogLimits() { return at(S, 'jog', 'limits'); }
  function stepList(mode) {
    const l = at(jogLimits(), 'steps', STEP_KEYS[mode]);
    return Array.isArray(l) ? l.filter((v) => typeof v === 'number') : [];
  }
  function jogKey() { return tab === 'record' ? 'record' : 'touches'; }   // Touches and Hover test: fine steps by default
  function defaultIdx(mode, forTab) {
    const n = stepList(mode).length;
    if (!n) return 0;
    return forTab === 'touches' ? 0 : Math.floor((n - 1) / 2);
  }
  function currentIdx() {
    const store = jog.sizeIdx[jogKey()];
    const n = stepList(jog.mode).length;
    let i = store[jog.mode];
    if (typeof i !== 'number' || i >= n) i = defaultIdx(jog.mode, jogKey());
    return i;
  }
  function currentSize() {
    const l = stepList(jog.mode);
    return l.length ? l[currentIdx()] : null;
  }
  function jogArm() { return selArm; }

  function jogState() {
    // returns {can, reason}
    const j = at(S, 'jog');
    const arm = jogArm();
    if (!isObj(j)) return { can: false, reason: 'No jog backend state yet.' };
    if (failCount >= 3) return { can: false, reason: 'Backend unreachable.' };
    if (authBad) return { can: false, reason: 'Token missing/invalid.' };
    if (!arm) return { can: false, reason: 'No arm selected.' };
    if (Array.isArray(j.arms) && j.arms.indexOf(arm) < 0) return { can: false, reason: 'Arm ' + armLabel(arm) + ' is not controlled by the jog backend.' };
    if (!stepList(jog.mode).length) return { can: false, reason: 'No step sizes from backend.' };
    if (jog.inflight) return { can: false, reason: 'Sending…' };
    if (j.busy === true) return { can: false, reason: 'Busy: ' + (j.busy_arm ? armLabel(j.busy_arm) + ' arm ' : '') + 'moving.' };
    if (Date.now() < jog.cooldownUntil) return { can: false, reason: 'Waiting for the arm…' };
    const k = hwKind(arm);
    if (k === 'unknown') return { can: false, reason: 'Hardware type of ' + armLabel(arm) + ' unknown: locked.' };
    if (k === 'real') {
      const mine = j.real_enabled === true && j.real_enabled_page === pageId;
      if (!mine) {
        return { can: false, reason: j.real_enabled === true
          ? 'Motion on the REAL robot is enabled from another page: locked here.'
          : 'REAL robot locked: tick "Enable motion on REAL robot".' };
      }
    }
    return { can: true, reason: 'Ready.' };
  }

  function buildAxes() {
    const mode = jog.mode;
    const axes = J('axes');
    axes.className = 'axes' + (mode === 'joint' ? '' : ' cart');
    let html = '';
    if (mode === 'joint') {
      for (let i = 1; i <= 6; i++) {
        html += '<div class="axis"><span class="name">J' + i + '</span>' +
          '<button type="button" data-axis="' + i + '" data-sign="-1" aria-label="Joint ' + i + ' minus">' + MINUS + '</button>' +
          '<button type="button" data-axis="' + i + '" data-sign="1" aria-label="Joint ' + i + ' plus">+</button></div>';
      }
    } else {
      for (const ax of ['x', 'y', 'z']) {
        const A = ax.toUpperCase();
        const nm = mode === 'rot' ? 'R' + ax : A;
        html += '<div class="axis"><span class="name">' + nm + '</span>' +
          '<button type="button" data-axis="' + ax + '" data-sign="-1" aria-label="' + (mode === 'rot' ? 'Rotate about ' : '') + A + ' minus">' + A + MINUS + '</button>' +
          '<button type="button" data-axis="' + ax + '" data-sign="1" aria-label="' + (mode === 'rot' ? 'Rotate about ' : '') + A + ' plus">' + A + '+</button></div>';
      }
    }
    axes.innerHTML = html;
    jog.builtMode = mode;
  }

  function updateJog() {
    if (!panel.isConnected) return;
    const arm = jogArm();
    const kind = arm ? hwKind(arm) : 'unknown';
    const tag = J('hwtag');
    tag.className = 'hwtag ' + kind;
    setText(tag, (arm ? armLabel(arm) + ': ' : '') + (kind === 'real' ? 'REAL ROBOT' : (kind === 'mock' ? 'MOCK' : 'UNKNOWN')));

    // mode / frame radios
    $$('[data-mode]', panel).forEach((b) => b.setAttribute('aria-checked', String(b.getAttribute('data-mode') === jog.mode)));
    $$('[data-frame]', panel).forEach((b) => b.setAttribute('aria-checked', String(b.getAttribute('data-frame') === jog.frame)));
    J('frame').hidden = jog.mode === 'joint';

    // step sizes
    const steps = stepList(jog.mode);
    const idx = currentIdx();
    const stepsHtml = '<span class="lbl">Step</span>' + steps.map((v, i) =>
      '<button type="button" role="radio" data-sidx="' + i + '" aria-checked="' + (i === idx) + '">' + esc(v) + STEP_UNITS[jog.mode] + '</button>').join('') +
      (steps.length ? '' : '<span class="muted">—</span>');
    setHTML(J('steps'), stepsHtml);

    if (jog.builtMode !== jog.mode) buildAxes();

    // enable gating
    const st = jogState();
    $$('[data-axis]', panel).forEach((b) => { b.disabled = !st.can; });

    // real-robot box
    const j = at(S, 'jog') || {};
    const rb = J('realbox');
    rb.hidden = kind === 'mock';
    const mine = j.real_enabled === true && j.real_enabled_page === pageId;
    const elsewhere = j.real_enabled === true && !mine;
    J('enable').checked = mine;
    J('enable').disabled = kind !== 'real' || elsewhere || jog.pendingEnable;
    J('lock').hidden = j.real_enabled !== true;
    let rem = '';
    if (mine) rem = 'Motion enabled: ' + fmtNum(j.real_enabled_remaining_s, 0, ' s') + ' left';
    else if (elsewhere) rem = 'Enabled from another page (' + fmtNum(j.real_enabled_remaining_s, 0, ' s') + ' left): locked here';
    else rem = 'Locked';
    setText(J('remaining'), rem);

    setText(J('stat'), st.reason);
    const l = jogLimits();
    setText(J('limits'), isObj(l)
      ? 'Max speed: ' + fmtNum(l.max_cart_speed_mm_s, 1, ' mm/s') + ', ' + fmtNum(l.max_rot_speed_deg_s, 1, ' °/s') +
        ', ' + fmtNum(l.max_joint_speed_rad_s, 2, ' rad/s') + '. One step per mouse click. ' +
        'Cartesian steps refused at a stretched pose (singularity)? Bend the elbow with a joint step first.'
      : 'Speed limits: —');
  }

  panel.addEventListener('click', (e) => {
    const t = e.target.closest('button, input');
    if (!t || !panel.contains(t)) return;
    if (t.hasAttribute('data-mode')) { jog.mode = t.getAttribute('data-mode'); updateJog(); return; }
    if (t.hasAttribute('data-frame')) { jog.frame = t.getAttribute('data-frame'); updateJog(); return; }
    if (t.hasAttribute('data-sidx')) {
      jog.sizeIdx[jogKey()][jog.mode] = Number(t.getAttribute('data-sidx'));
      updateJog();
      return;
    }
    if (t.hasAttribute('data-axis')) {
      if (e.detail === 0) return;           // keyboard-generated click (held Enter/Space would auto-repeat): mouse only
      const raw = t.getAttribute('data-axis');
      doStep(jog.mode === 'joint' ? Number(raw) : raw, Number(t.getAttribute('data-sign')));
      return;
    }
    if (t.getAttribute('data-j') === 'enable') {
      e.preventDefault(); // checkbox state is driven by backend state only
      const j = at(S, 'jog') || {};
      if (j.real_enabled === true && j.real_enabled_page === pageId) { doDisable(); } else { askEnable(); }
      return;
    }
    if (t.getAttribute('data-j') === 'lock') { doDisable(); }
  });

  async function doStep(axis, sign) {
    const st = jogState();
    if (!st.can) return;
    const size = currentSize();
    if (size == null) return;
    const body = { arm: jogArm(), mode: jog.mode, axis: axis, sign: sign, size: size, frame: jog.frame };
    jog.inflight = true;
    updateJog();
    const res = await api('POST', '/api/jog/step', body);
    jog.inflight = false;
    jog.cooldownUntil = Date.now() + 600; // let the next /api/state poll show busy
    if (!res.ok) showError('Jog: ' + (res.message || 'failed'));
    else setText($('jog-last'), 'Last jog: ' + (res.message || 'accepted'));
    updateJog();
    setTimeout(updateJog, 650);
  }

  const dlg = $('confirm-dialog');
  function askEnable() {
    if (!jogArm()) return;
    if (typeof dlg.showModal === 'function') {
      dlg.showModal();
    } else if (window.confirm('The arm will move. Keep the e-stop in reach.')) {
      doEnable();
    }
  }
  $('confirm-cancel').addEventListener('click', () => dlg.close());
  $('confirm-ok').addEventListener('click', () => { dlg.close(); doEnable(); });
  async function doEnable() {
    jog.pendingEnable = true;
    updateJog();
    const res = await api('POST', '/api/jog/enable', { arm: jogArm() });
    jog.pendingEnable = false;
    if (!res.ok) showError('Enable motion: ' + (res.message || 'failed'));
    await pollOnce();
  }
  async function doDisable() {
    const res = await api('POST', '/api/jog/disable', {});
    if (!res.ok) showError('Lock: ' + (res.message || 'failed'));
    await pollOnce();
  }

  // --------------------------------------------------------------------- TCP
  // Backend: tcp_ui.py.  The tab never moves the robot: the operator positions the arm with the Servo teleop keyboard and
  // re-seats the knife tip in a fixed countersink between poses.  Capture = fit / validate pose; Solve; mechanical check;
  // Write result (a YAML in the session directory) and the xacro patch as a diff to apply by hand (the UI never edits it).
  function tcpS() { const t = at(S, 'tcp'); return isObj(t) ? t : null; }
  function tcpVec(a, d) { return Array.isArray(a) ? a.map((x) => fmtNum(x, d == null ? 2 : d)).join(',  ') : '—'; }
  function tcpFmt(v, d, unit) { return typeof v === 'number' ? fmtNum(v, d, unit) : fmtAny(v); }
  function tcpChip(label, ok) { return chip(label, ok); }
  const TCP_VERDICT = { GOOD: 'ok', PASS: 'ok', FAIL: 'bad', INSUFFICIENT: 'warn' };
  function tcpKV(o) {
    if (o == null) return '—';
    if (typeof o === 'number') return fmtNum(o, 2);
    if (Array.isArray(o)) return o.map((x) => (typeof x === 'number' ? fmtNum(x, 2) : fmtAny(x))).join(', ');
    if (isObj(o)) return Object.keys(o).map((k) => esc(k) + ' ' + esc(tcpKV(o[k]))).join('; ');
    return fmtAny(o);
  }
  function tcpDiffHtml(diff) {
    return String(diff).split('\n').map((l) => {
      const cls = /^\+\+\+|^---/.test(l) ? 'hdr' : (l[0] === '+' ? 'add' : (l[0] === '-' ? 'del' : (l[0] === '@' ? 'hunk' : '')));
      return '<span class="dl ' + cls + '">' + esc(l) + '</span>';
    }).join('\n');
  }
  function tcpResultHtml(R) {
    if (!R) return '<p class="muted">no solve yet</p>';
    if (R.stale) {      // the numbers, outliers and reasons belong to poses that no longer exist: only this line
      return '<p class="tcp-stale"><b>Stale:</b> the poses changed since the last solve (' + esc(R.verdict || '?') +
        ', ' + esc(R.time || '') + '). Press Solve again.</p>';
    }
    const cls = TCP_VERDICT[R.verdict] || 'unk';
    let h = '<div class="tcp-verdict"><span class="verdict ' + cls + '">' + esc(R.verdict || '?') + '</span>' +
      ' <span class="muted">' + esc(R.time || '') + '</span></div>';
    if (Array.isArray(R.reasons) && R.reasons.length) h += '<ul class="tcp-reasons">' + R.reasons.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>';
    h += '<dl class="facts">' +
      '<dt>r in igus_tool0 (mm)</dt><dd><b>' + esc(tcpVec(R.r_mm)) + '</b></dd>' +
      '<dt>current URDF r (mm)</dt><dd>' + esc(tcpVec(R.current_r_mm)) + '</dd>' +
      '<dt>delta to current (mm)</dt><dd>' + esc(tcpKV(R.delta_to_current_mm)) + '</dd>' +
      '<dt>pivot point in base (mm)</dt><dd>' + esc(tcpVec(R.c_base_mm)) + '</dd>' +
      '<dt>1-sigma of r along tool0 x, y, z (mm)</dt><dd>' + esc(tcpVec(R.sigma_xyz_mm, 3)) + '</dd>' +
      '<dt>1-sigma principal directions (mm)</dt><dd>' + esc(tcpVec(R.sigma_axes_mm, 3)) + '</dd>' +
      '<dt>weakest direction (unit, tool0)</dt><dd>' + esc(tcpVec(R.weak_axis_tool0, 2)) + '</dd>' +
      '<dt>fit residual rms / max (mm)</dt><dd>' + esc(tcpFmt(R.rms_mm, 3)) + ' / ' + esc(tcpFmt(R.max_mm, 3)) + '</dd>' +
      '<dt>leave-one-out rms (mm)</dt><dd>' + esc(tcpFmt(R.loo_rms_mm, 3)) + '</dd>' +
      '<dt>held-out rms (mm)</dt><dd>' + esc(tcpFmt(R.heldout_rms_mm, 3)) + '</dd>' +
      '<dt>poses</dt><dd>' + esc(fmtAny(R.n_fit)) + ' fit, ' + esc(fmtAny(R.n_validate)) + ' validate</dd>' +
      '<dt>largest tilt / roll vs the mean pose (deg)</dt><dd>' + esc(tcpFmt(R.max_tilt_deg, 1)) + ' / ' + esc(tcpFmt(R.max_abs_roll_deg, 1)) + '</dd>' +
      '<dt>rank / cond / obs. min</dt><dd>' + esc(fmtAny(R.rank)) + ' / ' + esc(tcpFmt(R.cond_B, 1)) + ' / ' + esc(tcpFmt(R.obs_min, 3)) + '</dd></dl>';
    if (Array.isArray(R.outliers) && R.outliers.length) {
      h += '<p class="tcp-outlier-hint">Outlier pose(s) flagged: ' + R.outliers.map((i) => '#' + (i + 1)).join(', ') +
        '. Likely the tip was not re-seated in the same spot: use <b>Delete</b> on that row, re-capture a pose in that direction, solve again. Nothing is deleted automatically.</p>';
    }
    return h;
  }
  function tcpMechHtml(M) {
    if (!isObj(M)) return '';
    const c = isObj(M.check) ? M.check : null;
    let h = '<div class="tcp-mechbox"><b>Caliper ' + esc(tcpFmt(M.length_mm, 1, ' mm')) + ' vs ' + esc(M.r_source || '') + '</b>' +
      (c ? '<div>' + esc(c.text || '') + '</div><div class="muted">predicted (new model) ' + esc(tcpFmt(c.predicted_new_mm, 1, ' mm')) +
        ', delta ' + esc(tcpFmt(c.delta_new_mm, 1, ' mm')) + '; predicted (current URDF) ' + esc(tcpFmt(c.predicted_current_mm, 1, ' mm')) +
        ', delta ' + esc(tcpFmt(c.delta_current_mm, 1, ' mm')) + '</div>' : '') + '</div>';
    if (M.error) h += '<p class="result bad">' + esc(M.error) + '</p>';
    return h;
  }
  function renderTcp() {
    const T = tcpS();
    const supported = !T || T.supported !== false;
    $('tcp-unsupported').hidden = supported;
    $('tcp-body').hidden = !supported;
    if (T && !supported) { setText($('tcp-unsupported-text'), T.message || 'TCP calibration: single-arm modes only'); return; }
    if (!T) return;
    setText($('tcp-arm'), T.arm ? '(' + armLabel(T.arm) + ' arm)' : '');
    const th = isObj(T.thresholds) ? T.thresholds : {};
    setText($('tcp-still-s'), fmtNum(th.standstill_s, 1)); setText($('tcp-still-mrad'), fmtNum(th.standstill_mrad, 0));
    setText($('tcp-min-rot'), fmtNum(th.min_rotation_deg, 0));
    if (isObj(T.mechanical_note)) { setText($('tcp-mech-urdf'), fmtNum(T.mechanical_note.current_urdf_mm, 1)); setText($('tcp-mech-old'), fmtNum(T.mechanical_note.old_pivot_mm, 0)); }
    // checklist
    const items = isObj(T.checklist) && Array.isArray(T.checklist.items) ? T.checklist.items : [];
    const busy = pending.has('tcp');
    setHTML($('tcp-checklist'), items.map((it) =>
      '<li class="' + (it.checked ? 'ok' : 'bad') + '"><label><input type="checkbox" data-tcpck="' + esc(it.key) + '"' +
      (it.checked ? ' checked' : '') + (busy ? ' disabled' : '') + '> ' + esc(it.text) + '</label></li>').join(''));
    const allOk = !!(T.checklist && T.checklist.all_checked);
    setHTML($('tcp-warnings'), (Array.isArray(T.warnings) ? T.warnings : []).map((w) => '<div class="banner warn">' + esc(w) + '</div>').join(''));
    setHTML($('tcp-facts'),
      '<dt>Current URDF r (mm)</dt><dd>' + esc(tcpVec(T.current_r_mm)) + ' <span class="muted">' + esc(T.current_r_source || '') + '</span></dd>' +
      '<dt>Samples file</dt><dd>' + esc(fmtAny(T.file)) + '</dd>' +
      '<dt>Xacro (read only)</dt><dd>' + esc(fmtAny(T.xacro_path)) + '</dd>');
    // live guidance
    const L = isObj(T.live) ? T.live : null;
    if (L) {
      let h = tcpChip(L.stationary ? 'arm still' + (typeof L.spread_mrad === 'number' ? ' (' + fmtNum(L.spread_mrad, 2) + ' mrad)' : '') : 'arm moving / not settled', L.stationary === true) +
        tcpChip(L.tf_ok ? 'TF fresh (' + fmtNum(L.tf_age_s, 2) + ' s)' : 'TF not fresh', L.tf_ok === true);
      h += '<dl class="facts">' +
        '<dt>knife axis from vertical</dt><dd>' + esc(tcpFmt(L.tilt_vertical_deg, 1, '°')) + '</dd>' +
        '<dt>tilt vs the reference pose</dt><dd>' + esc(tcpFmt(L.tilt_deg, 1, '°')) + '</dd>' +
        '<dt>roll about the knife axis vs the reference pose</dt><dd>' + esc(tcpFmt(L.roll_deg, 1, '°')) + '</dd>' +
        '<dt>nearest stored pose</dt><dd>' + (L.nearest_index != null ? '#' + (L.nearest_index + 1) + ', ' + esc(fmtNum(L.nearest_deg, 1)) + '° apart' : '—') + '</dd></dl>';
      if (L.duplicate_of != null) h += '<p class="result bad">A capture here would be refused: less than ' + esc(fmtNum(th.min_rotation_deg, 0)) + '° from pose #' + (L.duplicate_of + 1) + '.</p>';
      if (!L.stationary && L.motion_problem) h += '<p class="muted">' + esc(L.motion_problem) + '</p>';
      if (!L.tf_ok && L.tf_problem) h += '<p class="muted">' + esc(L.tf_problem) + '</p>';
      setHTML($('tcp-live'), h);
    } else setHTML($('tcp-live'), '<p class="muted">no live data</p>');
    const C = isObj(T.coverage) ? T.coverage : null;
    const sg = $('tcp-suggestion');
    sg.hidden = !(C && C.suggestion);
    setText(sg, C && C.suggestion ? C.suggestion : '');
    setHTML($('tcp-plan'), C && Array.isArray(C.plan) ? C.plan.map((p) =>
      '<li class="' + (p.covered === true ? 'ok' : 'todo') + '"><span class="mark" aria-hidden="true">' + (p.covered === true ? '✓' : '○') +
      '</span><span><b>' + esc(p.label || p.id) + '</b> <span class="muted">(' + esc(p.role) + ')</span> ' +
      '<span class="tcp-cov">' + (p.covered === true ? 'covered' : 'not covered yet') + '</span><br>' + esc(p.text) + '</span></li>').join('') : '');
    setText($('tcp-plan-count'), C && Array.isArray(C.plan) ? '(' + C.plan.filter((p) => p.covered === true).length + ' of ' + C.plan.length + ' covered)' : '');
    // samples (+ solve columns)
    const R = isObj(T.result) ? T.result : null;
    const rows = new Map();
    if (R && Array.isArray(R.rows)) for (const r of R.rows) rows.set(r.index, r);
    const samples = Array.isArray(T.samples) ? T.samples : [];
    setText($('tcp-counts'), '(' + fmtAny(T.n_fit) + ' fit, ' + fmtAny(T.n_validate) + ' validate)');
    let tb = '<table class="tcp-table"><tr><th>#</th><th>role</th><th>time</th><th>tilt from vertical</th><th>tilt / roll vs ref.</th><th>nearest</th><th>residual</th><th>LOO</th><th>held-out</th><th></th></tr>';
    for (const s of samples) {
      const r = rows.get(s.index) || {};
      const outl = r.outlier === true && !(R && R.stale);
      tb += '<tr class="' + (outl ? 'outlier' : '') + '"><td>' + (s.index + 1) + '</td><td>' + esc(s.role) + '</td><td>' + esc((s.time || '').slice(11)) + '</td><td>' + esc(fmtNum(s.tilt_vertical_deg, 1)) + '°</td><td>' +
        esc(fmtNum(s.tilt_deg, 1)) + '° / ' + esc(fmtNum(s.roll_deg, 1)) + '°</td><td>' + esc(fmtNum(s.nearest_deg, 1)) + '°</td><td>' +
        esc(fmtNum(r.residual_mm, 2)) + '</td><td>' + esc(fmtNum(r.loo_mm, 2)) + '</td><td>' + esc(fmtNum(r.heldout_mm, 2)) + '</td><td>' +
        (outl ? '<span class="tcp-flag">outlier</span> ' : '') +
        '<button type="button" class="small" data-tcpdel="' + s.index + '" title="' + (outl ? 'Delete this pose and re-capture one in that direction' : 'Delete this pose (the file is kept as .bak)') + '"' + (busy ? ' disabled' : '') + '>Delete</button></td></tr>';
      if (s.note) tb += '<tr class="note"><td></td><td colspan="9">' + esc(s.note) + '</td></tr>';
    }
    tb += '</table>';
    setHTML($('tcp-samples'), samples.length ? tb : '<p class="muted">no poses captured yet</p>');
    // buttons
    const why = [];
    if (!allOk) why.push(T.checklist && T.checklist.expired
      ? 'the safety checklist expired (' + fmtNum(T.checklist.ttl_min, 0) + ' min without a capture): tick it again'
      : 'tick the whole safety checklist first');
    if (L && Array.isArray(L.blockers)) for (const b of L.blockers) why.push(b);
    for (const id of ['tcp-cap-fit', 'tcp-cap-val']) { $(id).disabled = busy || why.length > 0 || !S; $(id).title = why.join(' / '); }
    const whyEl = $('tcp-cap-why');
    whyEl.hidden = why.length === 0;
    setText(whyEl, why.length ? 'Capture is off: ' + why.join('; ') : '');
    $('tcp-solve').disabled = busy || !(T.n_fit > 0);
    $('tcp-mech-go').disabled = busy;
    $('tcp-write').disabled = busy || !R || R.stale === true || ['GOOD', 'PASS'].indexOf(R.verdict) < 0;
    $('tcp-write').title = !R ? 'Solve first' : (R.stale ? 'Poses changed: solve again' : (['GOOD', 'PASS'].indexOf(R.verdict) < 0 ? 'Only GOOD / PASS results are written' : ''));
    setHTML($('tcp-result'), tcpResultHtml(R));
    setHTML($('tcp-mech'), tcpMechHtml(T.mechanical));
    const mi = $('tcp-mech-len');
    if (isObj(T.mechanical) && typeof T.mechanical.length_mm === 'number' && mi.value === '' && document.activeElement !== mi) mi.value = String(T.mechanical.length_mm);
    const wr = isObj(T.written) ? T.written : null;
    setHTML($('tcp-written'), wr ? '<p class="result ok">Result file: <code>' + esc(wr.path) + '</code> (' + esc(wr.verdict) + ', ' + esc(wr.time) + ')</p>' : '');
    const dEl = $('tcp-diff');
    setHTML(dEl, T.diff ? tcpDiffHtml(T.diff) : (T.diff_error ? '<span class="dl">' + esc(T.diff_error) + '</span>' : '<span class="dl muted">(solve to see the patch)</span>'));
    setHTML($('tcp-steps'), (Array.isArray(T.apply_steps) ? T.apply_steps : []).map((x) => '<li>' + esc(x) + '</li>').join(''));
    setText($('tcp-rebase'), T.rebase_warning || '');
  }
  async function tcpAct(name, btn, resultEl, path, body, what) {
    const res = await runAction(name, btn, resultEl, () => api('POST', path, body), what);
    await pollOnce();
    return res;
  }
  $('tcp-checklist').addEventListener('change', async (e) => {
    const k = e.target && e.target.getAttribute && e.target.getAttribute('data-tcpck');
    if (!k) return;
    const res = await api('POST', '/api/tcp/checklist', { items: { [k]: !!e.target.checked } });
    if (!res.ok) showError('Checklist: ' + (res.message || 'failed'));
    await pollOnce();
  });
  $('tcp-cap-fit').addEventListener('click', () =>
    tcpAct('tcp', $('tcp-cap-fit'), $('tcp-cap-result'), '/api/tcp/capture', { role: 'fit', note: $('tcp-note').value }, 'Capture FIT pose'));
  $('tcp-cap-val').addEventListener('click', () =>
    tcpAct('tcp', $('tcp-cap-val'), $('tcp-cap-result'), '/api/tcp/capture', { role: 'validate', note: $('tcp-note').value }, 'Capture VALIDATE pose'));
  $('tcp-solve').addEventListener('click', () => tcpAct('tcp', $('tcp-solve'), $('tcp-solve-result'), '/api/tcp/solve', {}, 'Solve'));
  $('tcp-write').addEventListener('click', () => tcpAct('tcp', $('tcp-write'), $('tcp-write-result'), '/api/tcp/write', {}, 'Write result'));
  $('tcp-mech-go').addEventListener('click', () => {
    const v = Number($('tcp-mech-len').value);
    if ($('tcp-mech-len').value.trim() === '' || !isFinite(v)) { setResult($('tcp-mech-result'), 'bad', 'enter the measured length in mm'); return; }
    tcpAct('tcp', $('tcp-mech-go'), $('tcp-mech-result'), '/api/tcp/mechanical', { length_mm: v }, 'Mechanical check');
  });
  $('tcp-samples').addEventListener('click', (e) => {
    const b = e.target.closest && e.target.closest('[data-tcpdel]');
    if (!b || b.disabled) return;
    const idx = Number(b.getAttribute('data-tcpdel'));
    if (!window.confirm('Delete pose #' + (idx + 1) + '? The samples file is kept as tcp_pivot.jsonl.bak. Re-capture a pose in that direction afterwards.')) return;
    tcpAct('tcp', b, $('tcp-cap-result'), '/api/tcp/delete', { index: idx }, 'Delete pose');
  });

  // ------------------------------------------------------------------- tabs
  const TAB_IMG = {
    intrinsics: { id: 'img-intrinsics', overlay: 'intrinsics' },
    record: { id: 'img-record', overlay: 'detect' },
    touches: { id: 'img-touches', overlay: 'detect' },
    hovertest: { id: 'img-hovertest', overlay: 'detect' },
    wrist: { id: 'img-wrist', url: '/api/wrist/image.jpg?w=960' },
  };
  function switchTab(name) {
    if (tab === 'record' && name !== 'record') stopAutoQuietly();
    tab = name;
    $$('.tab').forEach((b) => {
      const on = b.getAttribute('data-tab') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
    });
    $$('.page').forEach((p) => { p.hidden = p.id !== 'page-' + name; });
    if (name === 'record' || name === 'touches' || name === 'hovertest' || name === 'tcp' || name === 'wrist') {
      $('jog-slot-' + name).appendChild(panel);
      builtReset();
    } else if (panel.parentNode) {
      panel.parentNode.removeChild(panel);
    }
    restartImageLoop();
    renderAll();
  }
  function builtReset() { jog.builtMode = null; }
  $$('.tab').forEach((b) => b.addEventListener('click', () => switchTab(b.getAttribute('data-tab'))));


  // ------------------------------------------------------------ wrist camera
  // Backend: wrist_ui.py.  Only "Start auto sequence" moves the robot, through the jog "Move there" machinery.
  function wrS() { const w = at(S, 'wrist'); return isObj(w) ? w : null; }
  function wrOk() { const w = wrS(); return !!w && w.supported === true; }
  function renderWristTabVisibility() {
    const b = $('tab-wrist');
    if (b) b.hidden = !wrOk();
  }
  function wrFmt(v, d) { return typeof v === 'number' ? v.toFixed(d == null ? 1 : d) : '—'; }
  function wrRunning() { const w = wrS(); return !!w && isObj(w.run) && (w.run.state === 'running' || w.run.state === 'stopping'); }
  function renderWrist() {
    const w = wrS();
    if (!w || w.supported !== true) {
      setText($('wr-msg'), w && w.message ? w.message : 'Wrist-camera calibration needs a single-arm mode (single_left / single_right).');
      return;
    }
    setText($('wr-armnote'), '(arm ' + armLabel(w.arm) + ', ' + (w.hardware_real ? 'REAL hardware' : 'mock') + ')');
    const b = w.board;
    setText($('wr-board'), b ? ('board [' + b.source + ']: centre ' + (b.centre_m || []).map((x) => wrFmt(x, 3)).join(', ') + ' m, ' +
      wrFmt(b.dist_from_base_axis_m, 3) + ' m from the base axis, tilt ' + wrFmt(b.tilt_from_vertical_deg, 1) + '° from vertical' +
      (b.warning ? ' — ' + b.warning : '')) : 'board not located yet');
    const sc = w.scene;
    setText($('wr-scene'), sc && sc.applied ? ('applied: ' + (sc.boxes || []).map((x) => x.id + ' (top z ' + wrFmt(x.top_z, 3) + ')').join('; ') +
      (sc.stale ? ' — STALE: board moved, re-add' : '')) : 'safety scene not applied');
    const sd = w.seed;
    setText($('wr-seedinfo'), sd ? ('seed set' + (sd.mock_no_image ? ' (mock, no image)' : '') +
      (sd.detection ? ': ' + sd.detection.n_corners + ' corners, board at ' + wrFmt(sd.detection.distance_m, 2) + ' m' : '')) : 'no seed yet');
    const v = w.views || {};
    const st = v.stats || {};
    const rej = st.rejected ? Object.keys(st.rejected).map((k) => k + ': ' + st.rejected[k]).join(', ') : '';
    setText($('wr-views'), (v.state || 'idle') + (v.message ? ' — ' + v.message : '') +
      (st.estimate ? ' (mount estimate: ' + st.estimate + ')' : '') + (rej ? '. Rejected: ' + rej : ''));
    setHTML($('wr-view-list'), (v.items || []).map((it) => '<div>' + (it.i + 1) + '. ' + esc(it.desc) +
      (it.max_joint_change_deg != null ? ' — max joint change ' + wrFmt(it.max_joint_change_deg, 0) + '°' : '') + '</div>').join(''));
    const sm = w.samples || {};
    setText($('wr-samples'), 'samples: ' + (sm.n || 0) + (sm.interim ? ' — interim RMS ' + wrFmt(sm.interim.rms_px, 2) + ' px, span ' + wrFmt(sm.interim.span_deg, 0) + '°' : ''));
    setHTML($('wr-cov'), (sm.coverage || []).map((row) => '<span class="badge">' + row.join(' ') + '</span>').join(''));
    setHTML($('wr-sample-list'), (sm.items || []).map((it) => '<div>#' + it.index + ': ' + it.n_corners + ' corners, PnP ' + wrFmt(it.pnp_rms_px, 2) + ' px</div>').join(''));
    const r = w.run || {};
    setText($('wr-run'), (r.state || 'idle') + (r.state && r.state !== 'idle' ? ' — view ' + (r.index || 0) + '/' + (r.total || 0) + ', skipped ' + (r.skipped || 0) : '') +
      (r.message ? ': ' + r.message : ''));
    setText($('wr-log'), (r.log || []).join('\n'));
    const R = w.result;
    if (R) {
      setHTML($('wr-result'), '<div><b>' + esc(R.verdict) + '</b>: RMS ' + wrFmt(R.rms_px, 2) + ' px (max ' + wrFmt(R.max_px, 2) + '), ' + R.n_used +
        ' samples used, held-out ' + wrFmt(R.heldout_position_mm, 2) + ' mm, rotation span ' + wrFmt(R.diversity && R.diversity.span_deg, 0) + '°, ' +
        (R.diversity ? R.diversity.n_axes : '—') + ' axes</div>' +
        '<div>xyz ' + (R.xyz_m || []).map((x) => wrFmt(x, 4)).join(', ') + ' m; rpy ' + (R.rpy_deg || []).map((x) => wrFmt(x, 2)).join(', ') + '°</div>' +
        (R.scene_cross_check ? '<div>board vs scene camera: ' + wrFmt(R.scene_cross_check.position_mm, 1) + ' mm / ' + wrFmt(R.scene_cross_check.rotation_deg, 2) + '°</div>' : '') +
        (R.reasons || []).map((x) => '<div class="muted">' + esc(x) + '</div>').join('') +
        (R.files && R.files.yaml ? '<div class="muted">' + esc(R.files.yaml) + '</div>' : ''));
    } else { setHTML($('wr-result'), ''); }
    const e = w.estimate;
    setText($('wr-estimate'), (e ? 'camera mount estimate used for predictions: ' + e.source : '') + (w.tf_published ? ' — static TF published' : ''));
    const running = wrRunning();
    const busy = jogBusyNow();
    $('wr-start').disabled = running || !$('wr-confirm').checked || (v.state !== 'ready');
    $('wr-stop').disabled = !running;
    for (const id of ['wr-locate', 'wr-mockboard', 'wr-scene-add', 'wr-scene-rm', 'wr-seed', 'wr-capture', 'wr-generate', 'wr-solve', 'wr-write', 'wr-pubtf']) {
      $(id).disabled = running || pending.has(id);
    }
    $('wr-mockboard').hidden = w.hardware_real === true;
    if (busy && !running) setText($('wr-armnote'), $('wr-armnote').textContent + ' — arm busy');
  }
  function wrBind(id, path, what, bodyFn) {
    $(id).addEventListener('click', () => runAction(id, $(id), $('wr-msg'), () => api('POST', path, bodyFn ? bodyFn() : {}), what)
      .then(() => pollOnce()));
  }
  wrBind('wr-locate', '/api/wrist/locate_board', 'Locate board');
  wrBind('wr-mockboard', '/api/wrist/mock_board', 'Mock board', () => ({ distance_m: 0.4 }));
  wrBind('wr-scene-add', '/api/wrist/scene_add', 'Safety scene');
  wrBind('wr-scene-rm', '/api/wrist/scene_remove', 'Remove safety scene');
  wrBind('wr-seed', '/api/wrist/seed', 'Seed');
  wrBind('wr-capture', '/api/wrist/capture', 'Capture');
  wrBind('wr-generate', '/api/wrist/generate', 'Generate views');
  $('wr-start').addEventListener('click', () => { setTimeout(() => { $('wr-confirm').checked = false; renderWrist(); }, 0); }); // one confirmation per run
  wrBind('wr-start', '/api/wrist/start', 'Start sequence', () => ({ confirm: $('wr-confirm').checked === true }));
  wrBind('wr-solve', '/api/wrist/solve', 'Solve');
  wrBind('wr-write', '/api/wrist/write', 'Write result');
  wrBind('wr-pubtf', '/api/wrist/publish_tf', 'Publish TF');
  $('wr-confirm').addEventListener('change', () => { renderWrist(); });
  $('wr-stop').addEventListener('click', async () => {
    const res = await api('POST', '/api/wrist/stop', {});
    report($('wr-msg'), res, 'Stop sequence');
    await pollOnce();
  });

  // ------------------------------------------------------------- image loop
  let imgGen = 0;
  let imgUrl = null; // current object URL of the visible frame
  function restartImageLoop() {
    imgGen++;
    // drop stale frame from all views
    $$('.imgwrap img').forEach((im) => { im.removeAttribute('src'); });
    if (imgUrl) { URL.revokeObjectURL(imgUrl); imgUrl = null; }
    const cfg = TAB_IMG[tab];
    if (cfg) imageTick(imgGen);
  }
  async function imageTick(gen) {
    if (gen !== imgGen) return;
    const cfg = TAB_IMG[tab];
    if (!cfg) return;
    if (document.hidden) { setTimeout(() => imageTick(gen), 500); return; }
    const img = $(cfg.id);
    const noimg = img.parentNode.querySelector('.noimg');
    const t0 = Date.now();
    let delay = IMG_PERIOD_MS;
    try {
      const sgShow = tab === 'record' && sgItems().length > 0 && sgSel() != null && at(sgS(), 'state') === 'ready';
      const hvShow = (tab === 'touches' || tab === 'hovertest') && hvReady();
      const url = cfg.url ? cfg.url + '&t=' + t0 : '/api/image.jpg?w=960&overlay=' + cfg.overlay + (sgShow ? '&suggest=1' : '') +
        (hvShow ? '&hover=1' + (hvSelId() != null ? '&sel=' + hvSelId() : '') : '') + '&t=' + t0;
      const resp = await fetch(url, { headers: { 'X-Token': token, 'X-Page-Id': pageId }, cache: 'no-store' });
      if (gen !== imgGen) return;
      if (resp.status === 401) { authBad = true; updateBanners(); }
      if (resp.ok) {
        const blob = await resp.blob();
        if (gen !== imgGen) return;
        const obj = URL.createObjectURL(blob);
        await new Promise((resolve) => { // wait until this frame loaded before requesting the next
          img.onload = resolve; img.onerror = resolve; img.src = obj;
        });
        if (gen !== imgGen) { URL.revokeObjectURL(obj); return; }
        if (imgUrl) URL.revokeObjectURL(imgUrl);
        imgUrl = obj;
        noimg.hidden = true;
      } else {
        img.removeAttribute('src');
        noimg.hidden = false;
        delay = 1000;
      }
    } catch (e) {
      if (gen !== imgGen) return;
      img.removeAttribute('src');
      noimg.hidden = false;
      delay = 1000;
    }
    setTimeout(() => imageTick(gen), Math.max(0, delay - (Date.now() - t0)));
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) restartImageLoop(); });

  // ----------------------------------------------------------------- render
  function renderAll() {
    ensureSelArm();
    renderHeader();
    renderArmSelectors();
    if (tab === 'tcp') renderTcp();
    else if (tab === 'setup') renderSetup();
    else if (tab === 'intrinsics') renderIntrinsics();
    else if (tab === 'record') renderRecord();
    else if (tab === 'touches') { renderTouches(); renderHover($('hover-card-touches'), 'touches'); }
    else if (tab === 'hovertest') renderHover($('hover-card-test'), 'test');
    else if (tab === 'wrist') renderWrist();
    renderWristTabVisibility();
    updateJog();
  }

  // ------------------------------------------------------------------- poll
  async function pollOnce() {
    const r = await api('GET', '/api/state');
    if (r.status === 200 && r.ok === true) {
      S = r;
      failCount = 0;
    } else if (r.status === 401) {
      failCount = 0; // reachable, just unauthorised
    } else {
      failCount++;
    }
    updateBanners();
    try { renderAll(); } catch (e) { console.error('render failed', e); }
  }
  async function pollLoop() {
    try { await pollOnce(); } catch (e) { console.error(e); }
    setTimeout(pollLoop, document.hidden ? 2000 : POLL_MS);
  }

  // ------------------------------------------------------------------- init
  switchTab('setup');
  pollLoop();
})();
