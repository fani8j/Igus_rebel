// Runs static/app.js against a stub DOM and a canned /api/state, to catch runtime errors in every render path.
// usage: node frontend_smoke.js app.js state.json [present]
const fs = require('fs');
const [appPath, statePath, mode] = process.argv.slice(2);
const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
const elements = {};
const listeners = {};
const TABS = ['summary', 'detection', 'phases', 'plan', 'log'];
function makeEl(id, extra) {
  const el = Object.assign({ id, dataset: {}, style: {}, hidden: false, textContent: '', innerHTML: '', className: '', disabled: false, scrollTop: 0, scrollHeight: 0,
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    addEventListener(ev, fn) { (listeners[id + ':' + ev] = listeners[id + ':' + ev] || []).push(fn); },
    setAttribute(k, v) { el['attr_' + k] = v; }, removeAttribute() {}, querySelectorAll() { return []; }, querySelector() { return makeEl(id + '>'); },
    showModal() { throw new Error('showModal() must not be used: it makes the Software stop button inert'); },
    show() { el.open = true; el.opened = true; if (id === 'dlg-confirm' && autoYes) setImmediate(() => el_yes()); },
    close() { const was = el.open; el.open = false; el.opened = false; if (was) for (const fn of listeners[id + ':close'] || []) fn(); } }, extra || {});
  return el;
}
let autoYes = true;
const el_yes = () => { const y = elements['dlg-confirm-yes']; if (y && y.onclick) y.onclick(); };
const tabEls = TABS.map((t) => makeEl('tab-' + t, { dataset: { tab: t } }));
for (const t of tabEls) elements[t.id] = t;
const document = { hidden: false, getElementById(id) { return elements[id] = elements[id] || makeEl(id); },
  querySelectorAll(sel) { return sel === '.tab' ? tabEls : []; }, addEventListener(ev, fn) { (listeners['document:' + ev] = listeners['document:' + ev] || []).push(fn); }, body: { classList: { add() {} } } };
const calls = [];
global.document = document;
global.window = { location: { href: 'http://127.0.0.1:1/' + (mode === 'present' ? '?view=present' : '') + '?token=T', search: mode === 'present' ? '?view=present' : '' }, crypto: { randomUUID: () => 'page-1' } };
global.crypto = global.window.crypto;
const store = {};
global.sessionStorage = { getItem: (k) => store[k] || null, setItem: (k, v) => { store[k] = v; } };
const lstore = process.env.SMOKE_LOCALSTORAGE ? JSON.parse(process.env.SMOKE_LOCALSTORAGE) : {};
global.localStorage = { getItem: (k) => (k in lstore ? lstore[k] : null), setItem: (k, v) => { lstore[k] = String(v); } };
global.history = { replaceState() {} };
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = () => 1;
global.cancelAnimationFrame = () => {};
global.setInterval = () => 1;
global.setTimeout = () => 1;
global.clearTimeout = () => {};
global.URL = Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL() {} });
global.fetch = async (path, opts) => {
  calls.push({ path, method: (opts && opts.method) || 'GET', body: opts && opts.body });
  const json = (obj, status) => ({ ok: !status || status < 400, status: status || 200, json: async () => obj, blob: async () => ({}) });
  if (path.startsWith('/api/state')) return json(state);
  if (path.startsWith('/api/detection')) return { ok: true, status: 200, json: async () => ({}), blob: async () => ({}) };
  if (path.startsWith('/api/action')) {
    const a = JSON.parse(opts.body).action;
    const extra = ['detect', 'plan', 'detect_and_plan'].includes(a) && state.job ? { job: state.job.id } : {};
    return json(Object.assign({ ok: true, message: a === 'rearm' ? 'rearmed (plan cleared)' : (extra.job ? a + ' started' : 'sent'), steps: {} }, extra));
  }
  return { ok: false, status: 503, json: async () => ({ message: 'no signal' }), blob: async () => ({}) };
};
const code = fs.readFileSync(appPath, 'utf8');
const settle = async (n) => { for (let i = 0; i < (n || 20); i++) await new Promise((r) => setImmediate(r)); };
(async () => {
  new Function(code)();
  await settle();
  const out = {
    pill: elements['state-pill'].textContent, pillClass: elements['state-pill'].className,
    chips: elements['chips'].innerHTML, steps: elements['steps'].innerHTML,
    cutDisabled: elements['cut-btn'].disabled, enableText: elements['btn-enable'].textContent, enable: elements['btn-enable']['attr_aria-checked'],
    readback: elements['gates-readback'].textContent, op: elements['op-label'].textContent, takeHidden: elements['take-control'].hidden,
    why: elements['why'].textContent, execLine: elements['exec-line'].textContent, execLineHidden: elements['exec-line'].hidden,
    calls: calls.map((c) => c.method + ' ' + c.path), tabs: {}, result: elements['result-body'].innerHTML,
    perm: { text: elements['perm-line'].textContent, cls: elements['perm-line'].className, hidden: elements['perm-line'].hidden },
    place: { html: elements['place-notes'].innerHTML, hidden: elements['place-notes'].hidden },
    allow: { hidden: elements['allow-row'].hidden, text: elements['btn-allow'].textContent, checked: elements['btn-allow']['attr_aria-checked'],
      disabled: elements['btn-allow'].disabled, readback: elements['allow-readback'].textContent },
  };
  // camera on/off menu
  const camChange = (cam, checked) => { for (const fn of listeners['cam-menu:change'] || []) fn({ target: { dataset: { cam }, checked } }); };
  out.cams = { menu: elements['cam-menu'].innerHTML, small: elements['small-tiles'].innerHTML, large: elements['cap-large'].textContent };
  camChange('scene', false);
  out.cams.afterSceneOff = { large: elements['cap-large'].textContent, small: elements['small-tiles'].innerHTML, saved: lstore['dual_cut_ui.cameras_off'] };
  const names = Object.keys(state.cameras || {});
  for (const n of names) camChange(n, false);
  out.cams.allOff = { large: elements['cap-large'].textContent, nosig: elements['nosig-large'].textContent, nosigHidden: elements['nosig-large'].hidden, small: elements['small-tiles'].innerHTML };
  for (const n of names) camChange(n, true);
  out.cams.allOn = { small: elements['small-tiles'].innerHTML, saved: lstore['dual_cut_ui.cameras_off'] };
  for (const fn of listeners['cam-menu-btn:click'] || []) fn();
  out.cams.menuOpen = elements['cam-menu'].hidden === false && elements['cam-menu-btn']['attr_aria-expanded'] === 'true';
  for (const fn of listeners['document:keydown'] || []) fn({ key: 'Escape' });
  out.cams.menuClosedByEsc = elements['cam-menu'].hidden === true;
  for (const t of tabEls) {
    for (const fn of listeners[t.id + ':click'] || []) fn();
    await settle(5);
    out.tabs[t.dataset.tab] = elements['result-body'].innerHTML;
  }
  for (const fn of listeners['tab-summary:click'] || []) fn();
  await settle(5);
  // an action reply must survive later renders
  if (mode !== 'present') {
    for (const fn of listeners['btn-rearm:click'] || []) fn();
    await settle(30);
    out.lastAfterRearm = elements['last-action'].textContent;
    out.lastClass = elements['last-action'].className;
    for (const fn of listeners['tab-plan:click'] || []) fn();      // forces render()
    for (const fn of listeners['tab-summary:click'] || []) fn();
    await settle(5);
    out.lastAfterRender = elements['last-action'].textContent;
  }
  const posts = (name) => calls.filter((c) => c.method === 'POST' && c.body && c.body.includes('"' + name + '"')).length;
  if (mode !== 'present') {
    // a job this page started: the last-action line shows its result once it has ended
    for (const fn of listeners['btn-dp:click'] || []) fn();
    await settle(30);
    out.lastAfterJob = elements['last-action'].textContent;
    out.lastAfterJobClass = elements['last-action'].className;
    // Software stop must stay clickable with every dialog open
    out.dialogs = {};
    const stopWorks = async () => { const n = posts('software_stop'); for (const fn of listeners['stop:click'] || []) await fn(); await settle(5); return posts('software_stop') === n + 1; };
    const backdropHidden = () => elements['backdrop'].hidden;
    for (const fn of listeners['cut-btn:click'] || []) fn();
    out.dialogs.cut = { open: elements['dlg-cut'].open === true, backdropShown: backdropHidden() === false, stop: await stopWorks(), stillOpen: elements['dlg-cut'].open === true };
    for (const fn of listeners['dlg-cut-no:click'] || []) fn();
    out.dialogs.cut.backdropGone = backdropHidden() === true;
    for (const fn of listeners['tab-detection:click'] || []) fn();
    await settle(20);
    for (const fn of listeners['result-body:click'] || []) fn({ target: { classList: { contains: () => true } } });
    out.dialogs.image = { open: elements['dlg-image'].open === true, backdropShown: backdropHidden() === false, stop: await stopWorks() };
    for (const fn of listeners['document:keydown'] || []) fn({ key: 'Escape' });
    out.dialogs.image.closedByEsc = elements['dlg-image'].open !== true && backdropHidden() === true;
    autoYes = false;
    const rearmBefore = posts('rearm');
    for (const fn of listeners['btn-rearm:click'] || []) fn();
    await settle(5);
    out.dialogs.confirm = { open: elements['dlg-confirm'].open === true, backdropShown: backdropHidden() === false, stop: await stopWorks() };
    for (const fn of listeners['backdrop:click'] || []) fn();
    await settle(5);
    out.dialogs.confirm.cancelledByBackdrop = elements['dlg-confirm'].open !== true && backdropHidden() === true && posts('rearm') === rearmBefore;
    autoYes = true;
  }
  if (mode !== 'present') {
    // the blocked-detection switch: a confirm dialog first (ON), no dialog for OFF
    for (const fn of listeners['btn-allow:click'] || []) fn();
    await settle(10);
    out.allowPosts = calls.filter((c) => c.method === 'POST' && c.body && c.body.includes('set_allow_blocked_detection')).map((c) => c.body);
    out.allowDialogTitle = elements['dlg-confirm-title'].textContent;
    out.allowDialogText = elements['dlg-confirm-text'].textContent;
  }
  for (const fn of listeners['stop:click'] || []) await fn();
  await settle(10);
  out.afterStop = calls.filter((c) => c.method === 'POST').map((c) => c.body);
  out.stopBanner = elements['banner-stop-text'].textContent;
  out.lastAfterStop = elements['last-action'].textContent;
  for (const fn of listeners['cut-btn:click'] || []) fn();
  out.cutDialogOpen = document.getElementById('dlg-cut').opened === true;
  out.cutSummary = document.getElementById('dlg-cut-summary').innerHTML;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error('SMOKE FAILED', e); process.exit(1); });
