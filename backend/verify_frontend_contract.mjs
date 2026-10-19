/* verify_frontend_contract.mjs
 *
 * Loads the REAL frontend/js/api.js, switches it to the HTTP transport with
 * the one line the README promises, and calls every service method against
 * a running backend. Then dereferences exactly the properties app.js reads.
 *
 * This is the check that matters: it uses the shipped client code rather
 * than a Python restatement of what that code is believed to want.
 *
 *   node backend/verify_frontend_contract.mjs http://127.0.0.1:5000
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const host = process.argv[2] || 'http://127.0.0.1:5000';
const here = dirname(fileURLToPath(import.meta.url));

// api.js is a browser IIFE: give it a `window` and a `fetch`, nothing else.
const sandbox = { window: {}, console, fetch, setTimeout, clearInterval, setInterval, Math, Date };
vm.createContext(sandbox);
vm.runInContext(readFileSync(join(here, '..', 'frontend', 'js', 'api.js'), 'utf8'), sandbox);
const API = sandbox.window.API;

// THE swap the README documents. If anything below needs more than this,
// the contract is not met.
API.configure({ source: 'http', host });

let failures = 0;
const need = (obj, fields, where) => {
  for (const f of fields) {
    if (obj === null || obj === undefined || !(f in obj)) {
      console.log(`  FAIL  ${where}.${f} missing`);
      failures++;
    }
  }
};
const ok = (msg) => console.log(`  ok    ${msg}`);

console.log(`api.js -> ${host}  (API.source = ${API.source})`);

// ── RobotService ──────────────────────────────────────────────
const s = await API.robot.status();
need(s, ['state', 'task', 'connected', 'x', 'y', 'heading_deg', 'speed_ms',
         'battery_pct', 'coverage_pct', 'localization_confidence', 'drift_px',
         'fps', 'cpu_pct', 'ram_mb', 'planner_ms', 'uptime_s',
         'deliveries_done', 'deliveries_pending'], 'status');
ok(`status: ${s.state} ${s.battery_pct}% batt, ${s.coverage_pct}% cov, tick ${s.tick}`);

const m = await API.robot.map();
need(m, ['width', 'height', 'cell', 'cells', 'pose', 'path', 'trail',
         'dock', 'goal', 'frontiers'], 'map');
need(m.pose, ['x', 'y', 'theta'], 'map.pose');
if (m.cells.length !== m.width * m.height) {
  console.log(`  FAIL  cells length ${m.cells.length} != ${m.width}x${m.height}`); failures++;
}
// app.js indexes exactly this way when painting the canvas.
let known = 0;
for (let y = 0; y < m.height; y++)
  for (let x = 0; x < m.width; x++)
    if (m.cells[y * m.width + x] !== 0) known++;
ok(`map: ${m.width}x${m.height} cell ${m.cell}, ${known} known cells, ${m.frontiers.length} frontiers, trail ${m.trail.length}`);

const c = await API.robot.camera();
need(c, ['fps', 'detections'], 'camera');
if (!Array.isArray(c.detections)) { console.log('  FAIL  camera.detections not an array'); failures++; }
ok(`camera: frame=${c.frame}, ${c.detections.length} detections`);

// ── MissionService ────────────────────────────────────────────
const created = await API.missions.add('ward-b', 'Contract check payload', 1);
need(created, ['id', 'label', 'payload', 'priority'], 'created mission');
ok(`missions.add -> ${created.id} ${created.label} p${created.priority}`);

const list = await API.missions.list();
need(list, ['active', 'queue', 'completed'], 'missions');
for (const item of list.queue) need(item, ['id', 'label', 'payload', 'priority'], 'queue item');
ok(`missions.list -> ${list.queue.length} queued, ${list.completed.length} completed`);

await API.missions.promote(created.id);
const promoted = (await API.missions.list()).queue[0];
if (promoted.id !== created.id) { console.log('  FAIL  promote did not move to front'); failures++; }
else ok('missions.promote -> moved to front');

const removed = await API.missions.remove(created.id);
if (removed.ok !== true) { console.log('  FAIL  remove did not return ok'); failures++; }
else ok('missions.remove -> ok');

// ── AnalyticsService ──────────────────────────────────────────
const a = await API.analytics.summary();
need(a, ['series', 'completed'], 'analytics');
need(a.series, ['coverage', 'battery', 'cpu', 'ram', 'fps', 'speed', 'drift', 'path'], 'analytics.series');
ok(`analytics: series keys ok, ${a.series.coverage.length} coverage points`);

const evts = await API.analytics.events();
if (!Array.isArray(evts) || !evts.length) { console.log('  FAIL  events empty'); failures++; }
else { need(evts[0], ['kind', 'title', 'detail', 'ts'], 'event'); ok(`events: ${evts.length}, newest "${evts[0].title}"`); }

// ── commands ──────────────────────────────────────────────────
for (const [name, fn] of [['pause', () => API.robot.pause()],
                          ['resume', () => API.robot.resume()],
                          ['estop', () => API.robot.emergencyStop()],
                          ['resume', () => API.robot.resume()],
                          ['dock', () => API.robot.dock()],
                          ['reset', () => API.robot.reset()],
                          ['start', () => API.robot.start()]]) {
  const r = await fn();
  if (r.ok !== true) { console.log(`  FAIL  ${name} -> ${JSON.stringify(r)}`); failures++; }
  else ok(`robot.${name} -> ok`);
}

// ── EventStream (what the dashboard actually subscribes to) ────
await new Promise((resolve) => {
  let gotStatus = false, gotEvents = false, disconnected = null;
  API.stream
    .on('status', () => { gotStatus = true; })
    .on('events', () => { gotEvents = true; })
    .on('disconnected', (e) => { disconnected = e; })
    .start(300);
  setTimeout(() => {
    API.stream.stop();
    if (disconnected) { console.log(`  FAIL  EventStream disconnected: ${disconnected}`); failures++; }
    else if (!gotStatus || !gotEvents) { console.log('  FAIL  EventStream produced no frames'); failures++; }
    else ok('EventStream: status + events frames received');
    resolve();
  }, 1500);
});

console.log(failures ? `\nFRONTEND CONTRACT: ${failures} FAILURE(S)` : '\nFRONTEND CONTRACT: PASS');
process.exit(failures ? 1 : 0);
