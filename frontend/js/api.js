/* ═══════════════════════════════════════════════════════════════
   api.js — THE ONLY FILE THAT CHANGES WHEN THE BACKEND ARRIVES.

   Everything above this layer (app.js, the DOM, the charts) talks to
   service objects and never to fetch(), a socket, or mock data.

   To go live:
       API.configure({ source:'http', host:'http://raspberrypi.local:5000' });

   That swaps MockTransport for HttpTransport. No other file is touched.

   Endpoints the backend must expose (all JSON):
       GET    /api/status              -> StatusPayload
       GET    /api/map                 -> MapPayload
       GET    /api/camera              -> CameraPayload
       GET    /api/missions            -> Mission[]
       POST   /api/missions            -> Mission            (body: {destination,payload,priority})
       DELETE /api/missions/{id}       -> {ok:true}
       POST   /api/missions/{id}/promote -> {ok:true}
       GET    /api/analytics           -> AnalyticsPayload
       POST   /api/robot/{command}     -> {ok:true,message?}  start|pause|resume|dock|reset|estop|manual
       GET    /api/events              -> Event[]            (poll fallback)

   Realtime: EventStream wraps the update channel. Today it polls;
   swap its _connect() for socket.io and nothing above notices.
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  // ── domain constants, mirrored from the robot's config ──────────
  const STATES = ['idle', 'explore', 'delivery_assignment', 'path_planning',
                  'navigation', 'delivery', 'return_to_dock', 'docking', 'charging'];

  const DESTINATIONS = {
    'ward-a':    { label: 'Ward A — Recovery',    pos: [381, 305] },
    'ward-b':    { label: 'Ward B — Paediatrics', pos: [580, 509] },
    'ward-c':    { label: 'Ward C — Isolation',   pos: [111, 139] },
    'theatre':   { label: 'Theatre 3',            pos: [640, 180] },
    'pharmacy':  { label: 'Pharmacy store',       pos: [200, 520] },
    'path-lab':  { label: 'Pathology lab',        pos: [470, 120] }
  };

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* ═══════════ TRANSPORTS ═══════════ */

  class HttpTransport {
    constructor(host) {
      this.host = (host || '').replace(/\/$/, '');
      try {
        const u = new URL(this.host);
        const defaultPort = u.protocol === 'https:' ? 443 : 80;
        const currentPort = Number(u.port || defaultPort);
        u.port = String(currentPort + 1); // telemetry :5000; commands :5001
        this.controlHost = u.origin;
      } catch (_) {
        this.controlHost = this.host.replace(/:5000$/, ':5001');
      }
    }
    async requestAt(base, method, path, body) {
      const res = await fetch(base + path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        cache: 'no-store'
      });
      let payload = null;
      try { payload = res.status === 204 ? null : await res.json(); }
      catch (_) { payload = null; }
      if (!res.ok) {
        const detail = payload && (payload.error || payload.message);
        throw new Error(detail || `${method} ${path} → ${res.status}`);
      }
      return payload;
    }
    request(method, path, body) { return this.requestAt(this.host, method, path, body); }
    command(name, body) {
      return this.requestAt(this.controlHost, 'POST', '/api/robot/' + encodeURIComponent(name), body);
    }
  }

  /**
   * Generates coherent demo telemetry: a simulated vehicle walking a
  /**
   * Generates coherent demo telemetry: a simulated vehicle walking a
   * corridor map, discharging its battery, and working a delivery queue.
   * Shapes match HttpTransport's contract exactly.
   */
  class MockTransport {
    constructor() {
      this.t0 = Date.now();
      this.tick = 0;
      this.gw = 80; this.gh = 60; this.cell = 10;
      this.grid = this._buildMap();
      this.known = new Float32Array(this.gw * this.gh); // -1 free, +1 wall, 0 unknown
      this.pose = { x: 125, y: 465, th: -Math.PI / 2 };
      this.dock = [700, 295];
      this.trail = [];
      this.battery = 96;
      this.state = 'explore';
      this.missions = [];
      this.completed = [];
      this.events = [];
      this.path = [];
      this.drift = 0;
      this.distance = 0;
      this.series = {};
      ['coverage','battery','cpu','ram','fps','speed','drift','path']
        .forEach(k => this.series[k] = []);
      this._seed();
      this._log('ok', 'Vehicle online', 'Docked and charged, ready to work.');
      this._timer = setInterval(() => this._advance(), 200);
    }

    /* -- a corridor layout roughly matching the robot's test map -- */
    _buildMap() {
      const g = new Uint8Array(this.gw * this.gh).fill(1); // 1 = wall
      const carve = (x0, y0, x1, y1) => {
        for (let y = y0; y <= y1; y++)
          for (let x = x0; x <= x1; x++)
            if (x >= 0 && x < this.gw && y >= 0 && y < this.gh) g[y * this.gw + x] = 0;
      };
      carve(8, 43, 72, 48);   // main east-west spine
      carve(10, 8, 15, 48);   // west riser
      carve(36, 12, 41, 48);  // centre riser
      carve(64, 20, 69, 48);  // east riser
      carve(10, 8, 69, 13);   // north corridor
      carve(20, 24, 34, 29);  // ward A spur
      carve(44, 30, 58, 35);  // ward B spur
      carve(56, 14, 62, 22);  // theatre spur
      return g;
    }
    _free(gx, gy) {
      return gx >= 0 && gx < this.gw && gy >= 0 && gy < this.gh &&
             this.grid[gy * this.gw + gx] === 0;
    }
    _seed() {
      [['ward-a', 'IV antibiotics, 2 trays', 1],
       ['pharmacy', 'Controlled drugs cabinet restock', 0]]
        .forEach(([d, p, pr]) => this._addMission(d, p, pr));
    }
    _addMission(destination, payload, priority) {
      const d = DESTINATIONS[destination] || DESTINATIONS['ward-a'];
      const m = {
        id: 'M' + (1000 + Math.floor(Math.random() * 8999)),
        destination, label: d.label, position: d.pos,
        payload: payload || 'Unspecified payload',
        priority: Number(priority) || 0,
        created: new Date().toISOString()
      };
      if (m.priority === 2) this.missions.unshift(m); else this.missions.push(m);
      return m;
    }
    _log(kind, title, detail) {
      this.events.unshift({ kind, title, detail, ts: new Date().toISOString() });
      this.events = this.events.slice(0, 40);
    }

    /* -- one simulated step of the world -- */
    _advance() {
      this.tick++;
      const target = this.missions.length
        ? this.missions[0].position
        : (this.state === 'charging' || this.state === 'idle' ? null : this.dock);

      if (target) {
        const dx = target[0] - this.pose.x, dy = target[1] - this.pose.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 14) {
          if (this.missions.length) {
            const m = this.missions.shift();
            m.completedAt = new Date().toISOString();
            m.durationSec = 40 + Math.floor(Math.random() * 180);
            this.completed.unshift(m);
            this._log('ok', 'Delivery complete', `${m.label} — ${m.payload}`);
            this.state = this.missions.length ? 'path_planning' : 'return_to_dock';
          } else {
            this.state = 'charging';
            this._log('ok', 'Docked', 'Charging contacts engaged.');
          }
        } else {
          const want = Math.atan2(dy, dx);
          let e = Math.atan2(Math.sin(want - this.pose.th), Math.cos(want - this.pose.th));
          this.pose.th += clamp(e, -0.16, 0.16);
          const step = Math.abs(e) > 0.5 ? 0.6 : 3.0;
          const nx = this.pose.x + step * Math.cos(this.pose.th);
          const ny = this.pose.y + step * Math.sin(this.pose.th);
          if (this._free(Math.floor(nx / this.cell), Math.floor(ny / this.cell))) {
            this.distance += Math.hypot(nx - this.pose.x, ny - this.pose.y);
            this.pose.x = nx; this.pose.y = ny;
          } else { this.pose.th += 0.35; }
          this.state = this.missions.length ? 'navigation' : 'return_to_dock';
          this.battery = clamp(this.battery - 0.02, 0, 100);
          this.path = this._route(target);
        }
        this.trail.push([this.pose.x, this.pose.y]);
        if (this.trail.length > 400) this.trail.shift();
      } else if (this.state === 'charging') {
        this.battery = clamp(this.battery + 0.25, 0, 100);
        this.path = [];
        if (this.battery >= 99.5) { this.state = 'idle'; this._log('ok', 'Charged', 'Battery at 100%.'); }
      }

      // reveal occupancy around the vehicle (simulated sensor sweep)
      const gx = Math.floor(this.pose.x / this.cell), gy = Math.floor(this.pose.y / this.cell);
      for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
        const x = gx + dx, y = gy + dy;
        if (x < 0 || y < 0 || x >= this.gw || y >= this.gh) continue;
        if (dx * dx + dy * dy > 18) continue;
        const i = y * this.gw + x;
        this.known[i] = this.grid[i] ? 1 : -1;
      }

      // drift accumulates with distance — mirrors the measured 0.94 px/m
      this.drift = this.distance * 0.025 * 0.94;
      if (this.battery < 25 && this.tick % 90 === 0)
        this._log('warn', 'Battery low', 'Returning to dock to recharge.');

      const s = this._status();
      this._push('coverage', s.coverage_pct); this._push('battery', s.battery_pct);
      this._push('cpu', s.cpu_pct);           this._push('ram', s.ram_mb);
      this._push('fps', s.fps);               this._push('speed', s.speed_ms);
      this._push('drift', s.drift_px);        this._push('path', s.path_len_m);
    }
    _push(k, v) { const a = this.series[k]; a.push(v); if (a.length > 60) a.shift(); }

    _route(target) {
      const pts = [[this.pose.x, this.pose.y]];
      let cx = this.pose.x, cy = this.pose.y;
      for (let i = 0; i < 40; i++) {
        const dx = target[0] - cx, dy = target[1] - cy;
        if (Math.hypot(dx, dy) < 18) break;
        if (Math.abs(dx) > Math.abs(dy)) cx += Math.sign(dx) * 18; else cy += Math.sign(dy) * 18;
        pts.push([cx, cy]);
      }
      pts.push(target);
      return pts;
    }

    _status() {
      let known = 0;
      for (let i = 0; i < this.known.length; i++) if (this.known[i] !== 0) known++;
      const free = this.grid.reduce((n, v) => n + (v ? 0 : 1), 0);
      return {
        state: this.state,
        task: this.missions.length ? `Delivering to ${this.missions[0].label}`
                                   : (this.state === 'charging' ? 'Charging at dock' : 'Standing by'),
        x: +this.pose.x.toFixed(1), y: +this.pose.y.toFixed(1),
        heading_deg: +((this.pose.th * 180 / Math.PI + 360) % 360).toFixed(1),
        speed_ms: +(0.10 + Math.random() * 0.05).toFixed(3),
        battery_pct: +this.battery.toFixed(1),
        coverage_pct: +clamp(known / free * 100, 0, 100).toFixed(1),
        localization_confidence: +clamp(1 - this.drift / 160, 0.05, 1).toFixed(2),
        drift_px: +this.drift.toFixed(1),
        cpu_pct: +(29 + Math.random() * 16).toFixed(1),
        ram_mb: +(148 + Math.random() * 22).toFixed(0),
        fps: +(27 + Math.random() * 4).toFixed(1),
        planner_ms: +(9 + Math.random() * 7).toFixed(1),
        path_len_m: +(this.path.length * 0.45).toFixed(2),
        deliveries_done: this.completed.length,
        deliveries_pending: this.missions.length,
        uptime_s: Math.floor((Date.now() - this.t0) / 1000),
        connected: true
      };
    }

    async request(method, path, body) {
      await new Promise(r => setTimeout(r, 40 + Math.random() * 70)); // latency
      const seg = path.split('?')[0].replace(/\/$/, '');

      if (method === 'GET' && seg === '/api/status') return this._status();
      if (method === 'GET' && seg === '/api/map') {
        return {
          width: this.gw, height: this.gh, cell: this.cell,
          cells: Array.from(this.known),
          pose: { x: this.pose.x, y: this.pose.y, theta: this.pose.th },
          path: this.path, trail: this.trail, dock: this.dock,
          goal: this.missions.length ? this.missions[0].position : this.dock,
          frontiers: this._frontiers()
        };
      }
      if (method === 'GET' && seg === '/api/camera') {
        return {
          width: 640, height: 480, frame: null, fps: +(9 + Math.random() * 3).toFixed(1),
          detections: this._detections()
        };
      }
      if (method === 'GET' && seg === '/api/missions')
        return { active: this.missions[0] || null, queue: this.missions, completed: this.completed };
      if (method === 'POST' && seg === '/api/missions') {
        const m = this._addMission(body.destination, body.payload, body.priority);
        this._log(m.priority === 2 ? 'bad' : 'info',
                  m.priority === 2 ? 'Emergency delivery added' : 'Delivery added',
                  `${m.label} — ${m.payload}`);
        if (this.state === 'idle' || this.state === 'charging') this.state = 'path_planning';
        return m;
      }
      if (method === 'DELETE' && seg.startsWith('/api/missions/')) {
        const id = seg.split('/').pop();
        const i = this.missions.findIndex(m => m.id === id);
        if (i >= 0) { const [m] = this.missions.splice(i, 1); this._log('warn', 'Delivery removed', m.label); }
        return { ok: true };
      }
      if (method === 'POST' && seg.endsWith('/promote')) {
        const id = seg.split('/')[3];
        const i = this.missions.findIndex(m => m.id === id);
        if (i > 0) { const [m] = this.missions.splice(i, 1); m.priority = 2; this.missions.unshift(m);
                     this._log('warn', 'Moved to front', m.label); }
        return { ok: true };
      }
      if (method === 'GET' && seg === '/api/analytics')
        return { series: this.series, completed: this.completed.slice(0, 12) };
      if (method === 'GET' && seg === '/api/events') return this.events;
      if (method === 'POST' && seg.startsWith('/api/robot/')) {
        const cmd = seg.split('/').pop();
        const say = {
          start:  ['ok',   'Mission started',  'Working through the delivery queue.'],
          pause:  ['warn', 'Mission paused',   'Vehicle holding position.'],
          resume: ['ok',   'Mission resumed',  'Continuing to the current destination.'],
          dock:   ['info', 'Returning to dock','Queue suspended.'],
          reset:  ['warn', 'Map cleared',      'Exploration will restart from the dock.'],
          estop:  ['bad',  'Emergency stop',   'Motors cut. Clear the hazard, then resume.'],
          manual: ['info', 'Manual driving',   'Autonomous control suspended.']
        }[cmd] || ['info', cmd, ''];
        this._log(say[0], say[1], say[2]);
        if (cmd === 'estop') this.state = 'idle';
        if (cmd === 'dock') this.missions = [];
        if (cmd === 'reset') this.known.fill(0);
        return { ok: true, message: say[1] };
      }
      throw new Error(`No mock route for ${method} ${path}`);
    }

    _frontiers() {
      const out = [];
      for (let y = 1; y < this.gh - 1; y++) for (let x = 1; x < this.gw - 1; x++) {
        const i = y * this.gw + x;
        if (this.known[i] !== -1) continue;
        if (this.known[i - 1] === 0 || this.known[i + 1] === 0 ||
            this.known[i - this.gw] === 0 || this.known[i + this.gw] === 0) out.push([x, y]);
      }
      return out;
    }
    _detections() {
      const kinds = [['Person', 0.94], ['Hospital bed', 0.88], ['Wheelchair', 0.91],
                     ['Chair', 0.79], ['Trolley', 0.83]];
      const n = Math.floor(Math.random() * 3);
      return Array.from({ length: n }, () => {
        const [label, base] = kinds[Math.floor(Math.random() * kinds.length)];
        return {
          label, confidence: +(base - Math.random() * 0.12).toFixed(2),
          box: [Math.random() * 380, 150 + Math.random() * 130,
                90 + Math.random() * 120, 110 + Math.random() * 150],
          range_m: +(0.6 + Math.random() * 2.6).toFixed(2)
        };
      });
    }
  }

  /* ═══════════ SERVICES ═══════════ */

  class Service {
    constructor(getTransport) { this._t = getTransport; }
    get(p)          { return this._t().request('GET', p); }
    post(p, b)      { return this._t().request('POST', p, b); }
    del(p)          { return this._t().request('DELETE', p); }
  }

  class RobotService extends Service {
    status()            { return this.get('/api/status'); }
    map()               { return this.get('/api/map'); }
    camera()            { return this.get('/api/camera'); }
    command(name, body) {
      const t = this._t();
      return typeof t.command === 'function' ? t.command(name, body) : this.post('/api/robot/' + name, body);
    }
    drive(direction) { return this.command('drive', { direction }); }
    start()             { return this.command('start'); }
    pause()             { return this.command('pause'); }
    resume()            { return this.command('resume'); }
    dock()              { return this.command('dock'); }
    reset()             { return this.command('reset'); }
    emergencyStop()     { return this.command('estop'); }
  }

  class MissionService extends Service {
    list()              { return this.get('/api/missions'); }
    add(destination, payload, priority, position = null) {
      const t = this._t();
      const body = { destination, payload, priority };
      if (destination === 'map-selected') {
        if (!Array.isArray(position) || position.length !== 2)
          throw new Error('Choose a target on the live map first.');
        body.position = position.map(v => Math.round(Number(v)));
      }
      // Hardware accepts only a SLAM-reachable waypoint or a selected live-map point.
      return typeof t.command === 'function'
        ? t.command('mission', body)
        : this.post('/api/missions', { destination, payload, priority });
    }
    remove(id)          { return this.del('/api/missions/' + id); }
    promote(id)         { return this.post(`/api/missions/${id}/promote`); }
  }

  class AnalyticsService extends Service {
    summary()           { return this.get('/api/analytics'); }
    events()            { return this.get('/api/events'); }
  }

  /**
   * Realtime channel. Polls today; to move to Flask-SocketIO replace
   * _connect() with a socket.io client that re-emits the same event
   * names. Subscribers are unaffected.
   */
  class EventStream {
    constructor(getTransport) {
      this._t = getTransport; this._subs = {}; this._timer = null; this._seen = null;
    }
    on(evt, fn) { (this._subs[evt] = this._subs[evt] || []).push(fn); return this; }
    emit(evt, data) { (this._subs[evt] || []).forEach(f => { try { f(data); } catch (e) { console.error(e); } }); }
    start(intervalMs = 700) {
      this.stop();
      this._connect(intervalMs);
      return this;
    }
    _connect(intervalMs) {
      const poll = async () => {
        try {
          const [status, events] = await Promise.all([
            this._t().request('GET', '/api/status'),
            this._t().request('GET', '/api/events')
          ]);
          this.emit('status', status);
          if (events && events.length) {
            const newest = events[0].ts;
            if (this._seen === null) this._seen = newest;
            else if (newest !== this._seen) {
              const fresh = [];
              for (const e of events) { if (e.ts === this._seen) break; fresh.push(e); }
              this._seen = newest;
              fresh.reverse().forEach(e => this.emit('alert', e));
            }
            this.emit('events', events);
          }
        } catch (err) { this.emit('disconnected', err); }
      };
      poll();
      this._timer = setInterval(poll, intervalMs);
    }
    stop() { if (this._timer) clearInterval(this._timer); this._timer = null; }
  }

  /* ═══════════ FACADE ═══════════ */

  let transport = new MockTransport();
  const get = () => transport;

  const API = {
    robot:     new RobotService(get),
    missions:  new MissionService(get),
    analytics: new AnalyticsService(get),
    stream:    new EventStream(get),
    source:    'mock',
    DESTINATIONS, STATES,

    /** The single switch between demo data and a real robot. */
    configure({ source, host } = {}) {
      if (source === 'http') { transport = new HttpTransport(host); this.source = 'http'; }
      else                   { transport = new MockTransport();     this.source = 'mock'; }
      return this;
    }
  };

  global.API = API;
})(window);
