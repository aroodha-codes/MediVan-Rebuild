/* ═══════════════════════════════════════════════════════════════
   app.js — UI behaviour only. Talks exclusively to the API facade
   in api.js; contains no fetch(), no socket, no mock data.
   Charts and the map are drawn on canvas so the dashboard has zero
   external dependencies and works offline on a ward terminal.
   ═══════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(n)) ? '—' : Number(n).toFixed(d);
  const titleCase = s => String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  const state = { page: 'dashboard', status: null, map: null, reduceMotion: false, selectedTarget: null };

  /* ═════════ ROUTER ═════════ */
  function route() {
    const page = (location.hash.replace('#/', '') || 'dashboard');
    state.page = page;
    $$('.page').forEach(p => p.classList.toggle('is-active', p.dataset.page === page));
    $$('.rail__item').forEach(a => a.classList.toggle('is-active', a.dataset.page === page));
    if (page === 'map' || page === 'dashboard') {
      if (!state.map) refreshMap();
      else drawMap();
    }
    if (page === 'camera') refreshCamera();
    if (page === 'analytics') refreshAnalytics();
    if (page === 'missions') refreshMissions();
  }
  window.addEventListener('hashchange', route);

  /* ═════════ TOASTS ═════════ */
  const KIND = { ok: 'toast--ok', warn: 'toast--warn', bad: 'toast--bad', info: '' };
  function toast(kind, title, detail) {
    const el = document.createElement('div');
    el.className = 'toast ' + (KIND[kind] || '');
    el.innerHTML = `<b></b><span></span>`;
    el.querySelector('b').textContent = title;
    el.querySelector('span').textContent = detail || '';
    $('#toasts').appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 320); }, 4600);
  }

  /* ═════════ VITALS TRACE (signature element) ═════════ */
  const trace = { buf: [], canvas: null, ctx: null };
  function initTrace() {
    trace.canvas = $('#vitalsTrace');
    const resize = () => {
      const r = trace.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      trace.canvas.width = Math.max(1, r.width * dpr);
      trace.canvas.height = Math.max(1, r.height * dpr);
      trace.ctx = trace.canvas.getContext('2d');
      trace.ctx.scale(dpr, dpr);
      trace.w = r.width; trace.h = r.height;
    };
    resize();
    window.addEventListener('resize', resize);
    drawTrace();
  }
  function drawTrace() {
    const c = trace.ctx;
    if (c) {
      const { w, h } = trace;
      c.clearRect(0, 0, w, h);
      c.strokeStyle = 'rgba(126,147,168,.16)';
      c.lineWidth = 1;
      c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2); c.stroke();

      const pts = trace.buf;
      if (pts.length > 1) {
        const step = w / 90;
        const draw = (key, colour, scale) => {
          c.beginPath(); c.strokeStyle = colour; c.lineWidth = 1.6;
          pts.forEach((p, i) => {
            const x = w - (pts.length - 1 - i) * step;
            const y = h - (p[key] / scale) * (h - 8) - 4;
            i ? c.lineTo(x, y) : c.moveTo(x, y);
          });
          c.stroke();
        };
        draw('cov', '#0E7C86', 100);
        draw('bat', '#E8A33D', 100);
        draw('spd', '#3FA34D', 0.4);
      }
    }
    requestAnimationFrame(() => setTimeout(drawTrace, 90));
  }

  /* ═════════ STATUS BINDING ═════════ */
  function applyStatus(s) {
    state.status = s;
    trace.buf.push({ cov: s.coverage_pct, bat: s.battery_pct, spd: s.speed_ms });
    if (trace.buf.length > 90) trace.buf.shift();

    $('#vBattery').textContent  = fmt(s.battery_pct, 0) + '%';
    // Battery sensing (ADS1115 + INA219). fmt() renders an em dash for
    // null, so an unavailable sensor shows '—' rather than a misleading 0.
    $('#vVoltage').textContent  = s.battery_voltage == null ? '—' : fmt(s.battery_voltage, 2) + ' V';
    $('#vCurrent').textContent  = s.battery_current == null ? '—' : fmt(s.battery_current, 2) + ' A';
    $('#vPower').textContent    = s.battery_power   == null ? '—' : fmt(s.battery_power, 2) + ' W';
    $('#vBattState').textContent = titleCase(s.battery_state || 'unknown');
    $('#vCoverage').textContent = fmt(s.coverage_pct, 0) + '%';
    $('#vSpeed').textContent    = fmt(s.speed_ms, 2);
    $('#vState').textContent    = titleCase(s.state);
    $('#linkState').classList.toggle('is-down', !s.connected);
    $('#linkState').childNodes[0].nodeValue = API.source === 'mock' ? 'DEMO DATA' : 'LIVE';

    $('#sMode').textContent      = titleCase(s.state);
    $('#sTask').textContent      = s.task || '—';
    $('#sBattery').textContent   = fmt(s.battery_pct, 0) + '%';
    $('#sBatteryBar').style.width  = clampPct(s.battery_pct);
    $('#sCoverage').textContent  = fmt(s.coverage_pct, 0) + '%';
    $('#sCoverageBar').style.width = clampPct(s.coverage_pct);
    $('#sDeliveries').textContent = s.deliveries_done;
    $('#sPending').textContent   = s.deliveries_pending
      ? `${s.deliveries_pending} waiting` : 'Queue empty';

    $('#sPos').textContent     = `${fmt(s.x, 1)}, ${fmt(s.y, 1)} px`;
    $('#sHeading').textContent = fmt(s.heading_deg, 1) + '°';
    $('#sVel').textContent     = fmt(s.speed_ms, 2) + ' m/s';
    $('#sConf').textContent    = fmt(s.localization_confidence * 100, 0) + '%';
    $('#sDrift').textContent   = fmt(s.drift_px, 1) + ' px';

    $('#sFps').textContent    = fmt(s.fps, 1) + ' Hz';
    $('#sCpu').textContent    = fmt(s.cpu_pct, 0) + '%';
    $('#sRam').textContent    = fmt(s.ram_mb, 0) + ' MB';
    $('#sPlan').textContent   = fmt(s.planner_ms, 1) + ' ms';
    $('#sUptime').textContent = formatUptime(s.uptime_s);

    // Live robot pose & motion synchronisation for canvas rendering
    if (s.x != null && s.y != null) {
      const thetaRad = ((Number(s.heading_deg) || 0) * Math.PI) / 180;
      if (!state.map) {
        state.map = {
          width: 80, height: 60, cell: 10,
          cells: new Array(4800).fill(0),
          pose: { x: Number(s.x), y: Number(s.y), theta: thetaRad },
          trail: [[Number(s.x), Number(s.y)]],
        };
      } else {
        state.map.pose = { x: Number(s.x), y: Number(s.y), theta: thetaRad };
        if (!state.map.trail) state.map.trail = [];
        const last = state.map.trail[state.map.trail.length - 1];
        if (!last || Math.hypot(last[0] - Number(s.x), last[1] - Number(s.y)) >= 0.8) {
          state.map.trail.push([Number(s.x), Number(s.y)]);
          if (state.map.trail.length > 800) state.map.trail.shift();
        }
      }
      drawMap();
    }

    const motionBadge = $('#overviewMotionBadge');
    if (motionBadge) {
      const speed = Number(s.speed_ms || 0);
      const ctrl = s.control || {};
      const drive = String(ctrl.effective_drive || ctrl.drive || '').toLowerCase();
      const stateLabel = String(s.state || '').toLowerCase();
      if (stateLabel === 'delivered') {
        motionBadge.textContent = 'DELIVERY COMPLETE (Target Reached)';
        motionBadge.style.background = '#0E7C86';
      } else if (stateLabel === 'delivering') {
        motionBadge.textContent = `Delivering (${speed.toFixed(2)} m/s)`;
        motionBadge.style.background = '#2e7d32';
      } else if (stateLabel === 'returning') {
        motionBadge.textContent = `Returning to Base (${speed.toFixed(2)} m/s)`;
        motionBadge.style.background = '#E8A33D';
      } else if (drive && drive !== 'stop' && !drive.includes('stop')) {
        motionBadge.textContent = `Moving: ${drive.toUpperCase()} (${speed.toFixed(2)} m/s)`;
        motionBadge.style.background = '#2e7d32';
      } else if (speed > 0.02) {
        motionBadge.textContent = `Moving (${speed.toFixed(2)} m/s)`;
        motionBadge.style.background = '#2e7d32';
      } else {
        motionBadge.textContent = 'Robot Stationary';
        motionBadge.style.background = '#4A5866';
      }
    }

    if (String(s.state || '').toLowerCase() === 'delivered' && state.lastNotifiedDelivered !== s.task) {
      state.lastNotifiedDelivered = s.task;
      toast('ok', 'Delivery Complete!', s.task || 'Arrived at delivery destination.');
      refreshMissions();
    }

    const control = s.control || {};
    const manualStatus = $('#manualDriveStatus');
    if (manualStatus) {
      const runtimeMode = control.runtime_mode || 'starting';
      const runtimeArmed = Boolean(control.runtime_armed);
      const estopLatched = Boolean(control.estop_latched);
      const hardwareStopped = Boolean(control.runtime_estop);
      const drive = control.effective_drive || 'stop';
      const pwmA = Number(control.effective_pwm_a || 0);
      const pwmB = Number(control.effective_pwm_b || 0);
      if (estopLatched) manualStatus.textContent = 'Controller status: emergency stop latched; acknowledge/re-arm before driving.';
      else if (!runtimeArmed || hardwareStopped) manualStatus.textContent = 'Controller status: disarmed/stopped; no motion commands are being applied.';
      else if (runtimeMode === 'manual') manualStatus.textContent = `Manual active · command ${drive} · PWM ${pwmA}/${pwmB}`;
      else manualStatus.textContent = `Controller status: ${titleCase(runtimeMode)} mode · phase ${control.mission_phase || s.state}`;
    }

    const n = s.deliveries_pending || 0;
    const badge = $('#navQueueCount');
    badge.textContent = n; badge.dataset.zero = n ? '0' : '1';
  }
  const clampPct = v => Math.max(0, Math.min(100, Number(v) || 0)) + '%';
  function formatUptime(sec) {
    if (!sec && sec !== 0) return '—';
    const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60);
    return h ? `${h}h ${m}m` : `${m}m ${sec % 60}s`;
  }

  /* ═════════ ACTIVITY FEED ═════════ */
  function renderFeed(events) {
    const ul = $('#feed');
    if (!events || !events.length) { ul.innerHTML = '<li class="feed__empty">No events yet.</li>'; return; }
    ul.innerHTML = events.slice(0, 14).map(e => `
      <li><time>${new Date(e.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
          <span class="dot dot--${e.kind === 'info' ? '' : e.kind}"></span>
          <span><strong>${escape(e.title)}</strong>${e.detail ? ' — ' + escape(e.detail) : ''}</span></li>`).join('');
  }
  const escape = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ═════════ MISSIONS ═════════ */
  async function refreshMissions() {
    const data = await API.missions.list();
    const ul = $('#queue');
    const q = data.queue || [];
    $('#queueCount').textContent = q.length ? `${q.length} waiting` : 'empty';
    if (!q.length) { ul.innerHTML = '<li class="queue__empty">Nothing queued. Add a delivery to start.</li>'; return; }
    ul.innerHTML = q.map((m, i) => `
      <li class="${i === 0 ? 'is-active' : ''}">
        <span class="queue__idx">${String(i + 1).padStart(2, '0')}</span>
        <span><span class="queue__dest">${escape(m.label)}</span><br>
              <span class="queue__item">${escape(m.payload)}</span></span>
        <span class="queue__pri pri-${m.priority}">${['Routine', 'Urgent', 'Emergency'][m.priority]}</span>
        <span>
          ${i > 0 ? `<button class="queue__del" data-promote="${m.id}" title="Move to front">↑</button>` : ''}
          <button class="queue__del" data-remove="${m.id}" title="Remove">×</button>
        </span>
      </li>`).join('');
  }

  /* ═════════ MAP ═════════ */
  async function refreshMap() {
    state.map = await API.robot.map();
    drawMap();
  }
  function drawMap() {
    const m = state.map;
    if (!m) return;
    const canvases = [$('#mapCanvas'), $('#overviewMapCanvas')].filter(Boolean);
    if (!canvases.length) return;

    canvases.forEach(cv => {
      // If the canvas element is not visible on page, skip drawing to save CPU
      if (cv.offsetParent === null && canvases.length > 1) return;
      const ctx = cv.getContext('2d');
      const cell = m.cell || 10;
      if (cv.width !== m.width * cell || cv.height !== m.height * cell) {
        cv.width = m.width * cell;
        cv.height = m.height * cell;
      }

      ctx.fillStyle = '#1B2733';
      ctx.fillRect(0, 0, cv.width, cv.height);
      if (m.cells) {
        for (let y = 0; y < m.height; y++) {
          for (let x = 0; x < m.width; x++) {
            const v = m.cells[y * m.width + x];
            if (v === 0) continue;
            ctx.fillStyle = v < 0 ? '#63798F' : '#D8E2EC';
            ctx.fillRect(x * cell, y * cell, cell, cell);
          }
        }
      }

      const tgFrontier = $('#tgFrontier');
      if (tgFrontier && tgFrontier.checked && m.frontiers) {
        ctx.fillStyle = 'rgba(232,163,61,.55)';
        m.frontiers.forEach(([x, y]) => ctx.fillRect(x * cell, y * cell, cell, cell));
      }

      const tgTrail = $('#tgTrail');
      const showTrail = tgTrail ? tgTrail.checked : true;
      if (showTrail && m.trail && m.trail.length > 1) {
        ctx.strokeStyle = 'rgba(63,163,77,.65)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        m.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
      }

      const tgPath = $('#tgPath');
      const showPath = tgPath ? tgPath.checked : true;
      if (showPath && m.path && m.path.length > 1) {
        ctx.strokeStyle = '#E8A33D';
        ctx.lineWidth = 3;
        ctx.setLineDash([9, 6]);
        ctx.beginPath();
        m.path.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (m.dock) {
        ctx.fillStyle = '#0E7C86';
        ctx.fillRect(m.dock[0] - 9, m.dock[1] - 9, 18, 18);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('D', m.dock[0], m.dock[1] + 4);
      }

      if (m.goal) {
        ctx.strokeStyle = '#E8A33D';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(m.goal[0], m.goal[1], 11, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (state.selectedTarget) {
        ctx.strokeStyle = '#F0C24B';
        ctx.fillStyle = '#F0C24B';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(state.selectedTarget[0], state.selectedTarget[1], 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(state.selectedTarget[0] - 14, state.selectedTarget[1]);
        ctx.lineTo(state.selectedTarget[0] + 14, state.selectedTarget[1]);
        ctx.moveTo(state.selectedTarget[0], state.selectedTarget[1] - 14);
        ctx.lineTo(state.selectedTarget[0], state.selectedTarget[1] + 14);
        ctx.stroke();
      }

      const p = m.pose;
      if (p) {
        // High visibility outer aura ring
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(63,163,77,.6)';
        ctx.lineWidth = 2;
        ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
        ctx.stroke();

        // Directional robot vehicle body
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.theta);
        ctx.fillStyle = '#3FA34D';
        ctx.beginPath();
        ctx.moveTo(15, 0);
        ctx.lineTo(-9, 9);
        ctx.lineTo(-5, 0);
        ctx.lineTo(-9, -9);
        ctx.closePath();
        ctx.fill();

        // Robot center beacon dot
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(0, 0, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    });

    const targetStatus = $('#mapTargetStatus');
    if (targetStatus) {
      targetStatus.textContent = state.selectedTarget
        ? `Selected target x=${Math.round(state.selectedTarget[0])} y=${Math.round(state.selectedTarget[1])}`
        : (API.source === 'http' ? 'Click explored map to set delivery target' : 'No delivery target selected');
    }
    const dispatchBtn = $('#btnDispatchTarget');
    if (dispatchBtn) {
      dispatchBtn.style.display = (state.selectedTarget && API.source === 'http') ? 'inline-block' : 'none';
    }

    if (m.pose) {
      const poseStr = `x ${m.pose.x.toFixed(1)}  y ${m.pose.y.toFixed(1)}  θ ${(m.pose.theta * 180 / Math.PI).toFixed(0)}°`;
      const mapPoseEl = $('#mapPose');
      if (mapPoseEl) mapPoseEl.textContent = poseStr;
      const overviewPoseEl = $('#overviewPose');
      if (overviewPoseEl) overviewPoseEl.textContent = poseStr;
    }
  }

  /* ═════════ CAMERA ═════════ */
  async function refreshCamera() {
    const c = await API.robot.camera();
    const cv = $('#camCanvas'); const ctx = cv.getContext('2d');
    // A real backend supplies c.frame as a data URL; the demo renders a
    // synthetic corridor so the overlay pipeline is visible without hardware.
    if (c.frame) {
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, cv.width, cv.height); overlay(ctx, c); };
      img.src = c.frame;
    } else {
      const g = ctx.createLinearGradient(0, 0, 0, cv.height);
      g.addColorStop(0, '#2E3F52'); g.addColorStop(.55, '#3C5064'); g.addColorStop(1, '#556C82');
      ctx.fillStyle = g; ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.strokeStyle = 'rgba(216,226,236,.18)'; ctx.lineWidth = 1;
      for (let i = 1; i < 9; i++) {
        ctx.beginPath(); ctx.moveTo(0, 250 + i * 26); ctx.lineTo(cv.width, 250 + i * 26); ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(230, 250); ctx.lineTo(60, 480); ctx.moveTo(410, 250); ctx.lineTo(580, 480);
      ctx.stroke();
      overlay(ctx, c);
    }
    $('#camFps').textContent = fmt(c.fps, 1) + ' fps';
    $('#camDet').textContent = `${c.detections.length} detected`;
    $('#detList').innerHTML = c.detections.length
      ? c.detections.map(d => `<li><span>${escape(d.label)}</span>
          <span class="mono">${(d.confidence * 100).toFixed(0)}% · ${d.range_m} m</span></li>`).join('')
      : '<li class="feed__empty">Nothing in view.</li>';
  }
  function overlay(ctx, c) {
    ctx.lineWidth = 2; ctx.font = 'bold 12px monospace';
    c.detections.forEach(d => {
      const [x, y, w, h] = d.box;
      ctx.strokeStyle = '#E8A33D'; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(232,163,61,.92)'; ctx.fillRect(x, y - 17, ctx.measureText(d.label).width + 14, 17);
      ctx.fillStyle = '#1B2733'; ctx.fillText(d.label, x + 6, y - 5);
    });
  }

  /* ═════════ CHARTS ═════════ */
  const CHART_COLOUR = {
    coverage: '#0E7C86', battery: '#E8A33D', cpu: '#C4453B', ram: '#4A5866',
    fps: '#3FA34D', speed: '#0E7C86', drift: '#E8A33D', path: '#4A5866'
  };
  function sparkline(canvas, data, colour) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 260, h = 150;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!data || data.length < 2) {
      ctx.fillStyle = '#4A5866'; ctx.font = '12px Inter,sans-serif';
      ctx.fillText('Collecting…', 8, h / 2); return;
    }
    const min = Math.min(...data), max = Math.max(...data);
    const pad = (max - min) * 0.15 || 1;
    const lo = min - pad, hi = max + pad;
    const X = i => (i / (data.length - 1)) * (w - 6) + 3;
    const Y = v => h - 18 - ((v - lo) / (hi - lo)) * (h - 34);

    ctx.strokeStyle = 'rgba(211,217,224,.7)'; ctx.lineWidth = 1;
    [0, .5, 1].forEach(f => {
      const y = 16 + f * (h - 34);
      ctx.beginPath(); ctx.moveTo(3, y); ctx.lineTo(w - 3, y); ctx.stroke();
    });

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, colour + '33'); grad.addColorStop(1, colour + '00');
    ctx.beginPath(); ctx.moveTo(X(0), Y(data[0]));
    data.forEach((v, i) => ctx.lineTo(X(i), Y(v)));
    ctx.lineTo(X(data.length - 1), h - 18); ctx.lineTo(X(0), h - 18); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath(); ctx.strokeStyle = colour; ctx.lineWidth = 2;
    data.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
    ctx.stroke();

    const last = data[data.length - 1];
    ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(X(data.length - 1), Y(last), 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 13px "JetBrains Mono",monospace'; ctx.fillStyle = '#0F1620';
    ctx.fillText(last.toFixed(1), 4, 13);
  }
  async function refreshAnalytics() {
    const a = await API.analytics.summary();
    $$('[data-chart]').forEach(cv => {
      const k = cv.dataset.chart;
      sparkline(cv, a.series[k], CHART_COLOUR[k] || '#0E7C86');
    });
    const tb = $('#deliveryTable tbody');
    tb.innerHTML = a.completed.length
      ? a.completed.map(m => `<tr><td>${escape(m.label)}</td><td>${escape(m.payload)}</td>
          <td>${new Date(m.completedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
          <td class="num mono">${Math.floor(m.durationSec / 60)}m ${m.durationSec % 60}s</td></tr>`).join('')
      : '<tr><td colspan="4" class="queue__empty">No deliveries completed yet.</td></tr>';
  }

  function updateDestinationOptions() {
    const select = $('#mDest');
    if (!select) return;
    const previous = select.value;
    const options = API.source === 'http'
      ? [
          ['auto-waypoint', 'Automatic reachable SLAM waypoint'],
          ['map-selected', 'Select target on live map']
        ]
      : Object.entries(API.DESTINATIONS).map(([key, dest]) => [key, dest.label]);
    select.replaceChildren(...options.map(([value, label]) => {
      const option = document.createElement('option');
      option.value = value; option.textContent = label; return option;
    }));
    select.value = options.some(([value]) => value === previous) ? previous : options[0][0];
    if (API.source !== 'http') state.selectedTarget = null;
    drawMap();
  }


  /* ═════════ EVENT WIRING ═════════ */
  function wire() {
    $('#missionForm').addEventListener('submit', async e => {
      e.preventDefault();
      const dest = $('#mDest').value;
      const item = $('#mItem').value.trim() || 'Unspecified payload';
      const pri  = $('#mPriority').value;
      try {
        if (API.source === 'http' && dest === 'map-selected' && !state.selectedTarget) {
          toast('warn', 'Select a target first', 'Open Live map, choose “Select on map”, then click a target position. Unknown or occupied cells will be rejected by the robot.');
          return;
        }
        const result = await API.missions.add(dest, item, pri, state.selectedTarget);
        $('#mItem').value = '';
        if (result && result.message) toast('info', 'Physical delivery request', result.message);
        else toast('ok', 'Delivery queued', '');
        refreshMissions();
      } catch (err) { toast('bad', 'Delivery request failed', err.message || String(err)); }
    });

    $('#queue').addEventListener('click', async e => {
      const rm = e.target.closest('[data-remove]');
      const pr = e.target.closest('[data-promote]');
      if (rm) { await API.missions.remove(rm.dataset.remove); refreshMissions(); }
      if (pr) { await API.missions.promote(pr.dataset.promote); refreshMissions(); }
    });

    $$('[data-cmd]').forEach(btn => btn.addEventListener('click', async () => {
      const cmd = btn.dataset.cmd;
      if (cmd === 'export-map' || cmd === 'export-logs') {
        toast('warn', 'Export unavailable', 'This build does not yet provide a file-download endpoint for this action.');
        return;
      }
      // Ask for confirmation before mutating the UI or sending any E-stop action.
      if (cmd === 'estop' && !confirm('Request an emergency stop now?')) return;
      // A mode-changing command cancels browser teleoperation first.
      if (['start', 'pause', 'resume', 'dock', 'reset', 'estop', 'rearm', 'finish-explore', 'save-map'].includes(cmd)) {
        stopDashboardDrive();
        $('#manualMode').checked = false;
      }
      try {
        const r = await API.robot.command(cmd);
        if (r && r.message) toast(cmd === 'estop' ? 'bad' : 'ok', r.message, '');
        refreshMissions();
      } catch (err) { toast('bad', 'Command failed', err.message || String(err)); }
    }));

    $('#manualMode').addEventListener('change', async e => {
      try {
        const r = await API.robot.command(e.target.checked ? 'manual' : 'auto');
        toast('info', e.target.checked ? 'Manual mode' : 'Manual mode disabled', r && r.message ? r.message : '');
        if (!e.target.checked) stopDashboardDrive();
      } catch (err) {
        e.target.checked = false;
        toast('bad', 'Mode change failed', err.message || String(err));
      }
    });

    // Press-and-hold joystick: refresh every 200 ms; server stops after 1.5 s
    // without a refresh, so a lost browser/network release does not keep driving.
    let dashboardDriveTimer = null;
    let dashboardDriveDirection = null;
    let dashboardDriveInFlight = false;
    let dashboardDriveFailures = 0;
    async function sendDashboardDrive(direction) {
      if (!dashboardDriveDirection && direction !== 'stop') return;
      if (dashboardDriveInFlight) return; // avoid piling up redundant HTTP requests
      dashboardDriveInFlight = true;
      try {
        await API.robot.drive(direction);
        dashboardDriveFailures = 0;
      } catch (err) {
        dashboardDriveFailures++;
        if (dashboardDriveFailures >= 8) {
          dashboardDriveDirection = null;
          if (dashboardDriveTimer) { clearInterval(dashboardDriveTimer); dashboardDriveTimer = null; }
          toast('bad', 'Manual drive stopped', err.message || String(err));
        }
      } finally {
        dashboardDriveInFlight = false;
      }
    }
    function stopDashboardDrive() {
      const wasDriving = !!dashboardDriveDirection;
      dashboardDriveDirection = null;
      dashboardDriveInFlight = false;
      dashboardDriveFailures = 0;
      if (dashboardDriveTimer) { clearInterval(dashboardDriveTimer); dashboardDriveTimer = null; }
      if (wasDriving && API.source === 'http') sendDashboardDrive('stop');
    }
    async function ensureManualMode() {
      if (!$('#manualMode').checked) {
        $('#manualMode').checked = true;
        try {
          await API.robot.command('manual');
          toast('info', 'Manual driving enabled', 'Drive with held buttons or W/A/S/D / Arrow keys.');
        } catch (err) {
          $('#manualMode').checked = false;
          throw err;
        }
      }
    }

    $$('[data-drive]').forEach(btn => {
      btn.addEventListener('contextmenu', e => e.preventDefault());
      btn.addEventListener('click', async e => {
        if (API.source !== 'http') {
          e.preventDefault();
          toast('warn', 'Manual drive unavailable', 'Dashboard is not connected to a live robot.');
          return;
        }
        try {
          await ensureManualMode();
        } catch (err) {
          toast('bad', 'Manual mode failed', err.message || String(err));
          return;
        }
        const dir = btn.dataset.drive;
        if (dir === 'stop') {
          stopDashboardDrive();
          return;
        }
        if (dashboardDriveTimer) clearInterval(dashboardDriveTimer);
        dashboardDriveDirection = dir;
        sendDashboardDrive(dashboardDriveDirection);
        dashboardDriveTimer = setInterval(() => {
          if (dashboardDriveDirection) sendDashboardDrive(dashboardDriveDirection);
        }, 250);
      });
    });

    const KEY_DIR_MAP = {
      ArrowUp: 'forward', KeyW: 'forward',
      ArrowDown: 'reverse', KeyS: 'reverse',
      ArrowLeft: 'left', KeyA: 'left',
      ArrowRight: 'right', KeyD: 'right',
      Space: 'stop',
    };
    window.addEventListener('keydown', async e => {
      if (API.source !== 'http') return;
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
      const dir = KEY_DIR_MAP[e.code];
      if (!dir) return;
      e.preventDefault();
      try {
        await ensureManualMode();
      } catch (_) {
        return;
      }
      if (dir === 'stop') {
        stopDashboardDrive();
        return;
      }
      if (dashboardDriveDirection === dir) return;
      if (dashboardDriveTimer) clearInterval(dashboardDriveTimer);
      dashboardDriveDirection = dir;
      sendDashboardDrive(dashboardDriveDirection);
      dashboardDriveTimer = setInterval(() => {
        if (dashboardDriveDirection) sendDashboardDrive(dashboardDriveDirection);
      }, 250);
    });

    $$('#tgPath,#tgFrontier,#tgTrail').forEach(t => t.addEventListener('change', drawMap));

    $('#mDest').addEventListener('change', () => {
      if ($('#mDest').value === 'map-selected' && API.source === 'http') {
        toast('info', 'Choose a map target', 'Open Live map and click a known-free location after the robot has mapped it.');
      }
      drawMap();
    });
    $('#mapCanvas').addEventListener('click', e => {
      if (API.source !== 'http') return;
      const destEl = $('#mDest');
      if (destEl) destEl.value = 'map-selected';
      const cv = e.currentTarget;
      const rect = cv.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = Math.max(0, Math.min(cv.width - 1, Math.round((e.clientX - rect.left) / rect.width * cv.width)));
      const y = Math.max(0, Math.min(cv.height - 1, Math.round((e.clientY - rect.top) / rect.height * cv.height)));
      state.selectedTarget = [x, y];
      drawMap();
      toast('info', 'Delivery target selected', `Target marked at (${x}, ${y}). Click 'Dispatch Delivery' to send MediVan.`);
    });
    const dispatchBtn = $('#btnDispatchTarget');
    if (dispatchBtn) {
      dispatchBtn.addEventListener('click', async () => {
        if (!state.selectedTarget) return;
        try {
          await API.missions.add('map-selected', 'Custom Map Delivery', 1, state.selectedTarget);
          toast('ok', 'Delivery dispatched', `Vehicle navigating to target (${state.selectedTarget[0]}, ${state.selectedTarget[1]}).`);
        } catch (err) {
          toast('error', 'Delivery rejected', err.message || String(err));
        }
      });
    }
    $('#setApply').addEventListener('click', () => {
      API.configure({ source: $('#setSource').value, host: $('#setHost').value });
      updateDestinationOptions();
      toast('ok', 'Data source changed',
        API.source === 'mock' ? 'Showing demo data.' : 'Connected to the robot address; physical destinations use live SLAM.');
      API.stream.start();
    });
    $('#setDock').addEventListener('input', e => $('#setDockOut').textContent = e.target.value + '%');
    $('#setMotion').addEventListener('change', e => {
      state.reduceMotion = e.target.checked;
      document.documentElement.style.setProperty('scroll-behavior', e.target.checked ? 'auto' : 'smooth');
    });
  }

  /* ═════════ BOOT ═════════ */
  function boot() {
    const hostField = $('#setHost');
    if (location.protocol.startsWith('http')) {
      const liveHost = `${location.protocol}//${location.hostname}:${location.port || '5000'}`;
      if (hostField) hostField.value = liveHost;
      const sourceField = $('#setSource');
      if (sourceField) sourceField.value = 'http';
      API.configure({ source: 'http', host: liveHost });
    }
    updateDestinationOptions();
    initTrace();
    wire();
    route();

    API.stream
      .on('status', s => {
        applyStatus(s);
        if (state.page === 'map' && !state.map) refreshMap();
      })
      .on('events', renderFeed)
      .on('alert', e => toast(e.kind, e.title, e.detail))
      .on('disconnected', () => {
        $('#linkState').classList.add('is-down');
        $('#linkState').childNodes[0].nodeValue = 'NO LINK';
      })
      .start(400);

    refreshMissions();
    setInterval(() => {
      if (state.page === 'dashboard' || state.page === 'map') refreshMap();
      if (state.page === 'camera') refreshCamera();
      if (state.page === 'analytics') refreshAnalytics();
      if (state.page === 'missions') refreshMissions();
    }, 1400);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();