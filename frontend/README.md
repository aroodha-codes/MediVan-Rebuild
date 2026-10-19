# MediVan — Hospital Mission Control (frontend)

A frontend-first dashboard. It runs today on generated demo data and switches
to a live robot by changing **one line in one file**.

```
frontend/
  index.html            shell + all six pages
  css/dashboard.css     design system, responsive down to phone
  js/api.js             ← THE ONLY FILE THE BACKEND WORK TOUCHES
  js/app.js             UI behaviour: router, canvas map, camera, charts
```

## Run it

No build step, no package install. Any static server:

```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

Opening `index.html` directly with `file://` also works.

## Pages

| Page | Contents |
|---|---|
| Overview | State, battery, coverage, deliveries, pose, localisation, compute, activity feed |
| Deliveries | Add / remove / promote, priority queue, active job highlighted |
| Live map | Occupancy grid, vehicle + heading, planned route, goal, dock, trail, frontiers |
| Camera | Forward view with detection overlay and a detection list |
| Analytics | Eight rolling charts + completed-delivery table |
| Settings | Data source switch, robot address, dock threshold, preferences |

Controls (Start, Pause, Resume, Return to dock, Reset map, Export map, Export
logs, Manual driving, Emergency stop) sit in a persistent right-hand dock so
they are reachable from every page.

## Connecting Flask later

`js/api.js` is the entire seam. It holds two transports behind one interface:

* `MockTransport` — simulates a vehicle: walks a corridor map, reveals
  occupancy, discharges battery, works the queue, accumulates drift at the
  measured 0.94 px/m.
* `HttpTransport` — plain `fetch()` against the endpoints below.

Switch at runtime from the Settings page, or in code:

```js
API.configure({ source: 'http', host: 'http://raspberrypi.local:5000' });
```

Nothing in `app.js`, the HTML or the CSS changes. `app.js` contains no
`fetch()`, no socket and no mock data — it only calls service methods.

### Endpoints the backend must implement

| Method | Path | Returns |
|---|---|---|
| GET | `/api/status` | pose, state, battery, coverage, cpu, ram, fps, drift, counts |
| GET | `/api/map` | `{width,height,cell,cells[],pose,path,trail,dock,goal,frontiers}` |
| GET | `/api/camera` | `{width,height,frame,fps,detections[]}` — `frame` a data URL |
| GET | `/api/missions` | `{active,queue[],completed[]}` |
| POST | `/api/missions` | body `{destination,payload,priority}` → the created mission |
| DELETE | `/api/missions/{id}` | `{ok:true}` |
| POST | `/api/missions/{id}/promote` | `{ok:true}` |
| GET | `/api/analytics` | `{series:{...}, completed[]}` |
| POST | `/api/robot/{cmd}` | `start\|pause\|resume\|dock\|reset\|estop\|manual` |
| GET | `/api/events` | newest-first event list |

`cells[]` is row-major `width × height`: `-1` free, `+1` wall, `0` unknown.

### Moving to WebSockets

`EventStream` in `api.js` polls `/api/status` and `/api/events` and re-emits
`status`, `events`, `alert` and `disconnected`. Replace its `_connect()` with a
Socket.IO client that emits the same four names. Subscribers are untouched.

## Design notes

Direction is hospital **vitals-monitor**, not generic admin panel: cool
clinical ground, dark instrument panels for anything sensor-derived, equipment
teal for controls, corridor-sodium amber for caution. The signature element is
the continuous telemetry trace across the header — coverage, battery and speed
drawn as live traces the way a patient monitor draws vitals.

Zero runtime dependencies. Charts and the map are drawn on canvas rather than
pulling Chart.js or Leaflet from a CDN, so the dashboard works on a ward
terminal with no internet. Fonts load from Google Fonts with system fallbacks;
if the network is unavailable the layout is unaffected.

Accessibility floor: visible keyboard focus, `aria-live` alerts, semantic
landmarks, and `prefers-reduced-motion` respected.

## Status

Built and syntax-checked (`node --check` passes on both JS files). **Not yet
opened in a browser** — this environment has no browser or screenshot tool, so
the rendering is unverified. Check it visually before demonstrating it.
