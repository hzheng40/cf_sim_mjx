from __future__ import annotations

import json
import math
import numbers
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .mjcf import assets_dir

VENDOR_MODULES = (
    ("vendor/three.module.js", "https://unpkg.com/three@0.164.1/build/three.module.js"),
    ("vendor/OrbitControls.js", "https://unpkg.com/three@0.164.1/examples/jsm/controls/OrbitControls.js"),
    ("vendor/OBJLoader.js", "https://unpkg.com/three@0.164.1/examples/jsm/loaders/OBJLoader.js"),
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def _copy_assets(web_dir: Path) -> None:
    dst = web_dir / "assets" / "cf2"
    dst.mkdir(parents=True, exist_ok=True)
    src_dir = assets_dir()
    for src in sorted(src_dir.glob("cf2_*.obj")):
        shutil.copy2(src, dst / src.name)
    for name in ("cf2.xml", "cf2_asset.xml"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)


def _ensure_vendor_modules(web_dir: Path) -> None:
    for rel_path, url in VENDOR_MODULES:
        dst = web_dir / rel_path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                dst.write_bytes(response.read())
        except Exception:
            if dst.exists():
                dst.unlink()


def generate_web_report(
    out_dir: str | Path,
    env: Any | None = None,
    rollout_payloads: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    if rollout_payloads is None:
        rollout_payloads = kwargs.pop("rollout_payloads", None)
    if summary is None:
        summary = kwargs.pop("summary", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if rollout_payloads is None:
        raise TypeError("generate_web_report requires rollout_payloads")
    if summary is None:
        summary = env.metadata() if env is not None and hasattr(env, "metadata") else {}
    web_dir = Path(out_dir) / "web_report"
    web_dir.mkdir(parents=True, exist_ok=True)
    _copy_assets(web_dir)
    _ensure_vendor_modules(web_dir)
    (web_dir / "report_data.js").write_text(
        "window.CRAZYFLIE_MJX_REPORT = "
        + json.dumps(_json_safe({"summary": summary, "rollouts": rollout_payloads}), allow_nan=False)
        + ";\n"
    )
    index_path = web_dir / "index.html"
    index_path.write_text(_html())
    return index_path


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crazyflie MJX Simulation</title>
  <style>
    :root { color-scheme: light; --text:#18202a; --muted:#607080; --line:#d8dee6; --panel:#fff; --band:#f4f7fa; --accent:#266c62; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:var(--band); letter-spacing:0; }
    header { padding:15px 20px 10px; background:var(--panel); border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:21px; font-weight:650; }
    .meta { color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    main { display:grid; grid-template-columns:minmax(360px,1fr) 320px; gap:14px; padding:14px 20px 22px; }
    #viewer { height:min(70vh,720px); min-height:420px; background:#eef2f5; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    #viewer canvas { display:block; width:100%; height:100%; touch-action:none; }
    aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; display:grid; gap:10px; align-self:start; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; }
    select,input[type=range] { width:100%; }
    button { border:1px solid var(--line); border-radius:6px; background:#fff; padding:7px 10px; cursor:pointer; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    td,th { border-bottom:1px solid var(--line); padding:5px 3px; text-align:left; }
    th { color:var(--muted); font-weight:600; }
    .row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    @media (max-width:900px) { main { grid-template-columns:1fr; padding-left:12px; padding-right:12px; } }
  </style>
</head>
<body>
  <header>
    <h1>Crazyflie MJX Simulation</h1>
    <div id="subtitle" class="meta"></div>
  </header>
  <main>
    <div id="viewer"></div>
    <aside>
      <label>Rollout <select id="rolloutSelect"></select></label>
      <div class="row"><button id="playBtn">Play</button><label style="flex:1">Speed <input id="speed" type="range" min="1" max="8" value="1" /></label></div>
      <label>Step <input id="scrub" type="range" min="0" max="0" value="0" /></label>
      <label>Camera <select id="cameraMode"><option value="track" selected>Track</option><option value="free">Free orbit</option></select></label>
      <label>Agent <select id="agentSelect"></select></label>
      <div id="rolloutMeta" class="meta"></div>
      <table id="metrics"></table>
    </aside>
  </main>
  <script src="./report_data.js"></script>
  <script type="importmap">{ "imports": { "three": "./vendor/three.module.js" } }</script>
  <script type="module">
    import * as THREE from "./vendor/three.module.js";
    import { OrbitControls } from "./vendor/OrbitControls.js";
    import { OBJLoader } from "./vendor/OBJLoader.js";

    const report = window.CRAZYFLIE_MJX_REPORT || { summary:{}, rollouts:[] };
    const fmt = (v) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(4) : String(v);
    document.getElementById("subtitle").textContent = `${report.summary.scenario || ""} | dt=${report.summary.timestep || ""} | horizon=${report.summary.horizon || ""}`;
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const viewer = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeef2f5);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(4, -5, 3);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 3));
    viewer.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.8);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa8b5, 1.5));
    const light = new THREE.DirectionalLight(0xffffff, 1.5);
    light.position.set(3, -4, 6);
    scene.add(light);
    const grid = new THREE.GridHelper(7, 28, 0x8795a5, 0xc5ced8);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const loader = new OBJLoader();
    const robots = new THREE.Group();
    const trails = new THREE.Group();
    const boxes = new THREE.Group();
    scene.add(boxes, trails, robots);
    const robotParts = ["cf2_0.obj", "cf2_1.obj", "cf2_2.obj", "cf2_3.obj", "cf2_4.obj", "cf2_5.obj", "cf2_6.obj"];
    const materials = [
      new THREE.MeshStandardMaterial({ color:0x5368d8, roughness:0.45 }),
      new THREE.MeshStandardMaterial({ color:0x1c2f18, roughness:0.5 }),
      new THREE.MeshStandardMaterial({ color:0xd4b45f, roughness:0.35 }),
      new THREE.MeshStandardMaterial({ color:0xa5abb0, roughness:0.45 }),
      new THREE.MeshStandardMaterial({ color:0xd8d8d8, roughness:0.25 }),
      new THREE.MeshStandardMaterial({ color:0x202020, roughness:0.55 }),
      new THREE.MeshStandardMaterial({ color:0xffffff, roughness:0.45 }),
    ];
    const templates = [];
    for (let i = 0; i < robotParts.length; i++) {
      const obj = await loader.loadAsync(`./assets/cf2/${robotParts[i]}`);
      obj.traverse((child) => {
        if (!child.isMesh) return;
        child.geometry.computeVertexNormals();
        child.material = materials[i];
      });
      templates.push(obj);
    }

    let active = null, step = 0, playing = false, lastTime = null, agentIndex = 0;
    const rolloutSelect = document.getElementById("rolloutSelect");
    const agentSelect = document.getElementById("agentSelect");
    const scrub = document.getElementById("scrub");
    const playBtn = document.getElementById("playBtn");
    const speed = document.getElementById("speed");
    const cameraMode = document.getElementById("cameraMode");

    function disposeGroup(group) {
      while (group.children.length) group.remove(group.children[0]);
    }
    function quatWxyz(q) {
      return new THREE.Quaternion(q?.[1] || 0, q?.[2] || 0, q?.[3] || 0, q?.[0] ?? 1).normalize();
    }
    function buildRobot(agent) {
      const root = new THREE.Group();
      templates.forEach((template, i) => {
        const part = template.clone(true);
        if (i === 0 || i === 6) {
          const hue = (agent * 0.17 + 0.58) % 1;
          const mat = new THREE.MeshStandardMaterial({ color:new THREE.Color().setHSL(hue, 0.7, 0.48), roughness:0.45 });
          part.traverse((child) => { if (child.isMesh) child.material = mat; });
        }
        root.add(part);
      });
      return root;
    }
    function drawBoxes() {
      disposeGroup(boxes);
      const sceneSpec = active?.scene || {};
      for (const b of sceneSpec.goals || []) {
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(2*b.halfsize[0], 2*b.halfsize[1], 2*b.halfsize[2]),
          new THREE.MeshStandardMaterial({ color:0x2fa36b, transparent:true, opacity:0.28 })
        );
        mesh.position.set(...b.center);
        boxes.add(mesh);
      }
      for (const b of sceneSpec.obstacles || []) {
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(2*b.halfsize[0], 2*b.halfsize[1], 2*b.halfsize[2]),
          new THREE.MeshStandardMaterial({ color:0xdd6d6d, transparent:true, opacity:0.65 })
        );
        mesh.position.set(...b.center);
        boxes.add(mesh);
      }
    }
    function loadRollout(idx) {
      active = report.rollouts[idx];
      step = 0;
      playing = false;
      playBtn.textContent = "Play";
      scrub.max = Math.max(0, active.positions.length - 1);
      scrub.value = 0;
      agentSelect.innerHTML = "";
      const n = active.positions?.[0]?.length || 1;
      for (let i = 0; i < n; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `Agent ${i}`;
        agentSelect.appendChild(opt);
      }
      disposeGroup(robots);
      for (let i = 0; i < n; i++) robots.add(buildRobot(i));
      drawBoxes();
      updateStep(0);
      renderMetrics();
    }
    function updateStep(s) {
      if (!active) return;
      step = Math.max(0, Math.min(Number(s), active.positions.length - 1));
      scrub.value = step;
      const positions = active.positions[step] || [];
      const quats = active.quaternions[step] || [];
      for (let i = 0; i < robots.children.length; i++) {
        const p = positions[i] || [0, 0, 0];
        robots.children[i].position.set(p[0], p[1], p[2]);
        robots.children[i].quaternion.copy(quatWxyz(quats[i]));
      }
      drawTrails();
      if (cameraMode.value === "track") {
        const p = positions[agentIndex] || [0, 0, 0.5];
        const target = new THREE.Vector3(p[0], p[1], p[2] + 0.15);
        const offset = camera.position.clone().sub(controls.target);
        controls.target.copy(target);
        camera.position.copy(target).add(offset.length() > 0.1 ? offset : new THREE.Vector3(3, -4, 2.5));
      }
      document.getElementById("rolloutMeta").textContent = `${active.name || "rollout"} | step ${step}/${active.positions.length - 1}`;
    }
    function drawTrails() {
      disposeGroup(trails);
      if (!active || step < 1) return;
      const n = active.positions?.[0]?.length || 1;
      for (let i = 0; i < n; i++) {
        const points = [];
        for (let t = 0; t <= step; t++) {
          const p = active.positions[t]?.[i];
          if (p) points.push(new THREE.Vector3(p[0], p[1], p[2]));
        }
        if (points.length > 1) {
          trails.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color:new THREE.Color().setHSL((i * 0.17 + 0.58) % 1, 0.75, 0.4), transparent:true, opacity:0.8 })
          ));
        }
      }
    }
    function renderMetrics() {
      const rows = Object.entries(active?.metrics || {}).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
      document.getElementById("metrics").innerHTML = `<tr><th>Metric</th><th>Value</th></tr>${rows}`;
    }
    function resize() {
      const rect = viewer.getBoundingClientRect();
      renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
      camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
      camera.updateProjectionMatrix();
    }
    window.addEventListener("resize", resize);
    rolloutSelect.addEventListener("change", () => loadRollout(Number(rolloutSelect.value)));
    scrub.addEventListener("input", () => updateStep(scrub.value));
    playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; lastTime = null; });
    agentSelect.addEventListener("change", () => { agentIndex = Number(agentSelect.value) || 0; });
    (report.rollouts || []).forEach((r, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = r.name || `rollout ${i}`;
      rolloutSelect.appendChild(opt);
    });
    function animate(now) {
      requestAnimationFrame(animate);
      if (playing && active) {
        if (lastTime === null) lastTime = now;
        const dt = Math.max(1e-6, Number(active.dt || report.summary.timestep || 1/60));
        const advance = Math.floor(((now - lastTime) / 1000) * Number(speed.value || 1) / dt);
        if (advance > 0) {
          lastTime = now;
          updateStep((step + advance) % Math.max(1, active.positions.length));
        }
      } else {
        lastTime = null;
      }
      controls.update();
      renderer.render(scene, camera);
    }
    resize();
    if ((report.rollouts || []).length) loadRollout(0);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
