import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

// ---------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const el = {
  coreBadge: $("core-badge"),
  coreBanner: $("core-banner"),
  statusMesh: $("status-mesh"),
  statusElements: $("status-elements"),
  statusDofs: $("status-dofs"),
  statusSolver: $("status-solver"),
  statusItertime: $("status-itertime"),
  themeToggle: $("theme-toggle"),
  exampleSelect: $("example-select"),
  exampleLoadBtn: $("example-load-btn"),
  exampleDesc: $("example-desc"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  fileList: $("file-list"),
  loadTableBody: $("load-table-body"),
  addLoadBtn: $("add-load-btn"),
  symmetryRows: $("symmetry-rows"),
  addSymmetryBtn: $("add-symmetry-btn"),
  runBtn: $("run-btn"),
  stopBtn: $("stop-btn"),
  runMessage: $("run-message"),
  downloadStlBtn: $("download-stl-btn"),
  downloadNpzBtn: $("download-npz-btn"),
  viewport: $("viewport"),
  toggleInputs: $("toggle-inputs"),
  toggleDesign: $("toggle-design"),
  toggleAxes: $("toggle-axes"),
  fitViewBtn: $("fit-view-btn"),
  logOutput: $("log-output"),
  chartCanvas: $("convergence-chart"),
  statusQubo: $("status-qubo"),
  pOptimizer: $("p-optimizer"),
  quantumCard: $("quantum-settings-card"),
  quantumUnavailableMsg: $("quantum-unavailable-msg"),
  quantumSettingsBody: $("quantum-settings-body"),
  quboBackend: $("qubo-backend"),
  paperMethod: $("paper-method"),
  paperResetBtn: $("paper-reset-btn"),
  paperNote: $("paper-note"),
  resultCard: $("result-card"),
  resultBody: $("result-body"),
  resultPaper: $("result-paper"),
  designView: $("design-view"),
};

// ---------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------
const state = {
  files: new Map(), // id -> {id,name,triangles,bbox,role,visible,mesh}
  loads: [],        // [{file_id, fx, fy, fz}]
  symmetry: [],     // [{plane, direction}]
  selectedLoadRow: -1,
  examples: {},
  currentJobId: null,
  polling: null,
  designMesh: null,
  running: false,
  quantumBackends: [],      // [{name, available, needs_token, token_env, token_set, reason, kind, max_n, description}]
  continuumQuboSupported: false,
  activeTab: "setup",
  quboStatusText: "",
  // Paper settings (docs/PAPER_SETTINGS.md): the example whose presets the form
  // holds, and GET /api/examples/<name>/paper for it.
  paper: { example: null, info: null },
  loadedExample: null,
  truss: {
    benchmarks: [],         // list_benchmarks() payload
    methods: [],
    currentBenchmark: null, // full {"id",...} entry
    currentProblem: null,   // prob.to_dict()
    currentResult: null,    // last TrussResult.to_dict()
    jobId: null,
    polling: null,
    runs: [],               // session-scoped comparison table rows
  },
  study: {
    jobId: null,
    polling: null,
    logCursor: 0,
    customOptions: null,
    past: [],
    shownId: null,
    studyId: null,
  },
};

const ROLE_LABELS = {
  domain: "Domain",
  fixed: "Fixed (all)",
  xfixed: "Fixed X",
  yfixed: "Fixed Y",
  zfixed: "Fixed Z",
  keepdom: "Keep domain",
  load: "Load region",
  unused: "Unused",
};

// Functional BC role colours (slightly desaturated for the light, paper-like
// viewport; same values as the .sw-* legend swatches in style.css).
const ROLE_COLORS = {
  domain: 0xb5b5b1,
  fixed: 0x3d63c4,
  xfixed: 0x3fb0c4,
  yfixed: 0x2f9c8e,
  zfixed: 0x227a70,
  load: 0xe07b39,
  keepdom: 0x4caf6e,
};
// Viewport / design-mesh constants (nitipong.com palette: off-white paper,
// mid-grey design, ink edges).
const VIEW_COLORS = {
  background: 0xf4f4f2,
  hemiSky: 0xffffff,
  hemiGround: 0x5a5a58,
  design: 0x9a9a96,
  designRunning: 0xa9a9a5,
  designEdges: 0x141414,
  selectedLoad: 0xc0392b,
  loadArrow: 0xc0392b,
};

// Roles a single file can hold exclusively — assigning one of these to a
// file demotes any other file currently holding it back to "unused", so the
// job payload can never end up with two "domain"s (etc.). "load" and
// "unused" are not unique: many files can be load regions, and any number
// can be unused.
const UNIQUE_ROLES = new Set(["domain", "fixed", "xfixed", "yfixed", "zfixed", "keepdom"]);

// ---------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------
let scene, camera, renderer, controls, axesHelper, arrowHelper;
const inputGroup = new THREE.Group();
const designGroup = new THREE.Group();

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(VIEW_COLORS.background);

  const rect = el.viewport.getBoundingClientRect();
  camera = new THREE.PerspectiveCamera(45, rect.width / Math.max(rect.height, 1), 0.1, 100000);
  camera.position.set(150, 120, 200);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(rect.width, rect.height);
  el.viewport.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const hemi = new THREE.HemisphereLight(VIEW_COLORS.hemiSky, VIEW_COLORS.hemiGround, 1.1);
  scene.add(hemi);
  const dir1 = new THREE.DirectionalLight(0xffffff, 1.1);
  dir1.position.set(1, 1.4, 1);
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dir2.position.set(-1, -0.5, -1);
  scene.add(dir2);

  axesHelper = new THREE.AxesHelper(50);
  scene.add(axesHelper);

  scene.add(inputGroup);
  scene.add(designGroup);

  window.addEventListener("resize", onResize);
  onResize();
  animate();
}

function onResize() {
  const rect = el.viewport.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function fitView() {
  const box = new THREE.Box3();
  let any = false;
  inputGroup.children.forEach((m) => {
    if (m.visible) { box.expandByObject(m); any = true; }
  });
  designGroup.children.forEach((m) => {
    if (m.visible) { box.expandByObject(m); any = true; }
  });
  if (!any) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const dist = maxDim * 1.8;
  camera.position.set(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.7);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

// ---------------------------------------------------------------------
// STL loading helpers
// ---------------------------------------------------------------------
const stlLoader = new STLLoader();

async function fetchGeometry(url) {
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const geom = stlLoader.parse(buf);
  geom.computeVertexNormals();
  return geom;
}

function materialForRole(role, selected) {
  const color = ROLE_COLORS[role] ?? 0x888888;
  if (role === "domain") {
    return new THREE.MeshStandardMaterial({
      color, transparent: true, opacity: 0.22, depthWrite: false,
      side: THREE.DoubleSide, roughness: 0.9,
    });
  }
  if (role === "load") {
    return new THREE.MeshStandardMaterial({
      color: selected ? VIEW_COLORS.selectedLoad : color,
      transparent: true, opacity: selected ? 0.9 : 0.55,
      side: THREE.DoubleSide, roughness: 0.6,
    });
  }
  return new THREE.MeshStandardMaterial({
    color, transparent: true, opacity: 0.85, side: THREE.DoubleSide, roughness: 0.6,
  });
}

async function ensureInputMesh(rec) {
  if (rec.mesh) return rec.mesh;
  try {
    const geom = await fetchGeometry(`/api/files/${rec.id}`);
    const mesh = new THREE.Mesh(geom, materialForRole(rec.role, false));
    mesh.userData.fileId = rec.id;
    rec.mesh = mesh;
    inputGroup.add(mesh);
    updateMeshVisibility(rec);
    return mesh;
  } catch (err) {
    console.error("Failed to load STL for preview", rec.name, err);
    return null;
  }
}

function updateMeshVisibility(rec) {
  if (!rec.mesh) return;
  const roleVisible = rec.role && rec.role !== "unused";
  rec.mesh.visible = !!rec.visible && !!roleVisible && el.toggleInputs.checked;
}

function refreshAllMeshMaterials() {
  for (const rec of state.files.values()) {
    if (rec.mesh) {
      const isSelectedLoad = rec.role === "load" && isFileInSelectedRow(rec.id);
      rec.mesh.material.dispose();
      rec.mesh.material = materialForRole(rec.role, isSelectedLoad);
      updateMeshVisibility(rec);
    }
  }
}

function isFileInSelectedRow(fileId) {
  if (state.selectedLoadRow < 0) return false;
  const row = state.loads[state.selectedLoadRow];
  return row && row.file_id === fileId;
}

function getDomainRecord() {
  for (const rec of state.files.values()) {
    if (rec.role === "domain") return rec;
  }
  return null;
}

function bboxDiagonal(bbox) {
  return Math.hypot(
    bbox.max[0] - bbox.min[0],
    bbox.max[1] - bbox.min[1],
    bbox.max[2] - bbox.min[2]
  );
}

// Fraction of the *domain's* bounding-box diagonal used for the force-arrow
// length — scaling by the (possibly tiny) load region's own bbox instead
// made arrows for small regions nearly invisible.
const ARROW_LENGTH_FRACTION = 0.18;

function updateArrowHelper() {
  if (arrowHelper) {
    scene.remove(arrowHelper);
    arrowHelper = null;
  }
  const row = state.loads[state.selectedLoadRow];
  if (!row || !row.file_id) return;
  const rec = state.files.get(row.file_id);
  if (!rec || !rec.bbox) return;
  const mag = Math.hypot(row.fx, row.fy, row.fz);
  if (mag < 1e-9) return;
  const dir = new THREE.Vector3(row.fx, row.fy, row.fz).normalize();
  const center = new THREE.Vector3(
    (rec.bbox.min[0] + rec.bbox.max[0]) / 2,
    (rec.bbox.min[1] + rec.bbox.max[1]) / 2,
    (rec.bbox.min[2] + rec.bbox.max[2]) / 2
  );
  const domainRec = getDomainRecord();
  const refDiag = (domainRec && bboxDiagonal(domainRec.bbox)) || bboxDiagonal(rec.bbox) || 10;
  const length = Math.max(refDiag * ARROW_LENGTH_FRACTION, 5);
  // Anchor the tail one arrow-length back along -dir so the ARROWHEAD sits
  // at the region centroid and the shaft extends outward from it — a small
  // load region no longer buries a tiny arrow inside the geometry.
  const tail = center.clone().addScaledVector(dir, -length);
  arrowHelper = new THREE.ArrowHelper(dir, tail, length, VIEW_COLORS.loadArrow, length * 0.25, length * 0.12);
  scene.add(arrowHelper);
}

// ---------------------------------------------------------------------
// File list UI
// ---------------------------------------------------------------------
function addFileRecord(rec) {
  rec.role = rec.role || "unused";
  rec.visible = rec.visible !== false;
  rec.mesh = null;
  state.files.set(rec.id, rec);
  renderFileList();
  renderLoadTable(); // load-file dropdowns depend on files marked role=load
  ensureInputMesh(rec);
}

function disposeFileMesh(rec) {
  if (rec.mesh) {
    inputGroup.remove(rec.mesh);
    rec.mesh.geometry?.dispose();
    rec.mesh.material?.dispose();
    rec.mesh = null;
  }
}

// Clears every uploaded/example file, its three.js mesh, all load-case rows
// and the force arrow — used before loading a new example so a second
// example can never leave the first one's files (and roles, e.g. two
// "domain"s) behind. See applyPrefill().
function resetInputs() {
  for (const rec of state.files.values()) disposeFileMesh(rec);
  state.files.clear();
  state.loads = [];
  state.selectedLoadRow = -1;
  if (arrowHelper) {
    scene.remove(arrowHelper);
    arrowHelper = null;
  }
  renderFileList();
  renderLoadTable();
}

// Assigns `role` to file `fileId`, demoting any other file currently
// holding a *unique* role (domain/fixed/xfixed/yfixed/zfixed/keepdom) back
// to "unused" first, so at most one file ever holds each unique role.
function setFileRole(fileId, role) {
  const rec = state.files.get(fileId);
  if (!rec) return;
  if (UNIQUE_ROLES.has(role)) {
    for (const other of state.files.values()) {
      if (other.id !== fileId && other.role === role) {
        other.role = "unused";
        updateMeshVisibility(other);
      }
    }
  }
  rec.role = role;
  updateMeshVisibility(rec);
}

// Returns the list of unique roles currently held by more than one file —
// should always be empty given setFileRole()/resetInputs(), but runJob()
// checks this defensively before ever submitting a job.
function findRoleConflicts() {
  const counts = {};
  for (const rec of state.files.values()) {
    if (UNIQUE_ROLES.has(rec.role)) counts[rec.role] = (counts[rec.role] || 0) + 1;
  }
  return Object.entries(counts).filter(([, n]) => n > 1).map(([role]) => role);
}

function renderFileList() {
  el.fileList.innerHTML = "";
  for (const rec of state.files.values()) {
    const row = document.createElement("div");
    row.className = "file-row";

    const vis = document.createElement("input");
    vis.type = "checkbox";
    vis.checked = rec.visible;
    vis.title = "Show/hide in viewport";
    vis.addEventListener("change", () => {
      rec.visible = vis.checked;
      updateMeshVisibility(rec);
    });

    const nameWrap = document.createElement("div");
    const nameSpan = document.createElement("div");
    nameSpan.className = "file-name";
    nameSpan.textContent = rec.name;
    nameSpan.title = rec.name;
    const meta = document.createElement("span");
    meta.className = "file-meta";
    meta.textContent = `${rec.triangles} tris`;
    if (rec.watertight === false) {
      const warn = document.createElement("span");
      warn.className = "file-meta file-warning";
      warn.textContent = "⚠ not watertight";
      warn.title = "This mesh has unpaired (open or non-manifold) edges. Non-watertight " +
        "geometry may cause holes or unexpected results in some FreeTO stages.";
      meta.appendChild(document.createTextNode(" · "));
      meta.appendChild(warn);
    }
    nameWrap.appendChild(nameSpan);
    nameWrap.appendChild(meta);

    const roleSelect = document.createElement("select");
    for (const [value, label] of Object.entries(ROLE_LABELS)) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      if (rec.role === value) opt.selected = true;
      roleSelect.appendChild(opt);
    }
    roleSelect.addEventListener("change", () => {
      setFileRole(rec.id, roleSelect.value);
      // Re-render the whole list: setFileRole may have demoted a DIFFERENT
      // file's role select (unique-role enforcement), which this row's own
      // element can't reflect.
      renderFileList();
      refreshAllMeshMaterials();
      renderLoadTable();
    });

    row.appendChild(vis);
    row.appendChild(nameWrap);
    row.appendChild(roleSelect);
    el.fileList.appendChild(row);
  }
}

// ---------------------------------------------------------------------
// Load-case table
// ---------------------------------------------------------------------
function loadEligibleFiles() {
  return Array.from(state.files.values()).filter((r) => r.role === "load");
}

function addLoadRow(preset) {
  if (state.loads.length >= 10) return;
  state.loads.push({ file_id: preset?.file_id || "", fx: preset?.fx ?? 0, fy: preset?.fy ?? 0, fz: preset?.fz ?? 0 });
  renderLoadTable();
}

function removeLoadRow(idx) {
  state.loads.splice(idx, 1);
  if (state.selectedLoadRow === idx) state.selectedLoadRow = -1;
  else if (state.selectedLoadRow > idx) state.selectedLoadRow -= 1;
  renderLoadTable();
}

function renderLoadTable() {
  el.loadTableBody.innerHTML = "";
  const eligible = loadEligibleFiles();
  state.loads.forEach((row, idx) => {
    const tr = document.createElement("tr");
    if (idx === state.selectedLoadRow) tr.classList.add("selected-row");

    const tdIdx = document.createElement("td");
    tdIdx.textContent = `F${idx + 1}`;
    tdIdx.style.cursor = "pointer";
    tdIdx.title = "Click to preview this load case's direction";
    tdIdx.addEventListener("click", () => {
      state.selectedLoadRow = state.selectedLoadRow === idx ? -1 : idx;
      renderLoadTable();
      refreshAllMeshMaterials();
      updateArrowHelper();
    });

    const tdFile = document.createElement("td");
    const sel = document.createElement("select");
    const emptyOpt = document.createElement("option");
    emptyOpt.value = "";
    emptyOpt.textContent = eligible.length ? "select file…" : "(mark a file as 'Load region' first)";
    sel.appendChild(emptyOpt);
    for (const rec of eligible) {
      const opt = document.createElement("option");
      opt.value = rec.id;
      opt.textContent = rec.name;
      if (row.file_id === rec.id) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => {
      row.file_id = sel.value;
      if (idx === state.selectedLoadRow) { refreshAllMeshMaterials(); updateArrowHelper(); }
    });
    tdFile.appendChild(sel);

    const mkNumCell = (key) => {
      const td = document.createElement("td");
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "any";
      inp.value = row[key];
      inp.addEventListener("input", () => {
        row[key] = parseFloat(inp.value) || 0;
        if (idx === state.selectedLoadRow) updateArrowHelper();
      });
      td.appendChild(inp);
      return td;
    };

    const tdRemove = document.createElement("td");
    const rmBtn = document.createElement("button");
    rmBtn.textContent = "✕";
    rmBtn.className = "btn small row-remove-btn";
    rmBtn.title = "Remove load case";
    rmBtn.addEventListener("click", () => removeLoadRow(idx));
    tdRemove.appendChild(rmBtn);

    tr.appendChild(tdIdx);
    tr.appendChild(tdFile);
    tr.appendChild(mkNumCell("fx"));
    tr.appendChild(mkNumCell("fy"));
    tr.appendChild(mkNumCell("fz"));
    tr.appendChild(tdRemove);
    el.loadTableBody.appendChild(tr);
  });
}

el.addLoadBtn.addEventListener("click", () => addLoadRow());

// ---------------------------------------------------------------------
// Symmetry rows
// ---------------------------------------------------------------------
function addSymmetryRow(preset) {
  if (state.symmetry.length >= 3) return;
  state.symmetry.push({ plane: preset?.plane || "x-y", direction: preset?.direction || "right" });
  renderSymmetryRows();
}
function removeSymmetryRow(idx) {
  state.symmetry.splice(idx, 1);
  renderSymmetryRows();
}
function renderSymmetryRows() {
  el.symmetryRows.innerHTML = "";
  state.symmetry.forEach((row, idx) => {
    const div = document.createElement("div");
    div.className = "symmetry-row";
    const planeSel = document.createElement("select");
    ["x-y", "y-z", "z-x"].forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p; opt.textContent = p;
      if (row.plane === p) opt.selected = true;
      planeSel.appendChild(opt);
    });
    planeSel.addEventListener("change", () => (row.plane = planeSel.value));
    const dirSel = document.createElement("select");
    ["right", "left"].forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d; opt.textContent = d;
      if (row.direction === d) opt.selected = true;
      dirSel.appendChild(opt);
    });
    dirSel.addEventListener("change", () => (row.direction = dirSel.value));
    const rm = document.createElement("button");
    rm.textContent = "✕"; rm.className = "btn small";
    rm.addEventListener("click", () => removeSymmetryRow(idx));
    div.appendChild(planeSel);
    div.appendChild(dirSel);
    div.appendChild(rm);
    el.symmetryRows.appendChild(div);
  });
}
el.addSymmetryBtn.addEventListener("click", () => addSymmetryRow());

// ---------------------------------------------------------------------
// Uploading
// ---------------------------------------------------------------------
async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const resp = await fetch("/api/upload", { method: "POST", body: form });
  if (!resp.ok) {
    const detail = await safeErrorDetail(resp);
    setRunMessage(`Upload failed: ${detail}`, true);
    return;
  }
  const records = await resp.json();
  records.forEach((rec) => addFileRecord(rec));
}

el.fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));
["dragenter", "dragover"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); })
);
el.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

// ---------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------
const CATEGORY_LABELS = {
  qff3d: "Paper examples (manuscript Table 1)",
  paper: "FreeTO examples",
  beam: "Beam-like",
  "truss-like": "Truss-like continuum",
  advanced: "Advanced",
};
const CATEGORY_ORDER = ["qff3d", "paper", "beam", "truss-like", "advanced"];

// Out-of-the-box path: the example select starts on DEFAULT_EXAMPLE and every
// parameter field holds the paper's value for it (QUBO-SA with the block
// Hessian, seed 0, the paper mesh); loading a paper example, or "Paper
// settings", puts the paper values of the selected paper run back.
const DEFAULT_EXAMPLE = "cantilever_beam";
const PAPER_METHOD_TITLES = {
  "QUBO-sa (block)": "QUBO-SA (block Hessian)",
  "MMA": "MMA (reference)",
  "BESO-sort": "BESO sorting",
  "QUBO-sa (diag)": "QUBO-SA (diag Hessian)",
  "QUBO-qaoa kb8 p1 penalty (+greedy)": "QUBO-QAOA (blocks of 8, p = 1, +greedy)",
  "BESO-sort (move 0.04)": "BESO sorting, move 0.04 (control)",
  "QUBO-sa (scalar)": "QUBO-SA (scalar, control)",
  "QUBO-sa (block, Qx3)": "QUBO-SA (block, Qx3, control)",
  "OC (FreeTO default, flagged)": "OC (FreeTO default, flagged)",
};

async function fetchExamples() {
  const resp = await fetch("/api/examples");
  const list = await resp.json();
  el.exampleSelect.innerHTML = '<option value="">Choose an example…</option>';

  // Group by category into <optgroup>s; the five examples of the manuscript
  // come first in their own group (in the paper's order).
  const byCategory = new Map();
  for (const ex of list) {
    state.examples[ex.name] = ex;
    const cat = ex.paper_example ? "qff3d" : (ex.category || "paper");
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat).push(ex);
  }
  const paperOrder = ["cantilever_beam", "mbb_beam", "bridge_deck", "l_bracket", "GE_bracket"];
  if (byCategory.has("qff3d")) {
    byCategory.get("qff3d").sort((a, b) => paperOrder.indexOf(a.name) - paperOrder.indexOf(b.name));
  }
  const orderedCats = [
    ...CATEGORY_ORDER.filter((c) => byCategory.has(c)),
    ...[...byCategory.keys()].filter((c) => !CATEGORY_ORDER.includes(c)),
  ];
  for (const cat of orderedCats) {
    const group = document.createElement("optgroup");
    group.label = CATEGORY_LABELS[cat] || cat;
    for (const ex of byCategory.get(cat)) {
      const opt = document.createElement("option");
      opt.value = ex.name;
      opt.textContent = ex.title;
      group.appendChild(opt);
    }
    el.exampleSelect.appendChild(group);
  }
  if (state.examples[DEFAULT_EXAMPLE]) {
    el.exampleSelect.value = DEFAULT_EXAMPLE;
    el.exampleDesc.textContent = state.examples[DEFAULT_EXAMPLE].description || "";
  }
}
el.exampleSelect.addEventListener("change", () => {
  const ex = state.examples[el.exampleSelect.value];
  el.exampleDesc.textContent = ex ? ex.description : "";
});

// --- paper settings ----------------------------------------------------
async function fetchPaperInfo(name) {
  try {
    const resp = await fetch(`/api/examples/${name}/paper`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

function setPaperInfo(name, info) {
  state.paper = { example: name, info: info || null };
  const methods = (info && info.is_paper && info.methods) || [];
  const keep = el.paperMethod.value;
  el.paperMethod.innerHTML = "";
  if (!methods.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "paper protocol (QUBO-SA, block)";
    el.paperMethod.appendChild(o);
  }
  for (const m of methods) {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = PAPER_METHOD_TITLES[m] || m;
    el.paperMethod.appendChild(o);
  }
  el.paperMethod.value = methods.includes(keep) ? keep : (info && info.default_method) || (methods[0] || "");
  const ex = state.examples[name];
  if (info && info.is_paper) {
    const p = info.presets[info.default_method] || {};
    el.paperNote.textContent = `Paper settings for ${ex ? ex.title : name}: MeshControl ${p.mesh_control}, ` +
      `V* = ${p.volfrac}, E = ${Number(p.youngs_modulus) / 1e9} GPa, ν = ${p.poisson_ratio}, q = ${p.penal}, ` +
      `rmin = ${p.rmin}, cap ${p.max_iter} iterations, seed 0, refined evaluation f = ${p.eval_refined}.`;
  } else {
    el.paperNote.textContent = "Not one of the paper's examples: the paper's run protocol (QUBO-SA with the " +
      "block Hessian, seed 0, cap 300, refined evaluation f = 2) on the example's own mesh and volume fraction.";
  }
}

function currentPreset() {
  const info = state.paper.info;
  if (!info) return null;
  if (info.is_paper) return info.presets[el.paperMethod.value] || info.presets[info.default_method] || null;
  return info.protocol_preset || null;
}

function setSelectValue(id, value) {
  const sel = $(id);
  const v = value == null ? "" : String(value);
  if (![...sel.options].some((o) => o.value === v)) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    sel.appendChild(o);
  }
  sel.value = v;
}

// Sets every form field from a preset / prefill object whose keys are the
// POST /api/jobs field names (webapp/paper_presets.py).
function applyPreset(p) {
  if (!p) return;
  const num = (id, v) => { if (v !== undefined) $(id).value = v == null ? "" : v; };
  const chk = (id, v) => { if (v !== undefined && v !== null) $(id).checked = !!v; };
  num("p-mesh_control", p.mesh_control);
  num("p-volfrac", p.volfrac);
  num("p-youngs_modulus", p.youngs_modulus);
  num("p-poisson_ratio", p.poisson_ratio);
  if (p.method) $("p-method").value = String(p.method).toUpperCase();
  if (p.optimizer) {
    const want = String(p.optimizer).toUpperCase();
    const opt = [...el.pOptimizer.options].find((o) => o.value === want && !o.disabled);
    if (opt) el.pOptimizer.value = want;
  }
  num("p-penal", p.penal);
  num("p-rmin", p.rmin);
  if (p.loadtype) $("p-loadtype").value = p.loadtype;
  chk("p-keep_bc", p.keep_bc);
  chk("p-keep_bcx", p.keep_bcx);
  chk("p-keep_bcy", p.keep_bcy);
  chk("p-keep_bcz", p.keep_bcz);
  num("p-max_iter", p.max_iter);
  if (p.solver) $("p-solver").value = p.solver;
  chk("eval_crisp", p.eval_crisp);
  chk("p-audit", p.audit);
  if (p.eval_refined !== undefined) setSelectValue("p-eval_refined", p.eval_refined);
  num("p-eval_beta", p.eval_beta);
  if (p.mma_constraint) $("p-mma_constraint").value = p.mma_constraint;
  chk("p-mma_feasible_stop", p.mma_feasible_stop);
  chk("eval_binary", p.eval_binary);
  // QUBO fields (all present in a preset; the form shows the ones a paper run changes)
  if (p.qubo_backend !== undefined) setSelectValue("qubo-backend", p.qubo_backend || "auto");
  if (p.qubo_hessian) setSelectValue("qubo-hessian", p.qubo_hessian);
  if (p.qubo_volume) $("qubo-volume").value = p.qubo_volume;
  num("qubo-frontier_fraction", p.qubo_frontier_fraction);
  num("qubo-block_size", p.qubo_block_size);
  num("qubo-sweeps", p.qubo_sweeps);
  num("qubo-num_reads", p.qubo_num_reads);
  num("qubo-qaoa_p", p.qubo_qaoa_p);
  num("qubo-qaoa_shots", p.qubo_qaoa_shots);
  num("qubo-seed", p.qubo_seed);
  num("qubo-lambda_q", p.qubo_lambda_q);
  num("qubo-gamma", p.qubo_gamma);
  num("qubo-move_penalty", p.qubo_move_penalty);
  if (p.qubo_blocks) $("qubo-blocks").value = p.qubo_blocks;
  if (p.qubo_init) $("qubo-init").value = p.qubo_init;
  num("qubo-er", p.qubo_er);
  num("qubo-n_warm", p.qubo_n_warm);
  num("qubo-patience", p.qubo_patience);
  if (p.qubo_qaoa_init) $("qubo-qaoa_init").value = p.qubo_qaoa_init;
  num("qubo-time_limit", p.qubo_time_limit);
  if (p.qubo_interp) $("qubo-interp").value = p.qubo_interp;
  num("qubo-move_limit", p.qubo_move_limit);
  num("qubo-move_limit_min", p.qubo_move_limit_min);
  num("qubo-guard_tol", p.qubo_guard_tol);
  num("qubo-guard_tol_target", p.qubo_guard_tol_target);
  num("qubo-max_rejects", p.qubo_max_rejects);
  num("qubo-hessian_scale", p.qubo_hessian_scale);
  chk("qubo-verify_exact", p.qubo_verify_exact);
  chk("qubo-protect_loads", p.qubo_protect_loads);
  chk("qubo-guard", p.qubo_guard);
  chk("qubo-qaoa_polish", p.qubo_qaoa_polish);
  chk("qubo-connectivity", p.qubo_connectivity);
  updateQuantumPanelVisibility();
}

function applyPaperSettings() {
  const p = currentPreset();
  if (!p) return;
  applyPreset(p);
  const m = el.paperMethod.value;
  setRunMessage(state.paper.info && state.paper.info.is_paper
    ? `Paper settings applied: ${PAPER_METHOD_TITLES[m] || m}, seed 0.`
    : "Paper run protocol applied.", false);
}

el.paperMethod.addEventListener("change", applyPaperSettings);
el.paperResetBtn.addEventListener("click", async () => {
  const name = state.loadedExample || el.exampleSelect.value || DEFAULT_EXAMPLE;
  if (state.paper.example !== name) setPaperInfo(name, await fetchPaperInfo(name));
  applyPaperSettings();
});

// Initial state: every field holds the paper value of the default example.
async function initPaperDefaults() {
  const info = await fetchPaperInfo(DEFAULT_EXAMPLE);
  if (!info || !info.available) return;
  setPaperInfo(DEFAULT_EXAMPLE, info);
  applyPreset(currentPreset());
}

async function loadExample(name) {
  const resp = await fetch(`/api/examples/${name}/load`, { method: "POST" });
  if (!resp.ok) {
    setRunMessage(`Could not load example: ${await safeErrorDetail(resp)}`, true);
    return;
  }
  const data = await resp.json();
  state.loadedExample = name;
  if (data.paper) setPaperInfo(name, data.paper);
  applyPrefill(data.prefill, data.files);
  // the selected paper run (QUBO-SA block by default) of this example
  const p = currentPreset();
  if (p) applyPreset(p);
}

function applyPrefill(prefill, filesMap) {
  // Loading an example replaces the whole problem setup: clear every
  // previously uploaded/example file (and dispose its mesh) first, so a
  // second example load can never leave the first one's files/roles behind
  // (which used to produce two "domain" files and a job built from a mix of
  // both examples — see VERIFICATION.md #2).
  resetInputs();

  const roleForId = new Map();
  if (prefill.domain_id) roleForId.set(prefill.domain_id, "domain");
  if (prefill.fixed_id) roleForId.set(prefill.fixed_id, "fixed");
  if (prefill.xfixed_id) roleForId.set(prefill.xfixed_id, "xfixed");
  if (prefill.yfixed_id) roleForId.set(prefill.yfixed_id, "yfixed");
  if (prefill.zfixed_id) roleForId.set(prefill.zfixed_id, "zfixed");
  if (prefill.keepdom_id) roleForId.set(prefill.keepdom_id, "keepdom");
  for (const l of prefill.loads) roleForId.set(l.file_id, "load");

  // The same file can appear under several filesMap keys (e.g. the GE
  // example's force STL is both force1 and force2) — register each
  // distinct file id once, or the second registration would silently
  // orphan (leak) the mesh created for the first.
  const seenIds = new Set();
  for (const rec of Object.values(filesMap)) {
    if (!rec || seenIds.has(rec.id)) continue;
    seenIds.add(rec.id);
    addFileRecord({ ...rec, role: roleForId.get(rec.id) || "unused" });
  }
  renderFileList();
  refreshAllMeshMaterials();

  state.loads = prefill.loads.map((l) => ({ ...l }));
  state.selectedLoadRow = state.loads.length ? 0 : -1;
  renderLoadTable();
  updateArrowHelper();
  refreshAllMeshMaterials();

  state.symmetry = (prefill.symmetry || []).map((s) => ({ ...s }));
  renderSymmetryRows();

  // Every run setting: the server's prefill holds the paper values for a
  // paper example (webapp/paper_presets.py) and the example's own values
  // plus the paper protocol otherwise.
  applyPreset(prefill);

  setTimeout(fitView, 300);
}

el.exampleLoadBtn.addEventListener("click", () => {
  const name = el.exampleSelect.value;
  if (!name) { setRunMessage("Pick an example first.", true); return; }
  loadExample(name);
});

// ---------------------------------------------------------------------
// Run / Stop
// ---------------------------------------------------------------------
function setRunMessage(msg, isError) {
  el.runMessage.textContent = msg;
  el.runMessage.classList.toggle("error", !!isError);
  el.runMessage.classList.toggle("ok", !isError && !!msg);
}

function buildJobPayload() {
  const domainRec = Array.from(state.files.values()).find((r) => r.role === "domain");
  const fixedRec = Array.from(state.files.values()).find((r) => r.role === "fixed");
  const xfixedRec = Array.from(state.files.values()).find((r) => r.role === "xfixed");
  const yfixedRec = Array.from(state.files.values()).find((r) => r.role === "yfixed");
  const zfixedRec = Array.from(state.files.values()).find((r) => r.role === "zfixed");
  const keepdomRec = Array.from(state.files.values()).find((r) => r.role === "keepdom");

  const exampleTitle = el.exampleSelect.value ? state.examples[el.exampleSelect.value]?.title : null;
  return {
    label: exampleTitle || (domainRec ? domainRec.name : "Job"),
    domain_id: domainRec ? domainRec.id : null,
    loads: state.loads
      .filter((l) => l.file_id)
      .map((l) => ({ file_id: l.file_id, fx: l.fx, fy: l.fy, fz: l.fz })),
    fixed_id: fixedRec ? fixedRec.id : null,
    xfixed_id: xfixedRec ? xfixedRec.id : null,
    yfixed_id: yfixedRec ? yfixedRec.id : null,
    zfixed_id: zfixedRec ? zfixedRec.id : null,
    keepdom_id: keepdomRec ? keepdomRec.id : null,
    mesh_control: parseInt($("p-mesh_control").value, 10),
    volfrac: parseFloat($("p-volfrac").value),
    youngs_modulus: parseFloat($("p-youngs_modulus").value),
    poisson_ratio: parseFloat($("p-poisson_ratio").value),
    method: $("p-method").value,
    optimizer: $("p-optimizer").value,
    penal: parseFloat($("p-penal").value),
    rmin: parseFloat($("p-rmin").value),
    loadtype: $("p-loadtype").value,
    keep_bc: $("p-keep_bc").checked,
    keep_bcx: $("p-keep_bcx").checked,
    keep_bcy: $("p-keep_bcy").checked,
    keep_bcz: $("p-keep_bcz").checked,
    symmetry: state.symmetry.map((s) => ({ plane: s.plane, direction: s.direction })),
    max_iter: parseInt($("p-max_iter").value, 10),
    solver: $("p-solver").value,
    eval_crisp: $("eval_crisp").checked,
    eval_refined: intOrNull("p-eval_refined"),
    eval_beta: numOrNull("p-eval_beta"),
    mma_constraint: $("p-mma_constraint").value,
    mma_feasible_stop: $("p-mma_feasible_stop").checked,
    audit: $("p-audit").checked,
    // the loaded example (only used to compare the result with the paper's values)
    example: state.loadedExample,
    ...($("p-optimizer").value === "QUBO" ? buildQuboPayload() : {}),
  };
}

// ---------------------------------------------------------------------
// Quantum / QUBO settings (Setup panel, optimizer="QUBO")
// ---------------------------------------------------------------------
function numOrNull(id) {
  const v = $(id).value;
  if (v === "") return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}
function intOrNull(id) {
  const v = $(id).value;
  if (v === "") return null;
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? null : n;
}
function strOrNull(id) {
  const v = $(id).value;
  return v === "" ? null : v;
}

function buildQuboPayload() {
  return {
    qubo_backend: strOrNull("qubo-backend"),
    qubo_hessian: strOrNull("qubo-hessian"),
    qubo_volume: strOrNull("qubo-volume"),
    qubo_lambda_q: numOrNull("qubo-lambda_q"),
    qubo_gamma: numOrNull("qubo-gamma"),
    qubo_move_penalty: numOrNull("qubo-move_penalty"),
    qubo_frontier_fraction: numOrNull("qubo-frontier_fraction"),
    qubo_block_size: intOrNull("qubo-block_size"),
    qubo_blocks: strOrNull("qubo-blocks"),
    qubo_sweeps: intOrNull("qubo-sweeps"),
    qubo_init: strOrNull("qubo-init"),
    qubo_er: numOrNull("qubo-er"),
    qubo_n_warm: intOrNull("qubo-n_warm"),
    qubo_patience: intOrNull("qubo-patience"),
    qubo_num_reads: intOrNull("qubo-num_reads"),
    qubo_seed: intOrNull("qubo-seed"),
    qubo_qaoa_p: intOrNull("qubo-qaoa_p"),
    qubo_qaoa_shots: intOrNull("qubo-qaoa_shots"),
    qubo_qaoa_init: strOrNull("qubo-qaoa_init"),
    qubo_time_limit: numOrNull("qubo-time_limit"),
    qubo_verify_exact: $("qubo-verify_exact").checked,
    qubo_interp: strOrNull("qubo-interp"),
    qubo_move_limit: numOrNull("qubo-move_limit"),
    qubo_move_limit_min: numOrNull("qubo-move_limit_min"),
    qubo_guard_tol: numOrNull("qubo-guard_tol"),
    qubo_guard_tol_target: numOrNull("qubo-guard_tol_target"),
    qubo_max_rejects: intOrNull("qubo-max_rejects"),
    qubo_protect_loads: $("qubo-protect_loads").checked,
    qubo_guard: $("qubo-guard").checked,
    qubo_qaoa_polish: $("qubo-qaoa_polish").checked,
    qubo_hessian_scale: numOrNull("qubo-hessian_scale"),
    qubo_connectivity: $("qubo-connectivity").checked,
    eval_binary: $("eval_binary").checked,
  };
}

// Populates a <select> with backend entries from GET /api/quantum/backends
// (name, availability, install/token hint) — shared by the Setup panel's
// QUBO backend select and the Truss tab's backend select. Unavailable
// backends stay listed but disabled, with the reason as a tooltip, so the
// researcher can see *what* would need installing/configuring.
function populateBackendSelect(selectEl, backends, { includeAuto = true } = {}) {
  if (!selectEl) return;
  const current = selectEl.value;
  selectEl.innerHTML = "";
  if (includeAuto) {
    const opt = document.createElement("option");
    opt.value = "auto";
    opt.textContent = "auto";
    opt.title = "exact for tiny free sets, else simulated annealing — never a cloud backend unless named";
    selectEl.appendChild(opt);
  }
  for (const b of backends) {
    const opt = document.createElement("option");
    opt.value = b.name;
    opt.textContent = b.kind ? `${b.name} (${b.kind})` : b.name;
    if (!b.available) {
      opt.disabled = true;
      opt.textContent += " — unavailable";
    }
    opt.title = b.available
      ? (b.description || "")
      : (b.reason || "not available");
    selectEl.appendChild(opt);
  }
  if ([...selectEl.options].some((o) => o.value === current)) selectEl.value = current;
}

async function fetchQuantumBackends() {
  let data;
  try {
    const resp = await fetch("/api/quantum/backends");
    data = await resp.json();
  } catch {
    data = { quantum_available: false, import_error: "request to /api/quantum/backends failed", backends: [], continuum_qubo_supported: false };
  }
  state.quantumBackends = data.backends || [];
  state.quantumAvailable = !!data.quantum_available;
  state.quantumImportError = data.import_error || null;
  state.continuumQuboSupported = !!data.continuum_qubo_supported;

  // Feature-detect: if freeto.quantum isn't importable at all, hide the
  // QUBO optimizer option outright rather than let the user pick a dead end.
  const qOpt = el.pOptimizer.querySelector('option[value="QUBO"]');
  if (qOpt) {
    qOpt.disabled = !state.quantumAvailable;
    qOpt.title = state.quantumAvailable
      ? "QUBO / quantum design update (freeto.quantum)"
      : `freeto.quantum is not available: ${state.quantumImportError || "not installed"}`;
    if (!state.quantumAvailable && el.pOptimizer.value === "QUBO") {
      el.pOptimizer.value = "OC";
    }
  }

  populateBackendSelect(el.quboBackend, state.quantumBackends, { includeAuto: true });
  updateQuantumPanelVisibility();
}

function updateQuantumPanelVisibility() {
  const isQubo = el.pOptimizer.value === "QUBO";
  el.quantumCard.hidden = !isQubo;
  if (!isQubo) return;
  const blocked = !state.quantumAvailable || !state.continuumQuboSupported;
  el.quantumUnavailableMsg.hidden = !blocked;
  el.quantumSettingsBody.style.display = state.quantumAvailable ? "" : "none";
  if (blocked) {
    el.quantumUnavailableMsg.textContent = !state.quantumAvailable
      ? `freeto.quantum is not available, so QUBO jobs cannot run. ${state.quantumImportError || ""}`
      : "This installed 'freeto' core does not wire optimizer=\"QUBO\" into the continuum loop yet " +
        "(freeto.quantum's backends are available, but FreeTOConfig has no 'qubo' field) — Run will refuse with a clear error.";
  }
}

el.pOptimizer.addEventListener("change", updateQuantumPanelVisibility);

async function safeErrorDetail(resp) {
  try {
    const data = await resp.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return resp.statusText;
  }
}

async function runJob() {
  const conflicts = findRoleConflicts();
  if (conflicts.length) {
    const labels = conflicts.map((r) => ROLE_LABELS[r] || r).join(", ");
    setRunMessage(`Multiple files are marked as the same role (${labels}). Each of Domain / Fixed* / Keep domain must be assigned to only one file.`, true);
    return;
  }
  const payload = buildJobPayload();
  if (!payload.domain_id) { setRunMessage("Mark one file's role as 'Domain'.", true); return; }
  if (!payload.loads.length) { setRunMessage("Add at least one load case with a file selected.", true); return; }
  if (payload.optimizer === "QUBO" && (!state.quantumAvailable || !state.continuumQuboSupported)) {
    setRunMessage(
      !state.quantumAvailable
        ? "freeto.quantum is not available; QUBO jobs cannot run."
        : "This installed core does not support optimizer=\"QUBO\" yet.",
      true
    );
    return;
  }

  setRunMessage("Submitting job…", false);
  const resp = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    setRunMessage(await safeErrorDetail(resp), true);
    return;
  }
  const data = await resp.json();
  state.currentJobId = data.job_id;
  setRunning(true);
  resetChart();
  el.logOutput.textContent = "";
  clearQuboStatus();
  clearAuditUI();
  clearDesignMesh();
  state.finishedJobId = null;
  el.resultCard.hidden = true;
  setRunMessage(
    data.queue_position && data.queue_position > 1
      ? `Queued (position ${data.queue_position}).`
      : "Running…",
    false
  );
  startPolling(data.job_id);
}

async function stopJob() {
  if (!state.currentJobId) return;
  await fetch(`/api/jobs/${state.currentJobId}/stop`, { method: "POST" });
  setRunMessage("Stop requested…", false);
}

function setRunning(running) {
  state.running = running;
  el.runBtn.disabled = running;
  el.stopBtn.disabled = !running;
}

el.runBtn.addEventListener("click", runJob);
el.stopBtn.addEventListener("click", stopJob);

// ---------------------------------------------------------------------
// Polling job status
// ---------------------------------------------------------------------
let logCursor = 0;
let lastPreviewIter = -1;

function startPolling(jobId) {
  if (state.polling) clearInterval(state.polling);
  logCursor = 0;
  lastPreviewIter = -1;
  state.polling = setInterval(() => pollOnce(jobId), 1000);
  pollOnce(jobId);
}

async function pollOnce(jobId) {
  let resp;
  try {
    resp = await fetch(`/api/jobs/${jobId}?since=${logCursor}`);
  } catch {
    return;
  }
  if (!resp.ok) return;
  const s = await resp.json();

  if (s.log_lines && s.log_lines.length) {
    const atBottom = el.logOutput.scrollTop + el.logOutput.clientHeight >= el.logOutput.scrollHeight - 4;
    el.logOutput.textContent += (el.logOutput.textContent ? "\n" : "") + s.log_lines.join("\n");
    if (atBottom) el.logOutput.scrollTop = el.logOutput.scrollHeight;
  }
  logCursor = s.log_cursor;

  if (s.setup) {
    el.statusMesh.textContent = `mesh ${s.setup.nelx}×${s.setup.nely}×${s.setup.nelz}`;
    el.statusElements.textContent = `elements ${s.setup.nele?.toLocaleString?.() ?? s.setup.nele}`;
    el.statusDofs.textContent = `DOFs ${s.setup.ndof?.toLocaleString?.() ?? s.setup.ndof}`;
    el.statusSolver.textContent = `solver ${s.setup.solver}`;
  }
  if (s.last_iter_time != null) {
    el.statusItertime.textContent = `iter time ${s.last_iter_time.toFixed(3)}s`;
  }
  updateQuboStatus(s.history);
  updateAuditStatus(s);

  updateChart(s.history);

  if (s.status === "running" || s.status === "queued") {
    if (s.queue_position && s.queue_position > 1) {
      setRunMessage(`Queued (position ${s.queue_position}).`, false);
    } else if (s.last_iter) {
      setRunMessage(`Running… iteration ${s.last_iter}${s.max_iter ? " / " + s.max_iter : ""}`, false);
    }
    maybeRefreshPreview(jobId, s.last_iter);
    return;
  }

  // terminal state
  clearInterval(state.polling);
  state.polling = null;
  setRunning(false);

  if (s.status === "done") {
    const ri = s.result_info || {};
    const crispTxt = ri.crisp_compliance != null
      ? ` Crisp compliance ${Number(ri.crisp_compliance).toPrecision(4)}` +
        (ri.crisp_volfrac != null ? ` at V=${Number(ri.crisp_volfrac).toFixed(3)}` : "") + "."
      : "";
    setRunMessage("Done." + crispTxt, false);
    renderResult(s);
    el.downloadStlBtn.disabled = false;
    el.downloadNpzBtn.disabled = false;
    state.finishedJobId = jobId;
    await loadFinalDesign(jobId);
    await loadAudit(jobId, s);
    refreshAuditJobList();
  } else if (s.status === "stopped") {
    setRunMessage("Stopped.", false);
    if (s.has_result) {
      el.downloadStlBtn.disabled = false;
      el.downloadNpzBtn.disabled = false;
      state.finishedJobId = jobId;
      await loadFinalDesign(jobId);
      await loadAudit(jobId, s);
      refreshAuditJobList();
    }
  } else if (s.status === "error") {
    setRunMessage(`Error: ${s.error ? s.error.split("\n").slice(-2).join(" ") : "unknown error"}`, true);
    console.error(s.error);
  }
}

// Reads the last entry of `history.qubo` (per-iteration QUBO solver stats —
// see docs/QUANTUM_API.md's `iter` callback `qubo` dict, forwarded verbatim
// by the job's history) into the header status bar; hidden for OC/MMA runs.
function refreshQuboStatusVisibility() {
  // The QUBO solver stats belong to the Setup tab's continuum job only: hide
  // them on the Truss/Study tabs and whenever there is nothing to show.
  el.statusQubo.hidden = !(state.activeTab === "setup" && state.quboStatusText);
}

function clearQuboStatus() {
  state.quboStatusText = "";
  el.statusQubo.textContent = "";
  refreshQuboStatusVisibility();
}

function updateQuboStatus(history) {
  const list = history && history.qubo;
  if (!Array.isArray(list) || !list.length) {
    clearQuboStatus();
    return;
  }
  const q = list[list.length - 1];
  const bits = [];
  if (q.backend != null) bits.push(q.backend);
  if (q.n_free != null) bits.push(`n_free=${q.n_free}`);
  if (q.n_blocks != null) bits.push(`blocks=${q.n_blocks}`);
  if (q.solver_time != null) bits.push(`solver=${Number(q.solver_time).toFixed(3)}s`);
  if (q.qpu_time != null) bits.push(`qpu=${Number(q.qpu_time).toFixed(3)}s`);
  if (q.approx_ratio != null) bits.push(`ratio=${Number(q.approx_ratio).toFixed(3)}`);
  state.quboStatusText = `qubo: ${bits.join(" ")}`;
  el.statusQubo.textContent = state.quboStatusText;
  refreshQuboStatusVisibility();
}

async function maybeRefreshPreview(jobId, iter) {
  if (iter === lastPreviewIter) return;
  lastPreviewIter = iter;
  try {
    const resp = await fetch(`/api/jobs/${jobId}/preview.stl?iter=${iter}`, { cache: "no-store" });
    if (!resp.ok) return;
    const buf = await resp.arrayBuffer();
    setDesignGeometry(buf, false);
  } catch (err) {
    console.warn("preview refresh failed", err);
  }
}

// Result card: the quantities of the paper (native c and V, element proxy,
// refined binary voxel c and V as in Fig. 5) and, for a paper example, the
// refined gap to the paper's MMA reference and the published values of the
// matching paper run.
function renderResult(s) {
  const ri = s.result_info || {};
  const f4 = (v) => (v == null || !isFinite(v) ? "—" : Number(v).toPrecision(4));
  const f3 = (v) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(3));
  const rows = [
    ["native (optimizer's own)", ri.native_compliance, ri.native_volfrac],
    ["crisp element proxy", ri.crisp_compliance, ri.crisp_volfrac],
    [`refined binary voxel${ri.refined_f ? ` (f = ${ri.refined_f})` : ""}`, ri.refined_compliance, ri.refined_volfrac],
  ];
  el.resultBody.innerHTML = rows.map(([lab, c, v]) =>
    `<tr><td>${escHtml(lab)}</td><td class="num">${f4(c)}</td><td class="num">${f3(v)}</td></tr>`).join("");
  const bits = [];
  if (ri.iterations != null) bits.push(`${ri.iterations} iterations`);
  const pp = s.paper || null;
  if (ri.gap_refined != null) {
    bits.push(`Refined gap to the paper's MMA reference (c = ${f4(ri.gap_reference)}): ` +
      `${(100 * ri.gap_refined >= 0 ? "+" : "")}${(100 * ri.gap_refined).toFixed(2)} %`);
  } else if (pp && pp.example) {
    bits.push(pp.same_problem ? "no refined gap (refined evaluation f = 2 is off)"
      : `not the paper's problem (${(pp.problem_differences || []).join(", ")} differ), no gap to the paper`);
  }
  if (pp && pp.method && pp.record) {
    const r = pp.record;
    bits.push(`same settings as the paper run "${pp.method}", seed ${pp.seed}; paper values: ` +
      `${r.iterations} iterations, native c = ${f4(r.compliance)}, refined c = ${f4(r.refined_compliance)}, ` +
      `V = ${f3(r.refined_volfrac)}`);
  } else if (pp && pp.example && pp.same_problem) {
    bits.push("settings differ from every paper run of this example (no published values to compare)");
  }
  el.resultPaper.textContent = bits.map((b) => b.charAt(0).toUpperCase() + b.slice(1)).join(". ") +
    (bits.length ? "." : "");
  el.resultCard.hidden = false;
}

async function loadFinalDesign(jobId) {
  try {
    let resp = null;
    if (el.designView.value === "evaluated") {
      resp = await fetch(`/api/jobs/${jobId}/evaluated.stl`, { cache: "no-store" });
    }
    if (!resp || !resp.ok) resp = await fetch(`/api/jobs/${jobId}/result.stl`, { cache: "no-store" });
    if (!resp.ok) return;
    const buf = await resp.arrayBuffer();
    setDesignGeometry(buf, true);
  } catch (err) {
    console.warn("final design load failed", err);
  }
}

function disposeGroupChildren(group) {
  // Group.clear() only detaches children from the scene graph — it does
  // NOT free their GPU-side geometry/material buffers, which leaks memory
  // every time a new preview mesh replaces the old one (a real cost: each
  // preview at a fine mesh_control can be tens of MB). Dispose explicitly
  // before clearing.
  for (const obj of group.children) {
    obj.geometry?.dispose();
    obj.material?.dispose();
  }
  group.clear();
}

function clearDesignMesh() {
  disposeGroupChildren(designGroup);
  state.designMesh = null;
}

function setDesignGeometry(arrayBuffer, isFinal) {
  const geom = stlLoader.parse(arrayBuffer);
  geom.computeVertexNormals();
  disposeGroupChildren(designGroup);
  const material = new THREE.MeshStandardMaterial({
    color: isFinal ? VIEW_COLORS.design : VIEW_COLORS.designRunning,
    metalness: 0.08,
    roughness: isFinal ? 0.42 : 0.6,
  });
  const mesh = new THREE.Mesh(geom, material);
  mesh.visible = el.toggleDesign.checked;
  designGroup.add(mesh);
  state.designMesh = mesh;
  if (isFinal) fitView();
}

el.toggleInputs.addEventListener("change", () => {
  for (const rec of state.files.values()) updateMeshVisibility(rec);
});
el.toggleDesign.addEventListener("change", () => {
  if (state.designMesh) state.designMesh.visible = el.toggleDesign.checked;
});
el.toggleAxes.addEventListener("change", () => { axesHelper.visible = el.toggleAxes.checked; });
el.designView.addEventListener("change", () => {
  if (state.finishedJobId && !state.running) loadFinalDesign(state.finishedJobId);
});
el.fitViewBtn.addEventListener("click", fitView);

el.downloadStlBtn.addEventListener("click", () => {
  if (state.currentJobId) window.location.href = `/api/jobs/${state.currentJobId}/result.stl`;
});
el.downloadNpzBtn.addEventListener("click", () => {
  if (state.currentJobId) window.location.href = `/api/jobs/${state.currentJobId}/result.npz`;
});

// ---------------------------------------------------------------------
// Physics check (docs/AUDIT_API.md): status bar, card, "audit any result"
// ---------------------------------------------------------------------
const auditState = { jobId: null, label: "", data: null };
const escHtml = (t) => String(t ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmtPct = (f, nd = 1) => (f == null || !isFinite(f) ? "—" : `${(100 * Number(f)).toFixed(nd)} %`);

function setStatusAudit(text, cls) {
  const box = $("status-audit");
  box.textContent = text || "";
  box.classList.remove("ok", "fail");
  if (cls) box.classList.add(cls);
  box.hidden = !text || state.activeTab !== "setup";
  box.dataset.text = text || "";
}

// Live "components: N, floating: x %" (QUBO runs), then the final verdict.
function updateAuditStatus(s) {
  if (s.audit_ok != null && s.n_components != null) {
    setStatusAudit(
      `physics ${s.audit_ok ? "PASS" : "FAIL"} · components: ${s.n_components}, floating: ${fmtPct(s.floating_frac)}`,
      s.audit_ok ? "ok" : "fail"
    );
    return;
  }
  const live = s.audit_live;
  if (live && live.n_components != null) {
    const bad = live.n_components > 1 || (live.floating_frac || 0) > 0.01;
    setStatusAudit(`components: ${live.n_components}, floating: ${fmtPct(live.floating_frac)}`, bad ? "fail" : "ok");
  } else if (s.status === "running" || s.status === "queued") {
    setStatusAudit("");
  }
}

function clearAuditUI() {
  auditState.jobId = null;
  auditState.data = null;
  $("audit-card").hidden = true;
  $("audit-img").removeAttribute("src");
  setStatusAudit("");
}

function fmtAuditValue(v, depth = 0) {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : Number(v.toPrecision(4)).toString();
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    const items = v.slice(0, 6).map((x) => fmtAuditValue(x, depth + 1));
    return `[${items.join(", ")}${v.length > 6 ? ", …" : ""}]`;
  }
  if (typeof v === "object") {
    const parts = Object.entries(v).map(([k, x]) => `${k}: ${fmtAuditValue(x, depth + 1)}`);
    return depth ? `{${parts.join(", ")}}` : parts.join("; ");
  }
  return String(v);
}

function renderAuditCard(jobId, label, a) {
  auditState.jobId = jobId;
  auditState.label = label || "";
  auditState.data = a;
  const card = $("audit-card");
  card.hidden = false;
  const pill = $("audit-pill");
  pill.textContent = a.ok ? "PASS" : "FAIL";
  pill.className = `pill pill-big ${a.ok ? "pill-ok" : "pill-bad"}`;
  const bits = [];
  if (label) bits.push(escHtml(label));
  if (a.n_components != null) bits.push(`components: ${a.n_components}`);
  if (a.floating_frac != null) bits.push(`floating: ${fmtPct(a.floating_frac, 2)}`);
  if (a.volume_fraction != null) bits.push(`V: ${Number(a.volume_fraction).toFixed(3)}`);
  if (a.native_compliance != null) bits.push(`native c: ${Number(a.native_compliance).toPrecision(4)}`);
  if (a.crisp_compliance != null) bits.push(`crisp c: ${Number(a.crisp_compliance).toPrecision(4)}`);
  $("audit-summary").innerHTML = bits.join(" · ");
  if (!a.ok) {
    $("audit-summary").innerHTML += (bits.length ? " · " : "") + "<b>physically invalid</b>";
  }

  const body = $("audit-checks-body");
  body.innerHTML = "";
  const checks = a.checks || {};
  for (const [name, c] of Object.entries(checks)) {
    const ok = c && c.pass;
    // two rows per check: name | pass | value, then the full-width detail sentence
    const full = fmtAuditValue(c ? c.value : null);
    const short = full.length > 90 ? `${full.slice(0, 88)}…` : full;
    const tr = document.createElement("tr");
    tr.className = `audit-main${ok ? "" : " fail"}`;
    tr.innerHTML =
      `<td class="audit-name">${escHtml(name)}</td>` +
      `<td><span class="pill ${ok ? "pill-ok" : "pill-bad"}">${ok ? "PASS" : "FAIL"}</span></td>` +
      `<td class="audit-value" title="${escHtml(full)}">${escHtml(short)}</td>`;
    body.appendChild(tr);
    if (c && c.detail) {
      const tr2 = document.createElement("tr");
      tr2.className = `audit-sub${ok ? "" : " fail"}`;
      tr2.innerHTML = `<td colspan="3" class="audit-detail">${escHtml(c.detail)}</td>`;
      body.appendChild(tr2);
    }
  }
  if (!Object.keys(checks).length) body.innerHTML = '<tr><td colspan="3" class="muted">no checks reported</td></tr>';

  const comps = Array.isArray(a.components) ? a.components : [];
  $("audit-components-count").textContent = `(${comps.length})`;
  const cb = $("audit-components-body");
  cb.innerHTML = "";
  const fmtBox = (b) => {
    if (!Array.isArray(b) || b.length !== 2) return "—";
    const r = (p) => p.map((x) => Math.round(x)).join(", ");
    return `(${r(b[0])}) – (${r(b[1])})`;
  };
  for (const c of comps) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${escHtml(c.label)}</td><td>${escHtml(c.n_elements)}</td>` +
      `<td>${c.grounded ? "yes" : '<span class="pill pill-bad">floating</span>'}</td>` +
      `<td>${c.has_load ? "yes" : "no"}</td><td class="audit-detail">${escHtml(fmtBox(c.bbox_mm))}</td>`;
    cb.appendChild(tr);
  }
  $("audit-components-details").open = comps.length > 1 || !a.ok;

  // overlay PNG (rendered lazily by the server on first request)
  const img = $("audit-img");
  const link = $("audit-img-link");
  const msg = $("audit-img-msg");
  const url = `/api/jobs/${jobId}/audit.png`;
  msg.textContent = "Rendering overlay figure…";
  link.hidden = true;
  img.onload = () => { msg.textContent = "Click the figure to enlarge."; link.hidden = false; };
  img.onerror = async () => {
    link.hidden = true;
    let detail = "";
    try { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) detail = await safeErrorDetail(r); } catch { /* ignore */ }
    msg.textContent = `Overlay figure unavailable${detail ? `: ${detail}` : "."}`;
  };
  img.src = `${url}?t=${Date.now()}`;
  const dl = $("audit-download-png");
  dl.href = url;
  dl.hidden = false;
}

async function loadAudit(jobId, status) {
  try {
    const resp = await fetch(`/api/jobs/${jobId}/audit`, { cache: "no-store" });
    if (!resp.ok) {
      $("audit-card").hidden = true;
      if (resp.status === 409) {
        setRunMessage($("run-message").textContent + " (physics check was off - use 'Audit any result' below.)", false);
      } else if (status && status.audit_error) {
        console.warn("audit unavailable:", status.audit_error);
      }
      return false;
    }
    const a = await resp.json();
    renderAuditCard(jobId, status ? status.label : "", a);
    updateAuditStatus({
      audit_ok: a.ok, n_components: a.n_components, floating_frac: a.floating_frac,
    });
    return true;
  } catch (err) {
    console.warn("audit load failed", err);
    return false;
  }
}

async function runAuditFor(jobId, label, msgBox, btns) {
  btns.forEach((b) => (b.disabled = true));
  msgBox.textContent = "Auditing…";
  msgBox.classList.remove("error");
  try {
    const resp = await fetch(`/api/jobs/${jobId}/audit/run`, { method: "POST" });
    if (!resp.ok) {
      msgBox.textContent = await safeErrorDetail(resp);
      msgBox.classList.add("error");
      return;
    }
    const a = await resp.json();
    renderAuditCard(jobId, label, a);
    updateAuditStatus({ audit_ok: a.ok, n_components: a.n_components, floating_frac: a.floating_frac });
    msgBox.textContent = `Audit ${a.ok ? "PASS" : "FAIL"} - see the Physics check card above.`;
    $("audit-card").scrollIntoView({ block: "nearest", behavior: "smooth" });
    refreshAuditJobList();
  } catch (err) {
    msgBox.textContent = `Request failed: ${err}`;
    msgBox.classList.add("error");
  } finally {
    btns.forEach((b) => (b.disabled = false));
  }
}

async function refreshAuditJobList() {
  const sel = $("audit-job-select");
  if (!sel) return;
  let jobs;
  try {
    const resp = await fetch("/api/jobs", { cache: "no-store" });
    if (!resp.ok) return;
    jobs = await resp.json();
  } catch { return; }
  const prev = sel.value;
  sel.innerHTML = '<option value="">Select a finished job…</option>';
  for (const j of jobs) {
    if (j.kind !== "continuum" || !j.has_result || !["done", "stopped"].includes(j.status)) continue;
    const opt = document.createElement("option");
    opt.value = j.id;
    opt.dataset.label = j.label || "";
    const verdict = j.audit_ok == null ? "not audited" : j.audit_ok ? "PASS" : "FAIL";
    const when = j.finished_at ? new Date(j.finished_at * 1000).toLocaleTimeString() : "";
    opt.textContent = `${j.label || "job"} · ${j.id} · ${when} · ${verdict}`;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

$("audit-job-refresh-btn").addEventListener("click", refreshAuditJobList);
$("audit-job-select").addEventListener("focus", refreshAuditJobList);
$("audit-job-run-btn").addEventListener("click", () => {
  const sel = $("audit-job-select");
  if (!sel.value) { $("audit-any-message").textContent = "Select a finished job first."; $("audit-any-message").classList.add("error"); return; }
  runAuditFor(sel.value, sel.selectedOptions[0]?.dataset.label || "", $("audit-any-message"),
    [$("audit-job-run-btn"), $("audit-rerun-btn")]);
});
$("audit-rerun-btn").addEventListener("click", () => {
  if (!auditState.jobId) return;
  runAuditFor(auditState.jobId, auditState.label, $("audit-any-message"), [$("audit-job-run-btn"), $("audit-rerun-btn")]);
});
$("audit-download-json-btn").addEventListener("click", () => {
  if (!auditState.data) return;
  const blob = new Blob([JSON.stringify(auditState.data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `audit_${auditState.jobId}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

function openLightbox(src, alt) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = alt || "";
  $("lightbox").hidden = false;
}
function closeLightbox() { $("lightbox").hidden = true; $("lightbox-img").removeAttribute("src"); }
$("audit-img-link").addEventListener("click", (e) => { e.preventDefault(); openLightbox($("audit-img").src, "Physics check overlay"); });
$("lightbox").addEventListener("click", (e) => { if (e.target.id !== "lightbox-img") closeLightbox(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("lightbox").hidden) closeLightbox(); });

// ---------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------
// nitipong.com palette: ink compliance line, dashed muted-grey volume
// fraction, hairline grid. CHART_COLORS_DARK are the inverted variants for the
// dark theme. Further series (if ever added) cycle through CHART_SERIES.
const CHART_COLORS = { compliance: "#141414", volfrac: "#8a8a8a", grid: "rgba(0,0,0,0.08)" };
const CHART_COLORS_DARK = { compliance: "#ededeb", volfrac: "#a3a3a0", grid: "rgba(255,255,255,0.10)" };
const CHART_SERIES = ["#141414", "#6b6b6b", "#DE5C8E", "#a3a3a0", "#3a3a3a"];
const CHART_FONT_SANS = '"Inter", system-ui, -apple-system, "Helvetica Neue", sans-serif';
const CHART_FONT_MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
const CHART_VOLFRAC_DASH = [5, 4];
let chart;
function initChart() {
  Chart.defaults.font.family = CHART_FONT_SANS;
  Chart.defaults.font.size = 11;
  const axisTitleFont = { family: CHART_FONT_MONO, size: 10 };
  chart = new Chart(el.chartCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Compliance",
          data: [],
          borderColor: CHART_COLORS.compliance,
          backgroundColor: "transparent",
          yAxisID: "y",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.15,
        },
        {
          label: "Volume fraction",
          data: [],
          borderColor: CHART_COLORS.volfrac,
          backgroundColor: "transparent",
          yAxisID: "y1",
          pointRadius: 0,
          borderWidth: 1.5,
          borderDash: CHART_VOLFRAC_DASH,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "ITERATION", font: axisTitleFont }, ticks: { maxRotation: 0, autoSkipPadding: 14 }, grid: {} },
        // Compliance/volume-fraction lines keep their fixed brand colors
        // (blue/orange, chosen to read fine on both light and dark
        // backgrounds) — only the theme-neutral text (axis titles/ticks,
        // legend) needs to track the active theme; see applyChartTheme().
        y: { position: "left", grid: {}, title: { display: true, text: "COMPLIANCE", font: axisTitleFont }, ticks: { color: CHART_COLORS.compliance } },
        y1: {
          position: "right",
          title: { display: true, text: "VOLUME FRACTION", font: axisTitleFont },
          grid: { drawOnChartArea: false, color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.volfrac },
        },
      },
      plugins: { legend: { labels: { boxWidth: 18, boxHeight: 1, font: { family: CHART_FONT_MONO, size: 10 } } } },
    },
  });
  applyChartTheme();
}

// Reads the current theme's CSS custom properties so the chart's neutral
// text (axis titles/ticks, legend) stays readable in both light and dark
// mode, instead of a color hard-coded for one theme (e.g. a light grey
// legend that disappears on a white card in light mode).
function currentThemeTextColors() {
  const styles = getComputedStyle(document.documentElement);
  return {
    text: styles.getPropertyValue("--text").trim() || "#141414",
    muted: styles.getPropertyValue("--text-muted").trim() || "#6b6b6b",
  };
}

function applyChartTheme() {
  if (!chart) return;
  const c = currentThemeTextColors();
  chart.options.scales.x.ticks.color = c.muted;
  chart.options.scales.x.title.color = c.muted;
  chart.options.scales.y.title.color = c.muted;
  chart.options.scales.y1.title.color = c.muted;
  chart.options.plugins.legend.labels.color = c.text;
  const dark = effectiveTheme() === "dark";
  const pal = dark ? CHART_COLORS_DARK : CHART_COLORS;
  chart.data.datasets[0].borderColor = pal.compliance;
  chart.data.datasets[1].borderColor = pal.volfrac;
  chart.options.scales.y.ticks.color = pal.compliance;
  // the dashed grey line is lighter than AA text allows, so its tick labels use --text-muted
  chart.options.scales.y1.ticks.color = c.muted;
  chart.options.scales.x.grid.color = pal.grid;
  chart.options.scales.y.grid.color = pal.grid;
  chart.options.scales.y1.grid.color = pal.grid;
  chart.update("none");
}

function resetChart() {
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.data.datasets[1].data = [];
  chart.update();
}

function updateChart(history) {
  if (!history || !history.compliance) return;
  const n = history.compliance.length;
  chart.data.labels = Array.from({ length: n }, (_, i) => i + 1);
  chart.data.datasets[0].data = history.compliance;
  chart.data.datasets[1].data = history.volfrac;
  chart.update("none");
}

// ---------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------
// The *visually active* theme: an explicit data-theme attribute if one was
// ever set, otherwise whatever prefers-color-scheme is currently rendering
// (the CSS falls back to it when the attribute is absent). Deriving "current"
// from the attribute alone made the toggle's first click a no-op whenever
// the OS was already in dark mode (attribute unset -> null -> "next" always
// computed as "dark", i.e. no visible change).
function effectiveTheme() {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

function initTheme() {
  let theme = null;
  try { theme = localStorage.getItem("freeto_theme"); } catch {}
  if (theme) document.documentElement.setAttribute("data-theme", theme);
  el.themeToggle.addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("freeto_theme", next); } catch {}
    applyChartTheme();
  });
}

// ---------------------------------------------------------------------
// Health / core badge
// ---------------------------------------------------------------------
async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const core = data.core || {};
    const src = core.source || "unknown";
    el.coreBadge.textContent = `core: ${src}`;
    el.coreBadge.classList.add(src === "real" ? "real" : "stub");
    if (src === "stub") {
      el.coreBadge.title = "FREETO_WEB_STUB=1 is set: intentionally using the fake shrinking-sphere solver, not the real optimizer.";
    }
    if (core.usable === false) {
      el.coreBadge.title = core.import_error || "The real core is unavailable.";
      if (el.coreBanner) {
        el.coreBanner.hidden = false;
        el.coreBanner.textContent =
          `Real FreeTO core unavailable — jobs are disabled (no fallback to a fake solver). ` +
          (core.import_error ? `Import error: ${core.import_error}` : "");
      }
      el.runBtn.disabled = true;
      el.runBtn.title = "The real FreeTO core is unavailable; see the banner above.";
    }
    // freeto.study is feature-detected independently of the continuum core
    // (see webapp/core_loader.py); show a plain notice on the Study tab
    // instead of a dead "Run study" button when it's missing.
    const study = data.study || {};
    const studyCard = $("study-unavailable-card");
    if (studyCard) {
      studyCard.hidden = !!study.usable;
      if (!study.usable) {
        $("study-unavailable-msg").textContent =
          `The 'freeto.study' module is not available, so study suites cannot run. ${study.import_error || ""}`;
        const runBtn = $("study-run-btn");
        if (runBtn) runBtn.disabled = true;
      }
    }
  } catch {
    el.coreBadge.textContent = "core: unreachable";
  }
}

// ---------------------------------------------------------------------
// Tabs (Setup / Truss / Study)
// ---------------------------------------------------------------------
// Drag handle on the right edge of every side panel (Setup / Truss / Study).
// One shared width (CSS var --side-w on <html>), remembered in localStorage.
function initPanelResizers() {
  const root = document.documentElement;
  const MIN = 340;
  const maxW = () => Math.floor(window.innerWidth * 0.62);
  try {
    const saved = parseInt(localStorage.getItem("freeto.sideWidth") || "", 10);
    if (saved >= MIN) root.style.setProperty("--side-w", `${Math.min(saved, maxW())}px`);
  } catch { /* storage unavailable: keep default */ }
  document.querySelectorAll("#left-panel, .side-panel").forEach((panel) => {
    const h = document.createElement("div");
    h.className = "panel-resizer";
    h.title = "Drag to resize the side panel (double-click to reset)";
    panel.appendChild(h);
    h.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      h.setPointerCapture(ev.pointerId);
      h.classList.add("dragging");
      const left = panel.getBoundingClientRect().left;
      const move = (e) => {
        const w = Math.max(MIN, Math.min(maxW(), e.clientX - left));
        root.style.setProperty("--side-w", `${w}px`);
        window.dispatchEvent(new Event("resize"));
      };
      const up = () => {
        h.classList.remove("dragging");
        h.removeEventListener("pointermove", move);
        h.removeEventListener("pointerup", up);
        try {
          localStorage.setItem("freeto.sideWidth", String(parseInt(getComputedStyle(panel).width, 10)));
        } catch { /* ignore */ }
      };
      h.addEventListener("pointermove", move);
      h.addEventListener("pointerup", up);
    });
    h.addEventListener("dblclick", () => {
      root.style.removeProperty("--side-w");
      try { localStorage.removeItem("freeto.sideWidth"); } catch { /* ignore */ }
      window.dispatchEvent(new Event("resize"));
    });
  });
}

function initTabs() {
  initPanelResizers();
  const btns = document.querySelectorAll(".tab-btn");
  const views = document.querySelectorAll(".tab-view");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.toggle("active", b === btn));
      views.forEach((v) => v.classList.toggle("active", v.dataset.tabView === btn.dataset.tab));
      state.activeTab = btn.dataset.tab;
      refreshQuboStatusVisibility();
      $("status-audit").hidden = !($("status-audit").dataset.text && state.activeTab === "setup");
      if (btn.dataset.tab === "study") refreshPastStudies();
      if (btn.dataset.tab === "setup") refreshAuditJobList();
      if (btn.dataset.tab === "truss") {
        // The canvas has zero layout size while its tab is hidden
        // (display:none) — redraw once it's actually visible and measurable.
        setTimeout(() => drawTruss(state.truss.currentProblem, state.truss.currentResult), 30);
      }
    });
  });
}

// ---------------------------------------------------------------------
// Truss tab
// ---------------------------------------------------------------------
async function fetchTrussBenchmarks() {
  let data;
  try {
    const resp = await fetch("/api/truss/benchmarks");
    data = await resp.json();
  } catch {
    data = { truss_available: false, import_error: "request to /api/truss/benchmarks failed", benchmarks: [], methods: [] };
  }
  state.truss.benchmarks = data.benchmarks || [];
  state.truss.methods = data.methods || [];

  const unavailable = !data.truss_available;
  const card = $("truss-unavailable-card");
  if (card) {
    card.hidden = !unavailable;
    if (unavailable) {
      $("truss-unavailable-msg").textContent =
        `The 'freeto.truss' module is not available, so truss benchmarks cannot run. ${data.import_error || ""}`;
    }
  }
  $("truss-run-btn").disabled = unavailable;

  const sel = $("truss-benchmark-select");
  sel.innerHTML = '<option value="">Choose a benchmark…</option>';
  for (const b of state.truss.benchmarks) {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = `${b.alias ? b.alias + " — " : ""}${b.title}`;
    sel.appendChild(opt);
  }
}

function updateTrussMethodAvailability(bench) {
  const methodSel = $("truss-method-select");
  const exactOpt = methodSel.querySelector('option[value="exact"]');
  if (exactOpt) {
    const ok = !!(bench && bench.exact_available);
    exactOpt.disabled = !ok;
    exactOpt.title = ok ? "" : "Not available for this benchmark (too many bars for enumeration).";
    if (!ok && methodSel.value === "exact") methodSel.value = "oc_round";
  }
  onTrussMethodChange();
}

function onTrussMethodChange() {
  const method = $("truss-method-select").value;
  const needsBackend = method === "qubo" || method === "oc_qubo";
  $("truss-backend-row").hidden = !needsBackend;
  if (needsBackend) populateBackendSelect($("truss-backend-select"), state.quantumBackends, { includeAuto: true });
}
$("truss-method-select").addEventListener("change", onTrussMethodChange);

async function onTrussBenchmarkChange() {
  const id = $("truss-benchmark-select").value;
  state.truss.currentResult = null;
  renderTrussStats(null);
  if (!id) {
    state.truss.currentBenchmark = null;
    state.truss.currentProblem = null;
    $("truss-benchmark-desc").textContent = "";
    drawTruss(null, null);
    return;
  }
  const bench = state.truss.benchmarks.find((b) => b.id === id);
  state.truss.currentBenchmark = bench;
  $("truss-benchmark-desc").textContent = bench ? bench.description || "" : "";
  updateTrussMethodAvailability(bench);
  try {
    const resp = await fetch(`/api/truss/benchmarks/${id}`);
    if (!resp.ok) { setTrussMessage(await safeErrorDetail(resp), true); return; }
    const data = await resp.json();
    state.truss.currentProblem = data.problem;
    drawTruss(data.problem, null);
  } catch (err) {
    console.warn("Failed to load truss benchmark detail", err);
  }
}
$("truss-benchmark-select").addEventListener("change", onTrussBenchmarkChange);

// Simple orthographic/cavalier projection for the (rare) 3-D truss
// benchmarks (T3 tower, T5 column) so the same 2-D canvas renderer handles
// both dims without a second three.js scene.
function projectTrussNode(node, dim) {
  if (dim === 3) {
    const [x, y, z] = node;
    return [x + 0.5 * z, y - 0.35 * z];
  }
  return [node[0], node[1]];
}

// Truss canvas colours (light viewport): faint grey candidates, ink members,
// BC-blue supports and load-orange arrows (same hues as the 3D legend).
const TRUSS_COLORS = {
  muted: "#6b6b6b",
  candidate: "#8a8a86",
  node: "#3a3a3a",
  selected: "#141414",
  support: "#3d63c4",
  load: "#e07b39",
};

function drawTruss(problem, result) {
  const canvas = $("truss-canvas");
  if (!canvas) return;
  const container = canvas.parentElement;
  const rect = container.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return; // tab hidden / not laid out yet
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!problem || !problem.nodes || !problem.nodes.length) {
    ctx.fillStyle = TRUSS_COLORS.muted; // truss canvas sits on the always-light viewport
    ctx.font = `13px ${CHART_FONT_SANS}`;
    ctx.textAlign = "center";
    ctx.fillText("Choose a benchmark to preview its ground structure.", rect.width / 2, rect.height / 2);
    ctx.textAlign = "start";
    return;
  }

  const dim = problem.dim || 2;
  const nodes = problem.nodes;
  const bars = problem.bars || [];
  const supports = problem.supports || [];
  const loads = problem.loads || [];

  const proj = nodes.map((n) => projectTrussNode(n, dim));
  const xs = proj.map((p) => p[0]);
  const ys = proj.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const pad = 42;
  const scale = Math.max(
    Math.min((rect.width - 2 * pad) / spanX, (rect.height - 2 * pad) / spanY),
    1e-6
  );
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const toCanvas = ([x, y]) => [
    rect.width / 2 + (x - cx) * scale,
    rect.height / 2 - (y - cy) * scale, // flip y: structural "up" = canvas "up"
  ];
  const pts = proj.map(toCanvas);

  // The truss canvas is drawn on the always-light viewport (--viewport-bg),
  // so these are fixed ink/grey tones, independent of the page theme.
  const mutedColor = TRUSS_COLORS.candidate;
  const textColor = TRUSS_COLORS.node;

  // 1) candidate bars, faint
  ctx.lineCap = "round";
  ctx.strokeStyle = mutedColor;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  for (const bar of bars) {
    const [a, b] = bar;
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // 2) selected/"on" bars, thick — scaled by relative area for multi-level results
  if (result && Array.isArray(result.on)) {
    const areas = Array.isArray(result.areas) ? result.areas : null;
    const maxArea = areas && areas.length ? Math.max(...areas, 1e-9) : 1;
    bars.forEach((bar, i) => {
      if (!result.on[i]) return;
      const [a, b] = bar;
      const areaFrac = areas ? Math.max(areas[i] / maxArea, 0.15) : 1;
      ctx.strokeStyle = TRUSS_COLORS.selected;
      ctx.lineWidth = 2 + areaFrac * 6;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    });
  }

  // 3) nodes
  ctx.fillStyle = textColor;
  pts.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // 4) supports: a small triangle marker under each supported node
  const supportedNodes = new Set(supports.map((s) => s[0]));
  ctx.fillStyle = TRUSS_COLORS.support;
  supportedNodes.forEach((ni) => {
    const p = pts[ni];
    if (!p) return;
    const [x, y] = p;
    ctx.beginPath();
    ctx.moveTo(x, y + 6);
    ctx.lineTo(x - 7, y + 18);
    ctx.lineTo(x + 7, y + 18);
    ctx.closePath();
    ctx.fill();
  });

  // 5) loads: arrows in the (projected) force direction
  ctx.strokeStyle = TRUSS_COLORS.load;
  ctx.fillStyle = TRUSS_COLORS.load;
  ctx.lineWidth = 2;
  const arrowLen = 28;
  for (const load of loads) {
    const ni = load[0];
    const p = pts[ni];
    if (!p) continue;
    const fx = load[1] ?? 0, fy = load[2] ?? 0, fz = dim === 3 ? (load[3] ?? 0) : 0;
    if (Math.hypot(fx, fy, fz) < 1e-12) continue;
    const dir2 = dim === 3 ? [fx + 0.5 * fz, fy - 0.35 * fz] : [fx, fy];
    const dmag = Math.hypot(dir2[0], dir2[1]) || 1;
    const ux = dir2[0] / dmag, uy = -dir2[1] / dmag; // canvas y flip
    const [x0, y0] = p;
    const x1 = x0 + ux * arrowLen, y1 = y0 + uy * arrowLen;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    const ah = 6;
    const angle = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - ah * Math.cos(angle - Math.PI / 6), y1 - ah * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x1 - ah * Math.cos(angle + Math.PI / 6), y1 - ah * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }
}
window.addEventListener("resize", () => {
  if (state.truss.currentProblem) drawTruss(state.truss.currentProblem, state.truss.currentResult);
});

function renderTrussStats(result) {
  const box = $("truss-stats");
  if (!box) return;
  if (!result) { box.innerHTML = ""; return; }
  const pct = (v) => (v == null ? "—" : `${(Number(v) * 100).toFixed(2)}%`);
  const num = (v, digits = 5) => (v == null ? "—" : Number(v).toPrecision(digits));
  const wall = result.timing && result.timing.wall != null ? Number(result.timing.wall).toFixed(2) : "—";
  const rows = [
    ["Compliance", num(result.compliance)],
    ["Volume fraction", result.volume_fraction != null ? Number(result.volume_fraction).toFixed(3) : "—"],
    ["Gap to exact", result.gap != null ? pct(result.gap) : "—"],
    ["Feasible", result.feasible === true ? "yes" : result.feasible === false ? "no" : "—"],
    ["Time (s)", wall],
  ];
  box.innerHTML = rows.map(([k, v]) => `<span><b>${k}:</b> ${v}</span>`).join("");
}

function addTrussRunRow(benchLabel, method, backend, result) {
  state.truss.runs.push({ benchLabel, method, backend: backend || "—", result });
  renderTrussResultsTable();
}

function renderTrussResultsTable() {
  const body = $("truss-results-body");
  if (!body) return;
  body.innerHTML = "";
  if (!state.truss.runs.length) {
    body.innerHTML = '<tr><td colspan="8" class="muted small">No runs yet.</td></tr>';
    return;
  }
  for (const row of state.truss.runs) {
    const r = row.result || {};
    const tr = document.createElement("tr");
    const feasible = r.feasible === true
      ? '<span class="pill pill-ok">yes</span>'
      : r.feasible === false
        ? '<span class="pill pill-bad">no</span>'
        : "—";
    const wall = r.timing && r.timing.wall != null ? Number(r.timing.wall).toFixed(2) : "—";
    tr.innerHTML = `
      <td>${row.benchLabel}</td>
      <td>${row.method}</td>
      <td>${row.backend}</td>
      <td>${r.compliance != null ? Number(r.compliance).toPrecision(5) : "—"}</td>
      <td>${r.volume_fraction != null ? Number(r.volume_fraction).toFixed(3) : "—"}</td>
      <td>${r.gap != null ? (Number(r.gap) * 100).toFixed(2) + "%" : "—"}</td>
      <td>${feasible}</td>
      <td>${wall}</td>`;
    body.appendChild(tr);
  }
}

function setTrussMessage(msg, isError) {
  const box = $("truss-run-message");
  box.textContent = msg;
  box.classList.toggle("error", !!isError);
  box.classList.toggle("ok", !isError && !!msg);
}
function setTrussRunning(running) {
  $("truss-run-btn").disabled = running;
  $("truss-stop-btn").disabled = !running;
}

async function runTruss() {
  const benchId = $("truss-benchmark-select").value;
  if (!benchId) { setTrussMessage("Choose a benchmark first.", true); return; }
  const method = $("truss-method-select").value;
  const needsBackend = method === "qubo" || method === "oc_qubo";
  const backend = needsBackend ? $("truss-backend-select").value : null;
  const seedVal = $("truss-seed").value;
  const payload = {
    benchmark_id: benchId,
    method,
    backend: backend || null,
    seed: seedVal === "" ? null : parseInt(seedVal, 10),
    options: {},
  };
  setTrussMessage("Submitting…", false);
  let resp;
  try {
    resp = await fetch("/api/truss/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    setTrussMessage(`Request failed: ${err}`, true);
    return;
  }
  if (!resp.ok) { setTrussMessage(await safeErrorDetail(resp), true); return; }
  const data = await resp.json();
  state.truss.jobId = data.job_id;
  setTrussRunning(true);
  setTrussMessage(
    data.queue_position && data.queue_position > 1 ? `Queued (position ${data.queue_position}).` : "Running…",
    false
  );
  startTrussPolling(data.job_id);
}

async function stopTruss() {
  if (!state.truss.jobId) return;
  await fetch(`/api/jobs/${state.truss.jobId}/stop`, { method: "POST" });
  setTrussMessage("Stop requested…", false);
}

function startTrussPolling(jobId) {
  if (state.truss.polling) clearInterval(state.truss.polling);
  state.truss.polling = setInterval(() => pollTrussOnce(jobId), 1000);
  pollTrussOnce(jobId);
}

async function pollTrussOnce(jobId) {
  let resp;
  try {
    resp = await fetch(`/api/jobs/${jobId}`);
  } catch {
    return;
  }
  if (!resp.ok) return;
  const s = await resp.json();
  if (s.status === "running" || s.status === "queued") {
    setTrussMessage(
      s.queue_position && s.queue_position > 1
        ? `Queued (position ${s.queue_position}).`
        : `Running… ${s.stage ? "(" + s.stage + ")" : ""}`,
      false
    );
    return;
  }
  clearInterval(state.truss.polling);
  state.truss.polling = null;
  setTrussRunning(false);
  if (s.status === "done" || s.status === "stopped") {
    setTrussMessage(s.status === "done" ? "Done." : "Stopped.", false);
    try {
      const rresp = await fetch(`/api/jobs/${jobId}/truss_result`);
      if (rresp.ok) {
        const result = await rresp.json();
        // Label from what the job was *submitted* with (server status.spec),
        // not from the selects, which the user may have changed during the run.
        const spec = s.spec || {};
        const benchId = spec.benchmark_id || (s.setup && s.setup.benchmark_id) || null;
        // Only paint the canvas when it still shows the job's benchmark (a
        // result drawn on another benchmark's geometry would be nonsense);
        // the row in the results table is added either way.
        if (!benchId || $("truss-benchmark-select").value === benchId) {
          state.truss.currentResult = result;
          drawTruss(state.truss.currentProblem, result);
          renderTrussStats(result);
        } else {
          setTrussMessage(`Done (${benchId}); canvas shows another benchmark — result is in the table.`, false);
        }
        const bench = state.truss.benchmarks.find((b) => b.id === benchId);
        const method = spec.method || result.method || "?";
        const backend = spec.backend || null;
        addTrussRunRow(bench ? (bench.alias || bench.title) : (benchId || "?"), method, backend, result);
      }
    } catch (err) {
      console.warn("Failed to load truss result", err);
    }
  } else if (s.status === "error") {
    setTrussMessage(`Error: ${s.error ? s.error.split("\n").slice(-2).join(" ") : "unknown error"}`, true);
    console.error(s.error);
  }
}

$("truss-run-btn").addEventListener("click", runTruss);
$("truss-stop-btn").addEventListener("click", stopTruss);

// ---------------------------------------------------------------------
// Study tab
// ---------------------------------------------------------------------
function renderStudyCustomBuilder() {
  renderStudyTrussBuilder();
  renderStudyContinuumBuilder();
}

function renderStudyTrussBuilder() {
  const probList = $("study-problems-list");
  const backList = $("study-backends-list");
  if (!probList || !backList) return;
  probList.innerHTML = "";
  for (const b of state.truss.benchmarks) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = b.id;
    cb.className = "study-problem-cb";
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${b.alias || b.id} — ${b.title}`));
    probList.appendChild(label);
  }
  backList.innerHTML = "";
  const relevant = state.quantumBackends.filter((b) =>
    ["exact", "sa", "tabu", "qaoa", "greedy", "dwave_sa", "dwave_tabu"].includes(b.name)
  );
  for (const b of relevant) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = b.name;
    cb.className = "study-backend-cb";
    if (!b.available) cb.disabled = true;
    label.title = b.available ? (b.description || "") : (b.reason || "not available");
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${b.name}${b.available ? "" : " (unavailable)"}`));
    backList.appendChild(label);
  }
}

// Continuum custom builder: problems x methods x seeds (+ mesh override), options
// from GET /api/study/custom_options (same ids the API's `custom` request takes).
async function renderStudyContinuumBuilder() {
  const pl = $("study-cproblems-list");
  const ml = $("study-cmethods-list");
  if (!pl || !ml) return;
  if (!state.study.customOptions) {
    try {
      const resp = await fetch("/api/study/custom_options");
      state.study.customOptions = await resp.json();
    } catch (err) {
      pl.textContent = `Could not load options: ${err}`;
      return;
    }
  }
  const opts = state.study.customOptions;
  const keep = (cls) => new Set([...document.querySelectorAll(`.${cls}:checked`)].map((c) => c.value));
  const prevP = keep("study-cproblem-cb");
  const prevM = keep("study-cmethod-cb");
  const firstTime = !pl.children.length;
  pl.innerHTML = "";
  ml.innerHTML = "";
  const defaultsP = new Set(["cantilever_beam"]);
  const defaultsM = new Set(["MMA", "QUBO-sa (block)"]);
  for (const p of opts.problems) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = p.id; cb.className = "study-cproblem-cb";
    cb.checked = firstTime ? defaultsP.has(p.id) : prevP.has(p.id);
    cb.addEventListener("change", updateStudyEstimate);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${p.id} (MC ${p.default_mesh_control})`));
    pl.appendChild(label);
  }
  for (const m of opts.methods) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = m.id; cb.className = "study-cmethod-cb";
    cb.checked = firstTime ? defaultsM.has(m.id) : prevM.has(m.id);
    cb.addEventListener("change", updateStudyEstimate);
    label.title = m.optimizer === "QUBO" ? `QUBO ${JSON.stringify(m.options || {})}` : m.optimizer;
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${m.id}`));
    ml.appendChild(label);
  }
  updateStudyEstimate();
}

function parseSeeds(text) {
  const out = [];
  for (const tok of String(text || "").split(/[\s,;]+/)) {
    if (tok === "") continue;
    const n = Number(tok);
    if (!Number.isInteger(n)) return null;
    out.push(n);
  }
  return out.length ? out : null;
}

function collectCustomContinuum() {
  const problems = [...document.querySelectorAll(".study-cproblem-cb:checked")].map((c) => c.value);
  const methods = [...document.querySelectorAll(".study-cmethod-cb:checked")].map((c) => c.value);
  const seeds = parseSeeds($("study-cseeds").value);
  const mesh = $("study-cmesh").value.trim();
  return {
    problems, methods, seeds,
    mesh_control: mesh === "" ? null : parseInt(mesh, 10),
    max_iter: parseInt($("study-cmaxiter").value, 10) || 100,
    include_baseline: $("study-cbaseline").checked,
  };
}

function updateStudyEstimate() {
  const box = $("study-cestimate");
  if (!box) return;
  const c = collectCustomContinuum();
  if (!c.problems.length || !c.methods.length || !c.seeds) { box.textContent = ""; return; }
  const stoch = c.methods.filter((m) => m.startsWith("QUBO") && m !== "BESO-sort").length;
  const det = c.methods.length - stoch + (c.include_baseline && !c.methods.includes("MMA") ? 1 : 0);
  const n = c.problems.length * (det + stoch * c.seeds.length);
  box.textContent = `${n} run${n === 1 ? "" : "s"} (${c.problems.length} problem${c.problems.length === 1 ? "" : "s"}; ` +
    `deterministic methods run once, QUBO methods ×${c.seeds.length} seed${c.seeds.length === 1 ? "" : "s"}).`;
}

$("study-suite-select").addEventListener("change", () => {
  const custom = $("study-suite-select").value === "custom";
  $("study-custom-card").hidden = !custom;
  if (custom) renderStudyCustomBuilder();
});
$("study-custom-kind").addEventListener("change", () => {
  const truss = $("study-custom-kind").value === "truss";
  $("study-custom-truss").hidden = !truss;
  $("study-custom-continuum").hidden = truss;
});
["study-cseeds", "study-cmesh", "study-cmaxiter", "study-cbaseline"].forEach((id) =>
  $(id).addEventListener("input", updateStudyEstimate));
$("study-rerun-quick-btn").addEventListener("click", () => {
  $("study-suite-select").value = "quick";
  $("study-suite-select").dispatchEvent(new Event("change"));
  setStudyMessage("Quick suite selected - press 'Run study' (about an hour on 2 cores; QUBO rows use the connectivity repair).", false);
});

function buildCustomStudySpec() {
  // truss kind: raw spec (problems x backends, iterative QUBO)
  const problems = [...document.querySelectorAll(".study-problem-cb:checked")].map((cb) => cb.value);
  const backends = [...document.querySelectorAll(".study-backend-cb:checked")].map((cb) => cb.value);
  if (!problems.length || !backends.length) return null;
  const runs = [];
  for (const p of problems) {
    for (const b of backends) {
      runs.push({ study: "custom", kind: "truss", problem: p, method: "qubo", backend: b, seeds: [0], label: `qubo-${b}` });
    }
  }
  return { name: "custom", runs };
}

function setStudyMessage(msg, isError) {
  const box = $("study-run-message");
  box.textContent = msg;
  box.classList.toggle("error", !!isError);
  box.classList.toggle("ok", !isError && !!msg);
}
function setStudyRunning(running) {
  $("study-run-btn").disabled = running;
  $("study-stop-btn").disabled = !running;
}

async function runStudy() {
  const suite = $("study-suite-select").value;
  let payload;
  if (suite === "custom" && $("study-custom-kind").value === "continuum") {
    const c = collectCustomContinuum();
    if (!c.problems.length || !c.methods.length) { setStudyMessage("Pick at least one problem and one method.", true); return; }
    if (!c.seeds) { setStudyMessage("Seeds must be comma-separated integers, e.g. 0,1,2.", true); return; }
    payload = { custom: c, figures: true };
  } else if (suite === "custom") {
    const spec = buildCustomStudySpec();
    if (!spec) { setStudyMessage("Pick at least one problem and one backend.", true); return; }
    payload = { spec, figures: true };
  } else {
    payload = { suite, figures: true };
  }
  setStudyMessage("Submitting…", false);
  $("study-progress").textContent = "";
  let resp;
  try {
    resp = await fetch("/api/study/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    setStudyMessage(`Request failed: ${err}`, true);
    return;
  }
  if (!resp.ok) { setStudyMessage(await safeErrorDetail(resp), true); return; }
  const data = await resp.json();
  state.study.jobId = data.job_id;
  state.study.logCursor = 0;
  $("study-log-output").textContent = "";
  $("study-figures").innerHTML = "";
  $("study-results-table-wrap").innerHTML = "";
  $("study-downloads").innerHTML = "";
  $("study-results-title").textContent = "";
  $("study-summary-details").hidden = true;
  state.study.studyId = data.study_id || null;
  setStudyRunning(true);
  setStudyMessage(
    data.queue_position && data.queue_position > 1 ? `Queued (position ${data.queue_position}).` : "Running…",
    false
  );
  startStudyPolling(data.job_id);
}

async function stopStudy() {
  if (!state.study.jobId) return;
  await fetch(`/api/jobs/${state.study.jobId}/stop`, { method: "POST" });
  setStudyMessage("Stop requested…", false);
}

function startStudyPolling(jobId) {
  if (state.study.polling) clearInterval(state.study.polling);
  state.study.polling = setInterval(() => pollStudyOnce(jobId), 1200);
  pollStudyOnce(jobId);
}

async function pollStudyOnce(jobId) {
  let resp;
  try {
    resp = await fetch(`/api/study/status/${jobId}?since=${state.study.logCursor}`);
  } catch {
    return;
  }
  if (!resp.ok) return;
  const s = await resp.json();

  if (s.log_lines && s.log_lines.length) {
    const box = $("study-log-output");
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    box.textContent += (box.textContent ? "\n" : "") + s.log_lines.join("\n");
    if (atBottom) box.scrollTop = box.scrollHeight;
  }
  state.study.logCursor = s.log_cursor;

  if (s.progress && s.progress.n_runs) {
    $("study-progress").textContent =
      `Run ${s.progress.index || 0} / ${s.progress.n_runs}` + (s.progress.label ? ` — ${s.progress.label}` : "");
  }

  if (s.status === "running" || s.status === "queued") {
    setStudyMessage(
      s.queue_position && s.queue_position > 1 ? `Queued (position ${s.queue_position}).` : "Running…",
      false
    );
    return;
  }

  clearInterval(state.study.polling);
  state.study.polling = null;
  setStudyRunning(false);
  if (s.status === "done" || s.status === "stopped") {
    setStudyMessage(s.status === "done" ? "Done." : "Stopped.", false);
    await loadStudyResults(jobId, `This session's study ${s.study_id || jobId}`);
    refreshPastStudies();
  } else if (s.status === "error") {
    setStudyMessage(`Error: ${s.error ? s.error.split("\n").slice(-2).join(" ") : "unknown error"}`, true);
    console.error(s.error);
  }
}

async function loadStudyResults(id, title) {
  let data;
  try {
    const resp = await fetch(`/api/study/results/${encodeURIComponent(id)}`);
    if (!resp.ok) { $("study-results-table-wrap").textContent = await safeErrorDetail(resp); return; }
    data = await resp.json();
  } catch (err) {
    console.warn("Failed to load study results", err);
    return;
  }
  state.study.shownId = data.study_id || id;
  $("study-results-title").textContent = title || (data.study_id ? `Study ${data.study_id}` : "");
  renderStudyFigures(data.files || []);
  renderStudyDownloads(data.files || []);
  renderStudyResultsTable(data.results);
  const det = $("study-summary-details");
  if (data.summary_md) { $("study-summary-md").textContent = data.summary_md; det.hidden = false; }
  else det.hidden = true;
  if (data.status === "incomplete") {
    $("study-results-table-wrap").textContent =
      "This study has no results.json (the run was interrupted, e.g. the server restarted). Figures/files written so far are shown above.";
  }
  highlightPastStudy();
}

// -- past studies (GET /api/study/list: survives a server restart) ------------
async function refreshPastStudies() {
  const box = $("study-past-list");
  if (!box) return;
  let data;
  try {
    const resp = await fetch("/api/study/list", { cache: "no-store" });
    if (!resp.ok) return;
    data = await resp.json();
  } catch { return; }
  state.study.past = data.studies || [];
  box.innerHTML = "";
  if (!state.study.past.length) {
    box.innerHTML = '<span class="muted small">No past studies found yet.</span>';
    return;
  }
  for (const st of state.study.past) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "past-item";
    b.dataset.id = st.id;
    const when = st.created || new Date(st.mtime * 1000).toLocaleString();
    const tags = [];
    if (!st.complete) tags.push('<span class="pill pill-bad">incomplete</span>');
    if (st.n_invalid) tags.push(`<span class="pill pill-bad">${st.n_invalid} physically invalid</span>`);
    else if (st.has_audit) tags.push('<span class="pill pill-ok">audit ok</span>');
    if (st.has_audit === false) tags.push('<span class="pill pill-na" title="Continuum rows have no audit data: run before the connectivity fix / audit. Re-run with corrections.">pre-fix</span>');
    if (st.n_errors) tags.push(`<span class="pill pill-bad">${st.n_errors} error${st.n_errors > 1 ? "s" : ""}</span>`);
    b.innerHTML =
      `<span><b>${escHtml(st.suite || "study")}</b> · ${escHtml(st.id)}${st.source === "cli" ? " (CLI)" : ""}</span>` +
      `<span class="past-meta">${escHtml(when)} · ${st.n_records ?? "?"} rows · ${st.n_figures} fig.</span>` +
      (tags.length ? `<span class="past-meta">${tags.join(" ")}</span>` : "");
    b.addEventListener("click", () => {
      loadStudyResults(st.id, `Study ${st.id}${st.suite ? ` (${st.suite})` : ""} — ${when}`);
    });
    box.appendChild(b);
  }
  highlightPastStudy();
}

function highlightPastStudy() {
  document.querySelectorAll("#study-past-list .past-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.id === state.study.shownId));
}
$("study-past-refresh-btn").addEventListener("click", refreshPastStudies);

function renderStudyFigures(files) {
  const box = $("study-figures");
  box.innerHTML = "";
  const pngs = files.filter((f) => f.name.toLowerCase().endsWith(".png"));
  for (const f of pngs) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = f.url;
    img.alt = f.name;
    img.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.textContent = f.name;
    fig.appendChild(img);
    fig.appendChild(cap);
    box.appendChild(fig);
  }
  if (!pngs.length) box.textContent = "No figures yet.";
}

function renderStudyDownloads(files) {
  const box = $("study-downloads");
  box.innerHTML = "";
  for (const name of ["results.csv", "results.json", "summary.md"]) {
    const f = files.find((x) => x.name === name);
    if (!f) continue;
    const a = document.createElement("a");
    a.href = f.url;
    a.textContent = name;
    a.className = "btn small";
    a.style.marginLeft = "6px";
    box.appendChild(a);
  }
}

// Study results table. Reading guide (also in the README):
//  * "compliance (native)" is the optimizer's own end-of-run value (its own
//    projection / Heaviside beta) and is NOT comparable across optimizers;
//  * "crisp compliance" is the common yardstick (final field projected to a
//    0/1 design at "V target", one FE solve) — continuum rows only;
//  * "gap" is crisp-vs-MMA for continuum rows and vs. the exact optimum for
//    truss rows; it is n/a when the run diverged / is infeasible / failed.
const STUDY_COLS = [
  ["study", "study", "Study block (S1, S3, ...)."],
  ["problem", "problem", "Benchmark / continuum example."],
  ["label", "label", "Method label as configured in the suite (e.g. 'MMA (V=0.26)')."],
  ["backend", "backend", "QUBO backend (blank for OC / MMA)."],
  ["compliance", "compliance (native)", "The optimizer's own end-of-run compliance (its own projection). Not comparable across optimizers - use the crisp column for continuum runs."],
  ["crisp_compliance", "crisp compliance", "Common crisp evaluation: final filtered field projected to a 0/1 design at the target volume, one FE solve. The comparable continuum metric."],
  ["volfrac_target", "V target", "Target volume fraction the crisp evaluation was projected to."],
  ["volume_fraction", "V native", "Volume fraction of the optimizer's own (native) final design."],
  ["c_ref", "c ref", "Reference compliance the gap is measured against (truss: exact optimum)."],
  ["gap", "gap (%)", "Truss: (c - c_exact)/c_exact. Continuum: (crisp c - MMA crisp c)/MMA crisp c. n/a = diverged / infeasible / failed."],
  ["status", "status", "ok | physically invalid (fails the physics audit) | diverged (crisp c > 10x MMA or singular) | infeasible (kinematically unstable truss) | error."],
  ["audit_ok", "audit", "Physics check of the design (freeto/audit.py): PASS = supports/loads on solid, one grounded body, volume on target. FAIL = physically invalid. Continuum rows only; blank = no audit data (older study)."],
  ["n_components", "n comp.", "Face-connected components of the crisp design (1 = one body)."],
  ["floating_frac", "floating (%)", "Share of the solid that touches no support (pass if < 1 %)."],
  ["loads_solid", "loads on solid (%)", "Share of the load magnitude acting on nodes adjacent to solid material (pass if >= 99 %)."],
  ["iterations", "iters", "Iterations (continuum) / optimizer steps (truss)."],
  ["wall_time", "wall (s)", "Wall-clock time of the run."],
  ["seed", "seed", "RNG seed."],
  ["run_id", "run id", "Unique run id (study - index - problem - label - seed)."],
];

function studyRecordStatus(rec) {
  if (rec.error) return "error";
  if (rec.audit_ok === false) return "physically invalid";
  if (rec.diverged) return "diverged";
  if (rec.feasible === false) return "infeasible";
  return "ok";
}

function renderStudyResultsTable(results) {
  const wrap = $("study-results-table-wrap");
  wrap.innerHTML = "";
  const records = results && Array.isArray(results.records) ? results.records : [];
  if (!records.length) { wrap.textContent = "No results yet."; return; }
  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const fmtNum = (v) => (Number.isInteger(v) ? String(v) : Number(v).toPrecision(5));
  const table = document.createElement("table");
  table.className = "results-table";
  const thead = document.createElement("thead");
  thead.innerHTML = `<tr>${STUDY_COLS.map(([, head, tip]) => `<th title="${esc(tip)}">${esc(head)}</th>`).join("")}</tr>`;
  const tbody = document.createElement("tbody");
  for (const rec of records) {
    const status = studyRecordStatus(rec);
    const tr = document.createElement("tr");
    if (status === "physically invalid") tr.className = "row-invalid";
    tr.innerHTML = STUDY_COLS.map(([key]) => {
      if (key === "audit_ok") {
        if (rec.audit_ok == null) return '<td class="na" title="no audit data for this row">—</td>';
        return `<td><span class="pill ${rec.audit_ok ? "pill-ok" : "pill-bad"}">${rec.audit_ok ? "PASS" : "FAIL"}</span></td>`;
      }
      if (key === "floating_frac" || key === "loads_solid") {
        const v = rec[key];
        if (v == null) return '<td class="na">—</td>';
        return `<td>${(100 * Number(v)).toFixed(key === "floating_frac" ? 2 : 1)}</td>`;
      }
      if (key === "status") {
        const cls = status === "ok" ? "pill-ok" : "pill-bad";
        const tip = rec.error ? ` title="${esc(rec.error)}"` : "";
        return `<td><span class="pill ${cls}"${tip}>${status}</span></td>`;
      }
      let v = rec[key];
      if (key === "gap" && v == null) {
        const tip = status === "ok" ? "no reference run to compare against"
          : "diverged/infeasible" + (rec.error ? ` (${rec.error})` : "");
        return `<td title="${esc(tip)}" class="na">n/a</td>`;
      }
      if (key === "gap" && v != null) return `<td>${(100 * Number(v)).toFixed(2)}%</td>`;
      if (v == null) v = "—";
      else if (typeof v === "number") v = fmtNum(v);
      else if (typeof v === "boolean") v = v ? "yes" : "no";
      return `<td>${esc(v)}</td>`;
    }).join("");
    tbody.appendChild(tr);
  }
  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
}

$("study-run-btn").addEventListener("click", runStudy);
$("study-stop-btn").addEventListener("click", stopStudy);

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
function init() {
  initTheme();
  initScene();
  initChart();
  initTabs();
  loadHealth();
  fetchExamples();
  // the paper's values in every field once the backend list is known
  fetchQuantumBackends().then(initPaperDefaults);
  fetchTrussBenchmarks();
  renderTrussResultsTable();
  refreshAuditJobList();
  refreshPastStudies();
  addLoadRow();
  // Exposed for debugging / smoke tests only.
  window.__freeto = { state, scene, inputGroup, designGroup, camera };
}
document.addEventListener("DOMContentLoaded", init);
