# Style changes: QFF-3D, matched to nitipong.com

The web app is now branded **QFF-3D** (Quantum Free-Form 3D Topology Optimisation) and styled
after the author's site, nitipong.com. Only its design tokens and fonts are reused (no images,
text or code from the site). Element ids, the classes app.js uses, API routes, job/file formats,
the `freeto` package name and the `FREETO_*` environment variables are unchanged. Numerical
behaviour is unchanged.

## Branding (static/index.html, launchers, server)
- Page title "QFF-3D · Quantum Free-Form 3D Topology Optimisation". The header has a square
  monogram box ("Q3", 1px ink border, a small pink corner marker), the name "QFF-3D" in Newsreader
  and the full name as a small mono line (hidden below 1100px).
- New footer: "QFF-3D · Nitipong Praphaphankul · Architectural Intelligence Research Group,
  Chulalongkorn University", the credit "Built on FreeTO (Ibhadode, Fu & Qureshi, 2024, MIT
  licence)" and a link to https://github.com/cnnitipong/Quantum-Free-Form-3D-topology.
- Launchers renamed to `Start QFF-3D (macOS).command` and `Start QFF-3D (Windows).bat`; window
  title and start-up text say QFF-3D. The venv folder keeps the name `FreeTO-Python`, so an
  existing install is reused. Server banner, FastAPI title and CLI help say QFF-3D.

## Quantum first (Setup tab)
- The optimizer select is the first, full-width field of "Parameters". It lists
  "QUBO - quantum-ready design update" first and selected, then OC and MMA.
- The QUBO card ("QUBO design update") is visible by default, preset to backend `sa` and Hessian
  `block` (the paper's QUBO-SA (block) runs; all other fields are the `QUBOOptions` defaults).
  QAOA stays selectable, not the default. If `freeto.quantum` is missing the UI falls back to OC.
- Loading an example keeps the selected optimizer (`prefill.optimizer` is now `null` unless an
  example prescribes one). The job API default is unchanged: a request without `optimizer` runs OC.
- The example select starts on "Cantilever beam". Loaded with QUBO selected it uses MeshControl
  25 (instead of the example's 50), so Load then Run finishes in about a minute.

## static/css/style.css (rewritten)
- Tokens on `:root`: `--bg #fbfbf9`, `--surface #fff`, `--surface-2 #f4f4f2`, `--ink`/`--text
  #141414`, `--soft #3a3a3a`, `--muted`/`--text-muted #6b6b6b`, `--line rgba(0,0,0,.12)`,
  `--line-strong rgba(0,0,0,.3)`, `--accent #DE5C8E` (Chula pink), `--ease
  cubic-bezier(.22,.61,.36,1)`. `--text` and `--text-muted` stay because app.js reads them.
- Fonts (Google Fonts link, with offline fallbacks): Inter 300-600 for body text; Newsreader
  (opsz 6..72) for the brand, card and panel titles, with slight negative tracking;
  JetBrains Mono at about 11px, uppercase, 0.1-0.14em tracking, for tabs, field labels, the
  status bar, badges and buttons. Fallbacks: system-ui / Georgia / ui-monospace stacks.
- Square corners everywhere, 1px hairlines in `--line`, white cards on the off-white page, no
  shadows, more whitespace.
- Header: off-white with a bottom hairline. Tabs: mono uppercase, the active one ink with a 1px
  pink underline. The theme button is an outlined mono button.
- Card titles are numbered like the site ("01 / Example problems"). The number comes from a CSS
  counter, so hidden cards are skipped. The legend floats inside the card as a heading.
- Buttons: primary is solid ink with white mono text (pink with ink text on hover, AA contrast);
  secondary is outlined ink; danger is outlined red; disabled is muted with a `--line` border.
- Inputs: white, 1px `--line` border, square; on focus an ink border and a 1px pink outline.
  Checkbox rows and lists stay sentence case. Greek symbols in labels (λ_q, γ, μ) are kept out
  of the uppercase transform.
- Status bar: mono uppercase small muted text separated by hairlines.
- PASS/FAIL pills and badges: square and outlined, green #2E7D32 / red #B3261E on light tints.
- Log panes: #f4f4f2, JetBrains Mono 12px, ink text.
- 3D and truss viewports: always light (#f4f4f2) with hairlines; the toolbar and legend are white
  overlays with hairline borders. BC legend colours are slightly desaturated.
- Dark theme (OS setting or the Theme button) is kept as an inverted ink/paper variant. The
  viewports stay light.

## static/js/app.js (colour/font constants only)
- `ROLE_COLORS` desaturated slightly (domain #b5b5b1, fixed #3d63c4, x/y/z-fixed #3fb0c4 /
  #2f9c8e / #227a70, load #e07b39, keep #4caf6e), matching the `.sw-*` swatches.
- `VIEW_COLORS`: scene background #f4f4f2, neutral hemisphere light, design mesh mid grey
  #9a9a96 (final) / #a9a9a5 (running), selected load and load arrow #c0392b.
- Chart.js: compliance ink #141414, volume fraction #8a8a8a dashed, grid rgba(0,0,0,.08);
  `CHART_SERIES` = #141414, #6b6b6b, #DE5C8E, #a3a3a0, #3a3a3a; Inter for ticks, JetBrains Mono
  for axis titles and the legend. Dark-theme variants in `CHART_COLORS_DARK`.
- `TRUSS_COLORS`: grey candidate bars, ink members and nodes, BC-blue supports, load-orange arrows.
- Quantum-first logic: `DEFAULT_EXAMPLE`, `QUBO_QUICK_MESH`, and `applyPrefill` keeping the
  selected optimizer.

## Screenshots
- Regenerated in webapp/screenshots at 1440x900: 01_initial, 02_example_loaded, 03_running,
  04_done (a finished default QUBO run on the cantilever), 05_truss_tab,
  06_truss_benchmark_loaded, 07_truss_done. Google Fonts could not be reached from the build
  sandbox, so they show the fallback fonts.
