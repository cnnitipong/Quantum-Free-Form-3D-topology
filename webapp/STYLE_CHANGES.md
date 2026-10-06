# Style changes: Chulalongkorn Architecture identity

Purely visual. No element id, JS-used class, API call or feature was changed.

## static/index.html
- Added Google Fonts link for Sarabun (400, 600, 700) with preconnect. The app still works offline using the fallback stack.

## static/css/style.css
- New palette as CSS custom properties on `:root`: yellow #FFDE59, brown #683817, tan #B38450, taupe #68543C, black #231F20, beige #DDD5C8, cream #FBF1CF, off-white #FAF8F4, panels white.
- Font: "Sarabun", "TH Sarabun New", "Helvetica Neue", Arial, sans-serif.
- Header: taupe bar with a 4px yellow bottom rule. Brand title is white, with "-Python" in yellow.
- Tabs: yellow underline on the active tab.
- Section titles (card legends): uppercase, small, brown, with a short yellow left bar.
- Inputs: beige border, brown focus ring.
- Buttons: primary is brown with white text and a darker hover. Secondary is white with brown text and a beige border. Danger is a tuned red tint.
- Status colours: PASS #2E7D32 and FAIL #C00000, on light green and light red tints.
- Info boxes and highlighted rows use cream (study note, selected rows, dropzone and so on).
- Log panels: cream background, dark monospace text.
- 3D viewport: warm dark #2B2622. Toolbar and legend overlays are warm dark. Legend and BC colours are unchanged in meaning (the domain swatch is a warm grey).
- Dark theme (toggle or OS setting) is kept, re-skinned as a warm dark variant with a yellow accent. Light is the default.

## static/js/app.js (colour constants only)
- three.js: scene background 0x2B2622, warm hemisphere light, design mesh light tan (0xC2B49E running, 0xD9C9B0 final), domain role colour a warm grey. The functional BC role colours are unchanged.
- Chart.js: compliance #683817, volume fraction #B38450, gridlines beige. A `CHART_SERIES` palette is defined for extra series (#683817, #B38450, #FFDE59, #68543C, #8C8C8C). Lighter variants are used only in the dark theme.
- Truss canvas (drawn on the dark viewport): light warm bars and nodes, selected bars brand yellow, supports tan, load arrow orange.

## Screenshots
- Regenerated in webapp/screenshots at 1440x900: 01 to 06, plus 07_truss_done.
