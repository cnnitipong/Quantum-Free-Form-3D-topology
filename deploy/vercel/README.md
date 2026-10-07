# Serving QFF-3D at https://nitipong.com/qff3d (Vercel)

nitipong.com is a static site on Vercel. Vercel cannot run QFF-3D's jobs (minutes of CPU in a
background thread), so the app runs in a Docker container on a Hugging Face Space
(`deploy/huggingface/README.md`) and nitipong.com forwards `/qff3d/*` to it with a Vercel
**external rewrite**. The browser stays on `nitipong.com/qff3d/...`; Vercel fetches each request from
`https://<space-subdomain>.hf.space/...` with the `/qff3d` prefix removed.

The app is built for this: every URL in its frontend is relative (no `/static/...` or `/api/...`),
so it works under any prefix. This was tested with a local reverse proxy that does exactly what the
rules below do (redirects, prefix stripping, a 4.5 MB body limit): page, examples, a full QUBO run,
STL/NPZ downloads, the Truss tab and uploads all worked with no 404s and no console errors.

## Option A (recommended): proxy with an external rewrite

1. Find the Space's direct address. For the Space `cnnitipong/qff3d` it is
   `https://cnnitipong-qff3d.hf.space` (owner and Space name joined by `-`, lower case; it is also
   shown under "Embed this Space" in the Space's `...` menu). Open it once to check the app loads.
2. In the **nitipong.com** project (not this repository), open `vercel.json` at the project root
   (create it if missing) and merge in the `redirects` and `rewrites` from
   [`vercel.json`](vercel.json) in this folder, replacing **`YOUR-SPACE-SUBDOMAIN`** (both places)
   with the subdomain from step 1, e.g. `cnnitipong-qff3d`:

   ```json
   {
     "redirects": [
       { "source": "/qff3d", "destination": "/qff3d/", "permanent": true },
       { "source": "/qqf3d", "destination": "/qff3d/", "permanent": true },
       { "source": "/qqf3d/:path*", "destination": "/qff3d/", "permanent": true }
     ],
     "rewrites": [
       { "source": "/qff3d/", "destination": "https://YOUR-SPACE-SUBDOMAIN.hf.space/" },
       { "source": "/qff3d/:path*", "destination": "https://YOUR-SPACE-SUBDOMAIN.hf.space/:path*" }
     ]
   }
   ```

   If the file already has `redirects` / `rewrites` arrays, add these entries to them (put them
   before any catch-all rewrite such as `"/(.*)" -> "/index.html"`, which would otherwise win).
   - `/qff3d` -> `/qff3d/` (308) is required: the app's relative URLs resolve against the page's
     directory, so it must be loaded with the trailing slash.
   - `/qqf3d` and `/qqf3d/...` -> `/qff3d/` is the typo alias.
   - The explicit `/qff3d/` rewrite covers the bare directory, which `/:path*` may not match.
3. Do **not** set `"trailingSlash": false` in that `vercel.json` (it would redirect `/qff3d/` back
   to `/qff3d` and loop with the rule above). Unset or `true` both work.
4. Commit and push the nitipong.com repository (or `vercel --prod`); Vercel redeploys.
5. Test (see also `DEPLOY_CHECKLIST.md`):
   - `https://nitipong.com/qff3d` -> redirects to `/qff3d/`, the QFF-3D app loads, the footer shows
     "Shared demo server: one job at a time, ...".
   - `https://nitipong.com/qqf3d` -> redirects to `/qff3d/`.
   - `https://nitipong.com/qff3d/api/health` -> JSON with `"status": "ok"` and `"public": {"enabled": true, ...}`.
   - Load an example, press Run, download the STL.

If the site is a Next.js app rather than plain static files, put the same entries in
`next.config.js` (`async redirects()` / `async rewrites()`) instead; the rules are identical.

### Caveats of external rewrites

- **Request body size: about 4.5 MB per request.** Vercel rejects larger request bodies on the
  proxied path (HTTP 413). The app uploads one STL per request and the bundled examples need no
  upload at all, so only custom STLs above ~4.5 MB are affected. The app says so in the upload card
  and in the error message: large custom STLs work in the local version, or directly on the
  `https://<space-subdomain>.hf.space/` address (20 MB limit there). To show that address in the
  message, set the Space variable `QFF3D_DIRECT_URL=https://<space-subdomain>.hf.space/`.
- **Request duration.** Proxied requests have a time limit, but the app never needs a long request:
  a job runs in a background thread on the Space and the browser polls its status about once per
  second; every request finishes in well under a second (the finished design's STL in a few seconds
  at most).
- **Caching.** The app sends no `s-maxage` / `public` cache headers on API responses, so Vercel's CDN
  does not cache them; `index.html` is sent with `Cache-Control: no-cache`, so a redeploy of the Space
  shows up immediately.
- **Client address.** Vercel passes the visitor's address in `X-Forwarded-For`; the app's
  one-job-per-visitor limit uses its first entry.
- **A sleeping Space.** A free CPU Space sleeps after a long idle period (48 h). The first request then
  wakes it, which takes a minute or two; meanwhile the proxied page shows Hugging Face's
  "starting" page, possibly unstyled. Opening the `hf.space` address directly, or reloading after a
  minute, fixes it. The Space must be **public**; a private Space cannot be proxied.
- Nothing else on nitipong.com changes: only paths under `/qff3d` and `/qqf3d` are affected.

## Option B: link or iframe (no proxy)

If you prefer not to proxy:

- **Link / redirect.** Point `/qff3d` straight at the Space (the address bar then shows `hf.space`):

  ```json
  { "redirects": [
      { "source": "/qff3d", "destination": "https://YOUR-SPACE-SUBDOMAIN.hf.space/", "permanent": false },
      { "source": "/qff3d/:path*", "destination": "https://YOUR-SPACE-SUBDOMAIN.hf.space/", "permanent": false },
      { "source": "/qqf3d/:path*", "destination": "https://YOUR-SPACE-SUBDOMAIN.hf.space/", "permanent": false }
  ] }
  ```

  No upload limit beyond the app's own 20 MB; `permanent: false` keeps the option of switching to
  the proxy later.
- **Iframe.** Save [`qff3d-iframe.html`](qff3d-iframe.html) as `qff3d/index.html` in the nitipong.com
  site (replace `YOUR-SPACE-SUBDOMAIN`). The address bar keeps `nitipong.com/qff3d/`, uploads go
  straight to the Space (no 4.5 MB limit), and the page links to the Space in its own tab. The app's
  theme setting is then stored per embedding site by the browser.
