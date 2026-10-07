# Hosting QFF-3D on a Hugging Face Space (Docker)

The online version at https://nitipong.com/qff3d is this repository's `Dockerfile` running on a
Hugging Face **Docker Space** (free "CPU basic" hardware: 2 vCPU, 16 GB RAM), with nitipong.com
proxying `/qff3d/*` to it (`deploy/vercel/README.md`). The image sets `QFF3D_PUBLIC=1`, so the
shared-server limits of `webapp/public.py` apply (one job at a time, MeshControl <= 50, 300
iterations, 20 min per job, runs deleted after 2 h, studies off).

## 1. Create the Space (once)

1. Sign in at https://huggingface.co (the examples below use the account `cnnitipong`).
2. **New Space**: https://huggingface.co/new-space
   - Owner: `cnnitipong`, Space name: `qff3d` -> Space id `cnnitipong/qff3d`
   - License: `mit`
   - SDK: **Docker**, template **Blank**
   - Hardware: **CPU basic** (free)
   - Visibility: **Public** (required for the nitipong.com proxy)
   - Create Space. It starts empty ("no application file").
3. The app's address is `https://cnnitipong-qff3d.hf.space/` (owner-name, lower case). The Space page
   itself is https://huggingface.co/spaces/cnnitipong/qff3d.

The Space's `README.md` must start with this YAML front matter; the deployment below writes it from
[`SPACE_README.md`](SPACE_README.md) (the repository's own README stays as it is):

```yaml
---
title: QFF-3D
emoji: 🧊
colorFrom: gray
colorTo: pink
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: Quantum Free-Form 3D Topology Optimisation (QUBO updates)
---
```

`sdk: docker` makes the Space build the `Dockerfile` at the repository root; `app_port: 7860` is the
port the container listens on (`PORT=7860` in the Dockerfile).

## 2. Keep it in sync from GitHub (recommended)

The workflow [`.github/workflows/deploy-hf-space.yml`](../../.github/workflows/deploy-hf-space.yml)
uploads the repository to the Space on every push to `main` and on demand (Actions -> "Deploy to
Hugging Face Space" -> Run workflow). It skips itself, without failing, when the `HF_TOKEN` secret is
not set (e.g. in forks).

1. Create a token: https://huggingface.co/settings/tokens -> **Create new token** -> type **Write**
   (or fine-grained with write access to the Space `cnnitipong/qff3d`). Copy it (`hf_...`).
2. In the GitHub repository: **Settings -> Secrets and variables -> Actions**
   - tab **Secrets** -> **New repository secret**: name `HF_TOKEN`, value the token.
   - tab **Variables** -> **New repository variable**: name `HF_SPACE`, value `cnnitipong/qff3d`
     (optional; this is the default).
3. Push to `main` or run the workflow by hand. It takes about a minute; then the Space rebuilds the
   image (about 5 to 10 minutes the first time: the MKL / PARDISO wheels are large) and starts.

The workflow runs [`push_to_space.py`](push_to_space.py), which uploads the files git tracks (minus
tests, screenshots and CI files) through the Hugging Face Hub API, with `SPACE_README.md` as the
Space's `README.md`, and removes files from the Space that are no longer in the repository. It uses
the Hub API rather than `git push` because the Hub rejects binary files (the example STLs, PNGs)
pushed with plain git.

## 3. Or push by hand

```bash
pip install huggingface_hub
export HF_TOKEN=hf_...                       # Windows PowerShell: $env:HF_TOKEN="hf_..."
python deploy/huggingface/push_to_space.py --space cnnitipong/qff3d
python deploy/huggingface/push_to_space.py --dry-run     # only list what would be uploaded
```

## 4. Check it

- The Space page shows "Building", then "Running"; the build log ends with
  `Uvicorn running on http://0.0.0.0:7860`.
- `https://cnnitipong-qff3d.hf.space/api/health` returns JSON with `"status": "ok"`,
  `"core": {"source": "real", "usable": true, ...}` and `"public": {"enabled": true, ...}`.
- `https://cnnitipong-qff3d.hf.space/` shows the app with the footer notice
  "Shared demo server: one job at a time, runs are deleted after 2 hours. ...". Load
  "Cantilever beam", press Run: a paper run (MeshControl 36, up to 300 iterations) takes a few
  minutes on CPU basic.

## Settings (Space -> Settings -> Variables and secrets)

All optional; the defaults are in `webapp/public.py` and the README's "Running it online" table.

| Variable | Default | Meaning |
|---|---|---|
| `QFF3D_MAX_QUEUED` | 8 | waiting jobs before new ones get HTTP 429 |
| `QFF3D_MAX_MESH_CONTROL` | 50 | largest MeshControl |
| `QFF3D_MAX_JOB_MINUTES` | 20 | wall time per job |
| `QFF3D_TTL_MINUTES` | 120 | job results and uploads are deleted after this |
| `QFF3D_DIRECT_URL` | unset | e.g. `https://cnnitipong-qff3d.hf.space/`, shown to users whose STL is too large for the nitipong.com proxy |
| `QFF3D_PUBLIC` | 1 | `0` turns the limits off (not advisable on a shared server) |

Changing a variable restarts the Space. Job data lives in the container (`/home/user/data`) and is
lost on every restart or rebuild, which is fine for a demo; nothing persistent is needed.

## Run the same image elsewhere

```bash
docker build -t qff3d .
docker run --rm -p 7860:7860 qff3d                     # http://localhost:7860/
docker run --rm -p 7860:7860 -e QFF3D_PUBLIC=0 qff3d   # without the demo limits
```

Any container host that routes HTTP to port 7860 (or sets `PORT`) works the same way.

## Notes

- Free Spaces sleep after about 48 hours without visitors; the next visit wakes the Space (a minute
  or two). A paid "always on" hardware tier avoids that.
- One thread per process (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `FREETO_THREADS=1`): the paper's
  setting, needed for bit-identical results, and fair on a shared CPU.
- If `pypardiso` (x86-64 only) cannot be installed, the image is built without it and the solver
  falls back to SciPy / PyAMG automatically (`freeto/fe.py`, solver "auto"). A different sparse solver
  or library version changes the floating-point summation order and so gives a different, equally
  valid design trajectory; bit-identical reproduction of the paper's records needs the study's
  software stack (`docs/PAPER_SETTINGS.md`).
