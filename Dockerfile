# QFF-3D web app (Quantum Free-Form 3D Topology Optimisation) as a container.
#
# Built for a Hugging Face Docker Space (CPU basic, port 7860; see
# deploy/huggingface/README.md) that https://nitipong.com/qff3d proxies to
# (deploy/vercel/README.md), but runs anywhere:
#
#   docker build -t qff3d .
#   docker run --rm -p 7860:7860 qff3d          # -> http://localhost:7860/
#   docker run --rm -p 7860:7860 -e QFF3D_PUBLIC=0 qff3d   # without the demo-server limits
#
# QFF3D_PUBLIC=1 turns on the shared-server limits of webapp/public.py (one
# job at a time, queue, mesh/iteration caps, 20 min per job, results deleted
# after 2 h, studies off). Every limit is an environment variable.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # one thread per BLAS / OpenMP / MKL pool: the paper default (the study
    # pins one thread too), reproducible floating-point summation order, and
    # fair sharing of a small CPU
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FREETO_THREADS=1 \
    MPLBACKEND=Agg \
    PORT=7860 \
    QFF3D_PUBLIC=1

# Hugging Face runs Docker Spaces as uid 1000; create that user up front.
RUN useradd --create-home --uid 1000 user

WORKDIR /home/user/app

# Dependencies first (cached layer). pypardiso (Intel MKL PARDISO) only exists
# for x86-64; requirements.txt already skips it elsewhere, and if it still
# fails to install the requirements are installed without it -- freeto then
# uses its SciPy / PyAMG solvers automatically (freeto/fe.py, solver "auto").
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && ( python -m pip install -r requirements.txt \
      || ( echo "pypardiso failed to install: continuing with the SciPy solvers" \
           && grep -v -i '^pypardiso' requirements.txt > /tmp/requirements-nopardiso.txt \
           && python -m pip install -r /tmp/requirements-nopardiso.txt ) ) \
 && python -c "import numpy, scipy, skimage, fastapi, uvicorn, matplotlib; print('core deps ok')" \
 && (python -c "import pypardiso; print('pypardiso available')" || echo "pypardiso not available")

COPY --chown=user:user . .

ENV HOME=/home/user \
    FREETO_WEBAPP_DIR=/home/user/data \
    MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /home/user/data && chown -R user:user /home/user/data

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '7860'), timeout=4).close()" || exit 1

# Bind to all interfaces on $PORT (7860 on Hugging Face); --strict-port: never
# fall forward to another port, the platform routes to exactly this one.
CMD ["sh", "-c", "exec python -m webapp.server --host 0.0.0.0 --port \"${PORT:-7860}\" --strict-port"]
