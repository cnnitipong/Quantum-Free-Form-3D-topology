# Contributing to QFF-3D

Thanks for your interest. Bug reports, questions and pull requests are welcome.

- **Bugs and questions:** open an issue (the bug-report template asks for the version, the
  command or web-app steps, and the log). For a run that gives an unexpected design, attach
  the job's settings (the web app's request JSON or the CLI command) and the STL files if you can.
- **Pull requests:** keep them focused, add or update a test in `tests/`, and run the suite:

  ```bash
  pip install -r requirements.txt pytest httpx
  OMP_NUM_THREADS=1 python -m pytest -q
  ```

  The browser tests (`tests/test_playwright_*.py`) skip themselves when Playwright is not installed.
- **Numerical behaviour:** changes must not alter the results of the paper's runs. The
  reproduction settings are in `docs/PAPER_SETTINGS.md`; `tests/test_webapp_paper.py` and
  `freeto/paper.py` guard them. If a change is meant to alter results, say so in the pull request
  and explain why.
- **Style:** plain Python 3.10+, NumPy-style docstrings where helpful, no new heavy dependencies in
  `requirements.txt` without discussion (optional extras go in `requirements-quantum.txt` /
  `pyproject.toml`). The frontend has no build step; use relative URLs only (the app is also
  served under `/qff3d/` behind a proxy).
- **Licence:** contributions are accepted under the MIT licence of this repository.
