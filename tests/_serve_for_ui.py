"""Start the web app for the Playwright audit smoke test.  If the real ``freeto.audit`` is
not importable yet, install the test fake (tests/_fake_audit.py) in this process so the
Physics-check UI can still be exercised.  Usage: python tests/_serve_for_ui.py PORT"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from webapp import core_loader  # noqa: E402

if not core_loader._load_audit():
    import _fake_audit
    _fake_audit.install(core_loader)
    print("[serve_for_ui] real freeto.audit missing: using the FAKE audit", flush=True)
else:
    print("[serve_for_ui] using the REAL freeto.audit", flush=True)

import webapp.server as server  # noqa: E402

sys.argv = [sys.argv[0], "--host", "127.0.0.1", "--port", sys.argv[1]]
server.main()
