"""Put ``analysis/src`` on ``sys.path`` for the tests under ``tests/``.

Core modules (``modeling``, ``modelvalues``, ...) are imported by bare name but are
not installed as a package, so pytest needs this directory on the path explicitly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
