import subprocess
import sys


def test_import_does_not_import_matplotlib():
    code = """
import sys
sys.modules.pop("matplotlib", None)
import pydoppler
assert "matplotlib" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)

