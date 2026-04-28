from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_app_renders_without_runtime_exception(self) -> None:
        app = AppTest.from_file(str(Path("app.py").resolve()))
        app.run(timeout=120)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
