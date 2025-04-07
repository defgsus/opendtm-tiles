import unittest
import tempfile
from pathlib import Path

from src.files import DeleteFileOnException


class TestFileUtil(unittest.TestCase):

    def test_100_delete_file_on_exception(self):
        with tempfile.TemporaryDirectory("opendtm-unittest") as base_path:
            base_path = Path(base_path)

            fn = base_path / "a.txt"
            with DeleteFileOnException(fn):
                fn.write_text("A")

            self.assertTrue(fn.exists())

            fn = base_path / "b.txt"
            with self.assertRaises(ZeroDivisionError):
                with DeleteFileOnException(fn):
                    fn.write_text("B")
                    1 / 0

            self.assertFalse(fn.exists())

            fn = base_path / "c.txt"
            with self.assertRaises(KeyboardInterrupt):
                with DeleteFileOnException(fn):
                    with fn.open("wt") as fp:
                        raise KeyboardInterrupt()

            self.assertFalse(fn.exists())
