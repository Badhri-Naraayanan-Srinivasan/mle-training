import subprocess
import unittest


class TestInstallation(unittest.TestCase):
    def test_installation(self):
        res = subprocess.getoutput("pip install src")
        res = "src" in res
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
