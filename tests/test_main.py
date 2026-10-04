import unittest

from main import greet


class GreetTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(greet(), "Hello, world!")

    def test_name(self):
        self.assertEqual(greet("MIK-GINGOS"), "Hello, MIK-GINGOS!")


if __name__ == "__main__":
    unittest.main()
