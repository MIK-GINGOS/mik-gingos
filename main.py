"""mik-gingos: a small Python program with its tests in the same file.

Run the program:  python3 main.py
Run the tests:    python3 main.py --test
"""

import sys
import unittest


def greet(name: str = "world") -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"


def main() -> None:
    print(greet())


class GreetTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(greet(), "Hello, world!")

    def test_name(self):
        self.assertEqual(greet("MIK-GINGOS"), "Hello, MIK-GINGOS!")


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0], "-v"])
    else:
        main()
