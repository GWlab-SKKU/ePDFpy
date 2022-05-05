import unittest
from unittest import TestCase, main


class MyTests(TestCase):
    def test_one_plus_two(self):
        self.assertEqual(1 + 2, 3)


if __name__ == '__main__':
    main()