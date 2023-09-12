import unittest


class TestTestCase(unittest.TestCase):
    def test_assert(self):
        self.assertEqual(0, 0)
        self.assertNotEqual(0, 1)
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertIs(self, self)
        self.assertIsNot(self, self.test_assert)
        self.assertIsNone(None)
        self.assertIsNotNone(self)
        self.assertIn(0, range(1))
        self.assertNotIn(1, range(1))
        self.assertIsInstance(self, TestTestCase)
        self.assertNotIsInstance(self, dict)
