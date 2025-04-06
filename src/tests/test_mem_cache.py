import unittest

from src.files import MemoryCache


class TestMemoryCache(unittest.TestCase):

    def test_100_cache(self):
        cache = MemoryCache(max_items=3)
        cache.put(1, "A")
        cache.put(2, "B")
        cache.put(3, "C")

        self.assertTrue(cache.has(1))
        self.assertTrue(cache.has(2))
        self.assertTrue(cache.has(3))

        cache.put(4, "D")

        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))
        self.assertTrue(cache.has(3))
        self.assertTrue(cache.has(4))

    def test_200_cache_counts(self):
        cache = MemoryCache(max_items=3)
        cache.put(1, "A")
        cache.put(2, "B")
        cache.put(3, "C")

        self.assertIsNotNone(cache.get(1))
        self.assertIsNotNone(cache.get(2))
        self.assertIsNotNone(cache.get(3))
        self.assertEqual(3, cache.num_hits)
        self.assertEqual(0, cache.num_misses)

        cache.put(4, "D")

        self.assertIsNone(cache.get(1))
        self.assertIsNotNone(cache.get(4))  # 4 accessed earlier than 2 & 3, will be dropped first
        self.assertIsNotNone(cache.get(2))
        self.assertIsNotNone(cache.get(3))
        self.assertEqual(6, cache.num_hits)
        self.assertEqual(1, cache.num_misses)

        cache.put(5, "E")

        self.assertIsNone(cache.get(1))
        self.assertIsNone(cache.get(4))
        self.assertIsNotNone(cache.get(2))
        self.assertIsNotNone(cache.get(3))
        self.assertIsNotNone(cache.get(5))
        self.assertEqual(9, cache.num_hits)
        self.assertEqual(3, cache.num_misses)
