import unittest

import numpy as np

from motionpngtuber.lipsync_core import AudioChunkBuffer


class AudioChunkBufferTests(unittest.TestCase):
    def test_tail_returns_latest_samples_across_chunk_boundaries(self):
        buf = AudioChunkBuffer(max_samples=10)
        buf.append(np.array([1, 2, 3], dtype=np.float32))
        buf.append(np.array([4, 5, 6, 7], dtype=np.float32))
        buf.append(np.array([8, 9], dtype=np.float32))

        got = buf.tail(5)
        want = np.array([5, 6, 7, 8, 9], dtype=np.float32)
        np.testing.assert_array_equal(got, want)

    def test_buffer_trims_oldest_samples_when_capacity_exceeded(self):
        buf = AudioChunkBuffer(max_samples=6)
        buf.append(np.array([1, 2, 3], dtype=np.float32))
        buf.append(np.array([4, 5, 6], dtype=np.float32))
        buf.append(np.array([7, 8], dtype=np.float32))

        self.assertEqual(len(buf), 6)
        np.testing.assert_array_equal(
            buf.tail(6),
            np.array([3, 4, 5, 6, 7, 8], dtype=np.float32),
        )

    def test_large_chunk_keeps_only_latest_capacity(self):
        buf = AudioChunkBuffer(max_samples=4)
        buf.append(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))

        self.assertEqual(len(buf), 4)
        np.testing.assert_array_equal(
            buf.tail(4),
            np.array([3, 4, 5, 6], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
