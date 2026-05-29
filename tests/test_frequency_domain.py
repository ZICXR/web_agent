import math
import unittest

import numpy as np
from numpy.fft import fft, fftfreq

from utils.frequency_domain import (
    compute_spectral_entropy,
    entropy_to_frequency,
    frequency_domain_report,
    probs_to_frequency,
    resample_signal,
    sliding_window_frequency,
    DEFAULT_RESAMPLE_LEN,
)


class TestResample(unittest.TestCase):
    """测试重采样函数。"""

    def test_same_length_unchanged(self):
        signal = np.random.rand(256)
        result = resample_signal(signal, 256)
        np.testing.assert_allclose(result, signal)

    def test_upsample_preserves_endpoints(self):
        signal = np.array([0.0, 1.0])
        result = resample_signal(signal, 100)
        self.assertAlmostEqual(result[0], 0.0, places=10)
        self.assertAlmostEqual(result[-1], 1.0, places=10)

    def test_downsample_preserves_endpoints(self):
        signal = np.linspace(0, 10, 200)
        result = resample_signal(signal, 50)
        self.assertAlmostEqual(result[0], 0.0, places=10)
        self.assertAlmostEqual(result[-1], 10.0, places=10)

    def test_linear_signal_upsample(self):
        """线性信号重采样后仍应为线性。"""
        signal = np.array([0.0, 5.0, 10.0])
        result = resample_signal(signal, 21)
        expected = np.linspace(0, 10, 21)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_single_element(self):
        result = resample_signal([3.14], 50)
        np.testing.assert_allclose(result, 3.14)

    def test_output_length(self):
        for src_len, tgt_len in [(30, 256), (500, 256), (100, 256)]:
            signal = np.random.rand(src_len)
            result = resample_signal(signal, tgt_len)
            self.assertEqual(len(result), tgt_len)


class TestResampleEliminatesLengthBias(unittest.TestCase):
    """验证重采样后不同长度的同类信号产生可比的谱熵。"""

    def test_same_distribution_different_lengths(self):
        """同分布噪声的不同长度子序列，重采样后谱熵应接近。"""
        np.random.seed(42)
        full = np.random.randn(500)
        short = full[:100]
        long_ = full[:400]

        se_short = compute_spectral_entropy(short)
        se_long = compute_spectral_entropy(long_)
        # 两者都是白噪声，重采样后谱熵应相近（差异 < 20%）
        diff = abs(se_short - se_long) / max(se_short, se_long)
        self.assertLess(diff, 0.20)

    def test_resample_makes_freq_bins_identical(self):
        """不同长度的序列重采样后频率轴和 bins 数完全一致。"""
        a = np.random.rand(50)
        b = np.random.rand(300)

        res_a = entropy_to_frequency(a)
        res_b = entropy_to_frequency(b)

        self.assertEqual(len(res_a["freqs"]), len(res_b["freqs"]))
        np.testing.assert_allclose(res_a["freqs"], res_b["freqs"])


class TestEntropyToFrequency(unittest.TestCase):
    """测试熵序列频域变换。"""

    def test_output_keys(self):
        entropies = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = entropy_to_frequency(entropies)
        for key in ("spectrum", "magnitude", "phase", "power", "freqs", "original_len"):
            self.assertIn(key, result)

    def test_output_is_resampled_length(self):
        """输出长度应等于重采样目标长度。"""
        entropies = np.random.rand(50)
        result = entropy_to_frequency(entropies)
        self.assertEqual(len(result["magnitude"]), DEFAULT_RESAMPLE_LEN)
        self.assertEqual(result["original_len"], 50)

    def test_custom_resample_len(self):
        entropies = np.random.rand(80)
        result = entropy_to_frequency(entropies, resample_len=128)
        self.assertEqual(len(result["magnitude"]), 128)
        self.assertEqual(len(result["freqs"]), 128)

    def test_dc_removed(self):
        """频域变换前已去均值，直流分量应接近 0。"""
        entropies = np.ones(20) * 5.0
        result = entropy_to_frequency(entropies)
        np.testing.assert_allclose(result["magnitude"], 0.0, atol=1e-10)

    def test_single_frequency_signal(self):
        """纯正弦信号重采样后频域能量仍集中在对应频率附近。"""
        n = 128
        t = np.arange(n)
        entropies = np.sin(2 * np.pi * 4 / n * t)
        result = entropy_to_frequency(entropies)

        half = len(result["freqs"]) // 2
        power = result["power"][:half]
        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = result["freqs"][peak_idx]

        # 重采样后频率分辨率改变，峰值在最近 bin，允许一定偏移
        original_freq = 4.0 / n
        self.assertAlmostEqual(abs(peak_freq), original_freq, places=1)


class TestProbsToFrequency(unittest.TestCase):
    """测试概率矩阵频域变换。"""

    def test_output_shape(self):
        probs = np.random.dirichlet(np.ones(5), size=10).tolist()
        result = probs_to_frequency(probs)
        self.assertEqual(result["magnitude"].shape, (10, 5))
        self.assertEqual(result["phase"].shape, (10, 5))
        self.assertEqual(len(result["freqs"]), 5)

    def test_uniform_distribution(self):
        """均匀分布的 top-k 概率在频域应有特定的结构。"""
        k = 8
        probs = [[1.0 / k] * k for _ in range(20)]
        result = probs_to_frequency(probs)
        np.testing.assert_allclose(result["magnitude"][:, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["magnitude"][:, 1:], 0.0, atol=1e-10)


class TestSpectralEntropy(unittest.TestCase):
    """测试谱熵计算。"""

    def test_zero_signal(self):
        se = compute_spectral_entropy([0.0] * 50)
        self.assertEqual(se, 0.0)

    def test_constant_signal(self):
        se = compute_spectral_entropy([5.0] * 50)
        self.assertEqual(se, 0.0)

    def test_pure_tone_has_low_spectral_entropy(self):
        n = 256
        t = np.arange(n)
        tone = np.sin(2 * np.pi * 10 / n * t)
        se = compute_spectral_entropy(tone)
        self.assertLess(se, 2.0)

    def test_white_noise_has_high_spectral_entropy(self):
        np.random.seed(42)
        noise = np.random.randn(256)
        se = compute_spectral_entropy(noise)
        max_possible = math.log(DEFAULT_RESAMPLE_LEN // 2 - 1)
        self.assertGreater(se, max_possible * 0.7)

    def test_noise_higher_than_tone(self):
        np.random.seed(0)
        n = 256
        t = np.arange(n)
        tone = np.sin(2 * np.pi * 10 / n * t)
        noise = np.random.randn(n)
        self.assertGreater(compute_spectral_entropy(noise), compute_spectral_entropy(tone))


class TestSlidingWindowFrequency(unittest.TestCase):
    """测试滑动窗口频域分析。"""

    def test_output_length(self):
        entropies = np.random.rand(100)
        result = sliding_window_frequency(entropies, window_size=32)
        self.assertEqual(len(result), 100)

    def test_result_keys(self):
        entropies = np.random.rand(50)
        result = sliding_window_frequency(entropies, window_size=16)
        for item in result:
            for key in ("step", "spectral_entropy", "dominant_freq", "total_power"):
                self.assertIn(key, item)

    def test_short_sequence(self):
        entropies = [1.0, 2.0]
        result = sliding_window_frequency(entropies, window_size=8)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["spectral_entropy"], 0.0)

    def test_step_indices(self):
        entropies = np.random.rand(30)
        result = sliding_window_frequency(entropies)
        for i, item in enumerate(result):
            self.assertEqual(item["step"], i)

    def test_window_resampling(self):
        """不足 window_size 的窗口也应被正确重采样。"""
        entropies = np.random.rand(10)
        result = sliding_window_frequency(entropies, window_size=32)
        # 前几个窗口不足 32，但重采样后仍应有有效值
        self.assertGreater(result[5]["spectral_entropy"], 0.0)


class TestFrequencyDomainReport(unittest.TestCase):
    """测试完整报告生成。"""

    def test_report_without_probs(self):
        entropies = np.random.rand(50).tolist()
        report = frequency_domain_report(entropies)
        self.assertIn("entropy_freq", report)
        self.assertIn("spectral_entropy", report)
        self.assertIn("sliding_freq", report)
        self.assertIn("original_len", report)
        self.assertNotIn("probs_freq", report)

    def test_report_with_probs(self):
        entropies = np.random.rand(50).tolist()
        probs = np.random.dirichlet(np.ones(5), size=50).tolist()
        report = frequency_domain_report(entropies, top_k_probs=probs)
        self.assertIn("probs_freq", report)

    def test_spectral_entropy_is_float(self):
        entropies = np.random.rand(30).tolist()
        report = frequency_domain_report(entropies)
        self.assertIsInstance(report["spectral_entropy"], float)


if __name__ == "__main__":
    unittest.main()
