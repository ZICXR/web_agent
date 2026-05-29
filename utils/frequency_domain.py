import numpy as np
from numpy.fft import fft, fftfreq


# 默认重采样目标长度，所有序列在频域分析前统一到这个长度
DEFAULT_RESAMPLE_LEN = 256


def resample_signal(signal, target_len):
    """将一维信号重采样到固定长度（线性插值）。"""
    signal = np.asarray(signal, dtype=float)
    if len(signal) == target_len:
        return signal
    if len(signal) < 2:
        return np.full(target_len, signal[0] if len(signal) == 1 else 0.0)
    old_indices = np.linspace(0, 1, len(signal))
    new_indices = np.linspace(0, 1, target_len)
    return np.interp(new_indices, old_indices, signal)


def _get_pos_power(entropies, resample_len=None):
    """获取正频率部分的归一化功率谱和频率轴。"""
    result = entropy_to_frequency(entropies, resample_len=resample_len)
    n = len(result["power"])
    half = n // 2
    power = result["power"][1:half]
    freqs = result["freqs"][1:half]
    return power, freqs, result


def probs_to_frequency(probs_list):
    """将每个 token 位置的 top-k 概率序列变换到频域。"""
    probs_array = np.array(probs_list)
    spectrum = fft(probs_array, axis=1)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n_freq = probs_array.shape[1]
    freqs = fftfreq(n_freq)
    return {"magnitude": magnitude, "phase": phase, "freqs": freqs}


def entropy_to_frequency(entropies, resample_len=None):
    """将 token 级别的熵序列变换到频域。先重采样到固定长度再做 FFT。"""
    entropies = np.asarray(entropies, dtype=float)
    original_len = len(entropies)
    target_len = resample_len or DEFAULT_RESAMPLE_LEN
    resampled = resample_signal(entropies, target_len)
    centered = resampled - np.mean(resampled)
    spectrum = fft(centered)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    power = magnitude ** 2
    freqs = fftfreq(target_len)
    return {
        "spectrum": spectrum,
        "magnitude": magnitude,
        "phase": phase,
        "power": power,
        "freqs": freqs,
        "original_len": original_len,
    }


# ============================================================
# 天然归一化的频域统计量（对序列长度不敏感）
# ============================================================

def compute_spectral_flatness(entropies, resample_len=None):
    """频谱平坦度 (Spectral Flatness / Wiener Entropy)。

    = 功率谱几何均值 / 功率谱算术均值
    恒在 [0, 1] 之间，完全不依赖序列长度。
    接近 1 = 类似白噪声（能量均匀），接近 0 = 类似单音（能量集中）。

    Returns:
        float: 0~1
    """
    power, _, _ = _get_pos_power(entropies, resample_len)
    if np.sum(power) < 1e-12:
        return 0.0
    log_power = np.log(power + 1e-30)
    geo_mean = np.exp(np.mean(log_power))
    arith_mean = np.mean(power)
    return float(np.clip(geo_mean / (arith_mean + 1e-30), 0.0, 1.0))


def compute_spectral_centroid(entropies, resample_len=None):
    """频谱质心 (Spectral Centroid)。

    功率谱的加权平均频率，归一化到 [0, 0.5]（Nyquist = 0.5）。
    衡量能量集中在高频还是低频。

    Returns:
        float: 归一化频率，0~0.5
    """
    power, freqs, _ = _get_pos_power(entropies, resample_len)
    total = np.sum(power)
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * power) / total)


def compute_spectral_bandwidth(entropies, resample_len=None):
    """频谱带宽 (Spectral Bandwidth)。

    功率谱围绕质心的加权标准差，归一化到 [0, 0.5]。
    衡量能量在频域上的集中或分散程度。

    Returns:
        float: 归一化带宽
    """
    power, freqs, _ = _get_pos_power(entropies, resample_len)
    total = np.sum(power)
    if total < 1e-12:
        return 0.0
    centroid = np.sum(freqs * power) / total
    bandwidth = np.sqrt(np.sum(power * (freqs - centroid) ** 2) / total)
    return float(bandwidth)


def compute_spectral_rolloff(entropies, threshold=0.85, resample_len=None):
    """频谱滚降点 (Spectral Rolloff)。

    累积功率达到总功率 threshold% 的频率，归一化到 [0, 0.5]。
    反映能量主要分布的频率范围有多宽。

    Returns:
        float: 归一化频率
    """
    power, freqs, _ = _get_pos_power(entropies, resample_len)
    total = np.sum(power)
    if total < 1e-12:
        return 0.0
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, threshold * total)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    return float(freqs[rolloff_idx])


def compute_spectral_contrast(entropies, n_bands=6, resample_len=None):
    """频谱对比度 (Spectral Contrast)。

    将正频率分成 n_bands 个子带，计算每个子带中峰值与谷值的分贝差，
    再取均值。越高说明频谱"锯齿状"越明显（频段间差异大）。

    Returns:
        float: 平均频谱对比度 (dB)
    """
    power, freqs, _ = _get_pos_power(entropies, resample_len)
    if len(power) < n_bands * 2:
        return 0.0
    band_size = len(power) // n_bands
    contrasts = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else len(power)
        band = power[start:end]
        if len(band) < 2:
            continue
        peak = np.percentile(band, 95)
        valley = np.percentile(band, 5)
        if valley < 1e-30:
            valley = 1e-30
        contrast_db = 10 * np.log10(peak / valley + 1e-30)
        contrasts.append(contrast_db)
    return float(np.mean(contrasts)) if contrasts else 0.0


def compute_slope_1f(entropies, resample_len=None):
    """1/f 斜率 (Power Spectrum Slope)。

    对 log(power) vs log(freq) 做线性回归，得到斜率。
    白噪声斜率 ≈ 0，1/f 噪声斜率 ≈ -1，1/f^2 斜率 ≈ -2。
    不受长度影响，因为是在对数空间做回归。

    Returns:
        float: 回归斜率
    """
    power, freqs, _ = _get_pos_power(entropies, resample_len)
    # 只用功率 > 0 的点
    mask = power > 1e-30
    if np.sum(mask) < 3:
        return 0.0
    log_f = np.log(freqs[mask] + 1e-30)
    log_p = np.log(power[mask])
    slope = np.polyfit(log_f, log_p, 1)[0]
    return float(slope)


def compute_spectral_entropy(entropies, resample_len=None):
    """谱熵 (Spectral Entropy)。功率谱归一化为概率分布后的 Shannon 熵。"""
    power, _, _ = _get_pos_power(entropies, resample_len)
    total = np.sum(power)
    if total < 1e-12:
        return 0.0
    psd_norm = power / total
    return float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))


def compute_all_spectral_features(entropies, resample_len=None):
    """一次性计算所有天然归一化的频域特征。

    Returns:
        dict: 所有特征值，每个都是标量，不受序列长度影响
    """
    return {
        "spectral_flatness": compute_spectral_flatness(entropies, resample_len),
        "spectral_centroid": compute_spectral_centroid(entropies, resample_len),
        "spectral_bandwidth": compute_spectral_bandwidth(entropies, resample_len),
        "spectral_rolloff": compute_spectral_rolloff(entropies, resample_len=resample_len),
        "spectral_contrast": compute_spectral_contrast(entropies, resample_len=resample_len),
        "slope_1f": compute_slope_1f(entropies, resample_len),
        "spectral_entropy": compute_spectral_entropy(entropies, resample_len),
    }


# ============================================================
# 滑动窗口 & 报告（保留原接口）
# ============================================================

def sliding_window_frequency(entropies, window_size=32):
    """对熵序列做滑动窗口频域分析。"""
    entropies = np.asarray(entropies, dtype=float)
    n = len(entropies)
    results = []

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = entropies[start : i + 1]

        if len(window) < 4:
            results.append({
                "step": i,
                "spectral_entropy": 0.0,
                "spectral_flatness": 0.0,
                "dominant_freq": 0.0,
                "total_power": 0.0,
            })
            continue

        resampled = resample_signal(window, window_size)
        centered = resampled - np.mean(resampled)
        spectrum = fft(centered)
        power = np.abs(spectrum) ** 2
        freqs = fftfreq(window_size)

        half = window_size // 2
        pos_power = power[1:half]
        pos_freqs = freqs[1:half]

        if len(pos_power) > 0:
            dominant_idx = np.argmax(pos_power)
            dominant_freq = float(pos_freqs[dominant_idx])
        else:
            dominant_freq = 0.0

        total = np.sum(pos_power)
        se = 0.0
        sf = 0.0
        if total > 1e-12:
            psd_norm = pos_power / total
            se = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
            log_p = np.log(pos_power + 1e-30)
            geo_mean = np.exp(np.mean(log_p))
            arith_mean = np.mean(pos_power)
            sf = float(np.clip(geo_mean / (arith_mean + 1e-30), 0.0, 1.0))

        results.append({
            "step": i,
            "spectral_entropy": se,
            "spectral_flatness": sf,
            "dominant_freq": dominant_freq,
            "total_power": float(np.sum(power)),
        })

    return results


def frequency_domain_report(entropies, top_k_probs=None, resample_len=None):
    """生成完整的频域分析报告。"""
    report = {}

    freq_result = entropy_to_frequency(entropies, resample_len=resample_len)
    report["entropy_freq"] = {
        "magnitude": freq_result["magnitude"],
        "phase": freq_result["phase"],
        "power": freq_result["power"],
        "freqs": freq_result["freqs"],
    }
    report["original_len"] = freq_result["original_len"]

    report["spectral_entropy"] = compute_spectral_entropy(entropies, resample_len=resample_len)
    report["sliding_freq"] = sliding_window_frequency(entropies)
    report["features"] = compute_all_spectral_features(entropies, resample_len=resample_len)

    if top_k_probs is not None:
        report["probs_freq"] = probs_to_frequency(top_k_probs)

    return report
