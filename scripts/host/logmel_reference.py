#!/usr/bin/env python3
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np


PROFILE_ID = "bounded-log-power-64-v1"
PROFILE_DOMAIN = "natural_log_mel_power"
PROFILE_LOWER_BOUND = -23.025850929940457
PROFILE_UPPER_BOUND = 5.9914645471079817
PROFILE_MEAN = -8.5171931914162382
PROFILE_STD = 14.508657738524219
NORMALIZED_BOUND_TOLERANCE = 1e-6


class FrontendConfig(NamedTuple):
    sample_rate_hz: int
    supported_channels: tuple
    window_samples: int
    stream_hop_samples: int
    frame_length: int
    hop_length: int
    fft_length: int
    expected_frames: int
    keep_frame_start: int
    keep_frame_count: int
    mel_bands: int
    mel_min_hz: float
    mel_max_hz: float
    log_floor: float
    profile_id: str
    profile_domain: str
    profile_lower_bound: float
    profile_upper_bound: float
    profile_mean: tuple
    profile_std: tuple
    tensor_shape: tuple
    source_sha256: str


class WavMetadata(NamedTuple):
    channels: int
    sample_rate_hz: int
    bits_per_sample: int
    total_samples: int
    data_offset: int
    data_bytes: int


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frontend_config(path: Path) -> FrontendConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["schema"] != 2:
            raise ValueError("unsupported schema")
        audio, stft = raw["audio"], raw["stft"]
        power, mel = raw["power_spectrum"], raw["mel"]
        log, tensor = raw["log"], raw["tensor"]
        normal, quant = raw["normalization"], raw["quantization"]
        lower_bound = _finite_number(normal["lower_bound"], "normalization.lower_bound")
        upper_bound = _finite_number(normal["upper_bound"], "normalization.upper_bound")
        mean = tuple(_finite_number(v, "normalization.mean") for v in normal["mean"])
        std = tuple(_finite_number(v, "normalization.std") for v in normal["std"])
        checks = (
            set(raw) == {
                "schema", "audio", "stft", "power_spectrum", "mel", "log",
                "tensor", "normalization", "quantization",
            },
            audio == {
                "sample_rate_hz": 16000, "sample_format": "s16le",
                "supported_channels": [1, 2], "window_samples": 16000,
                "stream_hop_samples": 4000,
            },
            stft == {
                "frame_length": 400, "hop_length": 160, "fft_length": 512,
                "window": "periodic_hann", "center": False,
                "expected_frames": 98, "keep_frame_start": 1,
                "keep_frame_count": 96,
            },
            power == {
                "formula": "magnitude_squared_div_fft_length", "divisor": 512.0,
                "single_sided_doubling": False,
                "window_energy_normalization": False,
            },
            mel == {
                "bands": 64, "min_hz": 20.0, "max_hz": 7600.0,
                "scale": "htk", "normalization": "none",
            },
            log == {"base": "natural", "floor": 1e-10},
            tensor == {
                "layout": "NCHW", "shape": [1, 1, 64, 96],
                "order": "mel_major_time_increasing", "dtype": "float32-le",
            },
            set(normal) == {"id", "domain", "lower_bound", "upper_bound", "mean", "std"},
            normal["id"] == PROFILE_ID,
            normal["domain"] == PROFILE_DOMAIN,
            lower_bound == PROFILE_LOWER_BOUND,
            upper_bound == PROFILE_UPPER_BOUND,
            len(mean) == 64 and all(value == PROFILE_MEAN for value in mean),
            len(std) == 64 and all(value == PROFILE_STD for value in std),
            PROFILE_MEAN == (lower_bound + upper_bound) / 2.0,
            PROFILE_STD == (upper_bound - lower_bound) / 2.0,
            lower_bound == math.log(float(log["floor"])),
            quant == {
                "dtype": "int8", "rounding": "half_away_from_zero",
                "clamp": [-128, 127],
                "scale_source": "runtime_dpu_input_tensor",
            },
        )
        if not all(checks):
            raise ValueError("configuration does not match D2 contract")
        if mel["max_hz"] > audio["sample_rate_hz"] / 2:
            raise ValueError("mel max exceeds Nyquist")
        if any(value <= 0.0 for value in std):
            raise ValueError("normalization std must be positive")
        return FrontendConfig(
            audio["sample_rate_hz"], tuple(audio["supported_channels"]),
            audio["window_samples"], audio["stream_hop_samples"],
            stft["frame_length"], stft["hop_length"], stft["fft_length"],
            stft["expected_frames"], stft["keep_frame_start"],
            stft["keep_frame_count"], mel["bands"], float(mel["min_hz"]),
            float(mel["max_hz"]), float(log["floor"]), normal["id"], normal["domain"],
            lower_bound, upper_bound, mean, std,
            tuple(tensor["shape"]), _sha256(path))
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError(f"invalid frontend configuration: {error}") from error


def _parse_pcm_wav(path: Path):
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    end = 8 + riff_size
    if end > len(data) or end < 12:
        raise ValueError("truncated RIFF payload")
    offset, fmt, audio = 12, None, None
    while offset < end:
        if end - offset < 8:
            raise ValueError("truncated chunk header")
        name = data[offset:offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        payload = offset + 8
        payload_end = payload + size
        padded_end = payload_end + (size & 1)
        if payload_end > end or padded_end > end:
            raise ValueError("truncated chunk payload")
        if name == b"fmt " and fmt is None:
            fmt = data[payload:payload_end]
        elif name == b"data" and audio is None:
            audio = (payload, size)
        offset = padded_end
    if fmt is None or audio is None or len(fmt) < 16:
        raise ValueError("missing fmt or data chunk")
    format_tag, channels, rate, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt, 0)
    if format_tag != 1 or channels not in (1, 2) or rate != 16000 or bits != 16:
        raise ValueError("unsupported PCM WAV format")
    expected_align = channels * 2
    if block_align != expected_align or byte_rate != rate * expected_align:
        raise ValueError("invalid PCM alignment")
    data_offset, data_bytes = audio
    if data_bytes % block_align:
        raise ValueError("partial PCM frame")
    metadata = WavMetadata(
        channels, rate, bits, data_bytes // block_align, data_offset, data_bytes)
    return data, metadata


def read_pcm_window(path: Path, start_sample: int, window_samples: int = 16000):
    if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
        raise ValueError("start_sample must be a non-negative integer")
    if isinstance(window_samples, bool) or not isinstance(window_samples, int) or window_samples <= 0:
        raise ValueError("window_samples must be a positive integer")
    data, metadata = _parse_pcm_wav(path)
    if start_sample + window_samples > metadata.total_samples:
        raise ValueError("requested WAV window is incomplete")
    frame_bytes = metadata.channels * 2
    begin = metadata.data_offset + start_sample * frame_bytes
    end = begin + window_samples * frame_bytes
    pcm = np.frombuffer(data[begin:end], dtype="<i2").reshape(window_samples, metadata.channels)
    if metadata.channels == 1:
        mono = pcm[:, 0].astype(np.float64) / 32768.0
    else:
        mono = pcm.astype(np.int32).sum(axis=1).astype(np.float64) / 65536.0
    return mono, metadata


def _round_half_away(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _float_to_pcm(values):
    scaled = _round_half_away(np.asarray(values, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype("<i2")


def synthesize_case(name: str, config: FrontendConfig):
    count = config.window_samples
    time = np.arange(count, dtype=np.float64) / config.sample_rate_hz
    channels = 1
    if name == "silence":
        samples = np.zeros(count, dtype="<i2")
    elif name == "impulse":
        samples = np.zeros(count, dtype="<i2")
        samples[count // 2] = 32767
    elif name == "sine_440":
        samples = _float_to_pcm(0.5 * np.sin(2.0 * np.pi * 440.0 * time))
    elif name == "dual_tone":
        wave = (0.35 * np.sin(2.0 * np.pi * 440.0 * time) +
                0.20 * np.sin(2.0 * np.pi * 1000.0 * time))
        samples = _float_to_pcm(wave)
    elif name == "noise_seeded":
        state, values = 20261010, np.empty(count, dtype=np.int16)
        for index in range(count):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            values[index] = (((state >> 16) & 0xFFFF) - 32768) // 3
        samples = values.astype("<i2", copy=False)
    elif name == "clipping":
        samples = np.empty(count, dtype="<i2")
        samples[0::2], samples[1::2] = -32768, 32767
    elif name == "stereo_antiphase":
        mono = _float_to_pcm(0.5 * np.sin(2.0 * np.pi * 523.25 * time))
        samples = np.empty(count * 2, dtype="<i2")
        samples[0::2], samples[1::2] = mono, -mono
        channels = 2
    else:
        raise ValueError(f"unknown synthetic case: {name}")
    return samples, {
        "name": name, "channels": channels, "sample_rate_hz": config.sample_rate_hz,
        "bits_per_sample": 16, "frames": count,
        "seed": 20261010 if name == "noise_seeded" else None,
    }


def encode_pcm_wav(samples, channels: int, sample_rate_hz: int = 16000):
    samples = np.asarray(samples, dtype="<i2")
    if channels not in (1, 2) or samples.size % channels:
        raise ValueError("invalid PCM channel layout")
    audio = samples.tobytes(order="C")
    block_align = channels * 2
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate_hz,
                      sample_rate_hz * block_align, block_align, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(audio)) + audio
    return b"RIFF" + struct.pack("<I", len(body)) + body


def periodic_hann(length: int):
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Hann length must be a positive integer")
    index = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * index / length)


def frame_audio(samples, config: FrontendConfig):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or samples.size != config.window_samples:
        raise ValueError("audio window must contain exactly 16000 mono samples")
    if not np.isfinite(samples).all():
        raise ValueError("audio samples must be finite")
    frames = np.lib.stride_tricks.sliding_window_view(samples, config.frame_length)
    frames = frames[::config.hop_length]
    if frames.shape != (config.expected_frames, config.frame_length):
        raise ValueError("unexpected STFT frame count")
    return np.array(frames, dtype=np.float64, copy=True)


def compute_power_spectrum(frames, config: FrontendConfig, apply_window=True):
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != config.frame_length:
        raise ValueError("frames must have shape [N,400]")
    if not np.isfinite(frames).all():
        raise ValueError("frames must be finite")
    values = frames * periodic_hann(config.frame_length) if apply_window else frames
    spectrum = np.fft.rfft(values, n=config.fft_length, axis=1)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag)
    power /= float(config.fft_length)
    return power


def _hz_to_htk_mel(frequency):
    return 2595.0 * np.log10(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def _htk_mel_to_hz(value):
    return 700.0 * (np.power(10.0, np.asarray(value, dtype=np.float64) / 2595.0) - 1.0)


def build_mel_filterbank(config: FrontendConfig):
    if (not math.isfinite(config.mel_min_hz) or not math.isfinite(config.mel_max_hz) or
            config.mel_min_hz < 0.0 or config.mel_max_hz <= config.mel_min_hz or
            config.mel_max_hz > config.sample_rate_hz / 2.0):
        raise ValueError("invalid Mel frequency range")
    mel_points = np.linspace(
        _hz_to_htk_mel(config.mel_min_hz), _hz_to_htk_mel(config.mel_max_hz),
        config.mel_bands + 2, dtype=np.float64)
    hz_points = _htk_mel_to_hz(mel_points)
    frequencies = np.arange(config.fft_length // 2 + 1, dtype=np.float64)
    frequencies *= config.sample_rate_hz / float(config.fft_length)
    filters = np.zeros((config.mel_bands, frequencies.size), dtype=np.float64)
    for band in range(config.mel_bands):
        left, center, right = hz_points[band:band + 3]
        left_width, right_width = center - left, right - center
        if left_width <= 0.0 or right_width <= 0.0:
            raise ValueError("degenerate Mel triangle")
        rising = (frequencies - left) / left_width
        falling = (right - frequencies) / right_width
        filters[band] = np.maximum(0.0, np.minimum(rising, falling))
    if not np.isfinite(filters).all():
        raise ValueError("non-finite Mel filterbank")
    return filters


def log_crop_normalize(mel_energies, config: FrontendConfig):
    energies = np.asarray(mel_energies, dtype=np.float64)
    expected = (config.mel_bands, config.expected_frames)
    if energies.shape != expected or not np.isfinite(energies).all() or np.any(energies < 0.0):
        raise ValueError("Mel energies must be finite, non-negative and shaped [64,98]")
    logged = np.log(np.maximum(energies, config.log_floor))
    start, stop = config.keep_frame_start, config.keep_frame_start + config.keep_frame_count
    kept = logged[:, start:stop]
    mean = np.asarray(config.profile_mean, dtype=np.float64)[:, None]
    std = np.asarray(config.profile_std, dtype=np.float64)[:, None]
    normalized = (kept - mean) / std
    if normalized.shape != (64, 96) or not np.isfinite(normalized).all():
        raise ValueError("invalid normalized Log-Mel tensor")
    if (np.any(normalized < -1.0 - NORMALIZED_BOUND_TOLERANCE) or
            np.any(normalized > 1.0 + NORMALIZED_BOUND_TOLERANCE)):
        raise ValueError("normalized Log-Mel tensor exceeds D2 bounds")
    return normalized.astype("<f4")


def compute_logmel(samples, config: FrontendConfig):
    frames = frame_audio(samples, config)
    power = compute_power_spectrum(frames, config, apply_window=True)
    mel_energies = build_mel_filterbank(config).dot(power.T)
    return log_crop_normalize(mel_energies, config)


def quantize_int8(features, input_scale: float):
    features = np.asarray(features)
    scale = _finite_number(input_scale, "input_scale")
    if scale <= 0.0:
        raise ValueError("input_scale must be positive")
    if features.shape != (64, 96) or not np.isfinite(features).all():
        raise ValueError("features must be finite and shaped [64,96]")
    scaled = features.astype(np.float64) * scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return np.clip(rounded, -128, 127).astype(np.int8)
