"""Benchmark a float32 MLP against a GF(137) modular network.

This is a small mathematical simulator, not an AGI implementation. The task is
intentionally synthetic: classify primality from the first fractional base-e
digits of integers. Previous audits found no strong prime/composite signal in
these digits, so near-chance accuracy is a valid outcome.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "modular_agi_benchmark.md"
CPP_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "cpp_kernel_benchmark.md"
TABLE_PATH = PROJECT_ROOT / "outputs" / "tables" / "modular_agi_benchmark_metrics.csv"

P_FIELD = 137
I5 = 42
INV_P_FIELD_FLOAT32 = np.float32(1.0 / P_FIELD)


try:
    from cosmo_gradient.modular_core import (
        COMPILE_LOG as CPP_COMPILE_LOG,
        compile_cpp_kernel,
        fused_two_layer_mod137_predict_into as cpp_fused_two_layer_mod137_predict_into,
        fused_two_layer_mod137_predict as cpp_fused_two_layer_mod137_predict,
        fused_two_layer_mod137_predict_repeated as cpp_fused_two_layer_mod137_predict_repeated,
        gf137_thread_count,
    )

    CPP_KERNEL_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    CPP_COMPILE_LOG = PROJECT_ROOT / "build" / "modular_core" / "compile.log"
    CPP_KERNEL_AVAILABLE = False


try:
    import torch  # type: ignore
    from torch import nn as NN  # type: ignore

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

    class _FallbackModule:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class NN:  # type: ignore[no-redef]
        Module = _FallbackModule


try:
    from numba import njit  # type: ignore

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


try:
    import mlx.core as mx  # type: ignore

    MLX_AVAILABLE = True
except ModuleNotFoundError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore[assignment]


try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    JAX_AVAILABLE = True
except ModuleNotFoundError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_all: np.ndarray
    y_all: np.ndarray
    numbers: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    memory_kb: float
    forward_1000_ms: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    notes: str


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = math.isqrt(value)
    factor = 5
    while factor <= limit:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def primes_in_range(start: int, stop: int) -> list[int]:
    return [value for value in range(start, stop + 1) if is_prime(value)]


def composites_in_range(start: int, stop: int) -> list[int]:
    return [value for value in range(start, stop + 1) if value > 1 and not is_prime(value)]


def precompute_e_weights(max_exponent: int, frac_digits: int, precision: int) -> dict[int, Decimal]:
    with localcontext() as ctx:
        ctx.prec = precision
        return {
            exponent: +Decimal(exponent).exp()
            for exponent in range(max_exponent, -frac_digits - 1, -1)
        }


def fractional_base_e_digits(
    number: int,
    *,
    frac_digits: int,
    precision: int,
    weights: dict[int, Decimal],
) -> tuple[int, ...]:
    max_exponent = int(math.floor(math.log(number)))
    with localcontext() as ctx:
        ctx.prec = precision
        remainder = Decimal(number)
        fractional: list[int] = []
        for exponent in range(max_exponent, -frac_digits - 1, -1):
            weight = weights[exponent]
            digit = int((remainder / weight).to_integral_value(rounding=ROUND_FLOOR))
            if digit < 0 or digit > 2:
                raise ArithmeticError(
                    f"Base-e digit outside alphabet {{0,1,2}}: "
                    f"number={number}, exponent={exponent}, digit={digit}"
                )
            remainder -= Decimal(digit) * weight
            if exponent < 0:
                fractional.append(digit)
        return tuple(fractional)


def make_dataset(
    *,
    n_samples: int,
    frac_digits: int,
    seed: int,
    start: int,
    stop: int,
    precision: int,
) -> DatasetSplit:
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for balanced prime/composite sampling.")
    n_each = n_samples // 2
    primes = primes_in_range(start, stop)
    if len(primes) < n_each:
        raise ValueError(f"Need {n_each} primes, found {len(primes)}.")
    selected_primes = primes[:n_each]
    composites = composites_in_range(min(selected_primes), max(selected_primes))
    if len(composites) < n_each:
        raise ValueError(f"Need {n_each} composites, found {len(composites)}.")

    rng = np.random.default_rng(seed)
    selected_composites = sorted(
        int(value) for value in rng.choice(composites, size=n_each, replace=False)
    )

    numbers = np.asarray(selected_primes + selected_composites, dtype=np.int64)
    labels = np.asarray([1] * n_each + [0] * n_each, dtype=np.float32)
    max_exponent = int(math.floor(math.log(int(numbers.max()))))
    weights = precompute_e_weights(max_exponent, frac_digits, precision)
    features = np.asarray(
        [
            fractional_base_e_digits(
                int(number),
                frac_digits=frac_digits,
                precision=precision,
                weights=weights,
            )
            for number in numbers
        ],
        dtype=np.uint8,
    )

    order = rng.permutation(len(numbers))
    features = features[order]
    labels = labels[order]
    numbers = numbers[order]

    train_end = int(0.70 * n_samples)
    val_end = int(0.80 * n_samples)
    return DatasetSplit(
        x_train=features[:train_end],
        y_train=labels[:train_end],
        x_val=features[train_end:val_end],
        y_val=labels[train_end:val_end],
        x_test=features[val_end:],
        y_test=labels[val_end:],
        x_all=features,
        y_all=labels,
        numbers=numbers,
    )


class Float32MLP:
    def __init__(self, in_features: int, hidden_features: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(0.0, 0.18, size=(in_features, hidden_features))).astype(np.float32)
        self.b1 = np.zeros(hidden_features, dtype=np.float32)
        self.w2 = (rng.normal(0.0, 0.18, size=(hidden_features, 1))).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_float = x.astype(np.float32) / 2.0
        hidden_pre = x_float @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ self.w2 + self.b2
        return self._sigmoid(logits).reshape(-1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, lr: float) -> None:
        x_float = x.astype(np.float32) / 2.0
        y_col = y.astype(np.float32).reshape(-1, 1)
        n = np.float32(len(x_float))
        for _ in range(epochs):
            hidden_pre = x_float @ self.w1 + self.b1
            hidden = np.maximum(hidden_pre, 0.0)
            logits = hidden @ self.w2 + self.b2
            pred = self._sigmoid(logits)
            d_logits = (pred - y_col) / n
            grad_w2 = hidden.T @ d_logits
            grad_b2 = np.sum(d_logits, axis=0)
            grad_hidden = d_logits @ self.w2.T
            grad_hidden_pre = grad_hidden * (hidden_pre > 0.0)
            grad_w1 = x_float.T @ grad_hidden_pre
            grad_b1 = np.sum(grad_hidden_pre, axis=0)
            self.w1 -= lr * grad_w1.astype(np.float32)
            self.b1 -= lr * grad_b1.astype(np.float32)
            self.w2 -= lr * grad_w2.astype(np.float32)
            self.b2 -= lr * grad_b2.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.forward(x) >= 0.5).astype(np.float32)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return self.w1.nbytes + self.b1.nbytes + self.w2.nbytes + self.b2.nbytes


class ModularLayer(NN.Module):
    """Linear layer over Z/137Z with I5-threshold activation support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        seed: int,
        modulus: int = P_FIELD,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modulus = modulus
        rng = np.random.default_rng(seed)
        self.weights = rng.integers(0, modulus, size=(in_features, out_features), dtype=np.uint8)
        self.bias = rng.integers(0, modulus, size=(out_features,), dtype=np.uint8)

    def set_weights_bias(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.weights = weights.astype(np.uint8, copy=True)
        self.bias = bias.astype(np.uint8, copy=True)

    def forward_residue(self, x: np.ndarray) -> np.ndarray:
        residue = (x.astype(np.int64) @ self.weights.astype(np.int64) + self.bias.astype(np.int64))
        return (residue % self.modulus).astype(np.uint8)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.forward_residue(x) >= I5).astype(np.uint8)

    def memory_bytes(self) -> int:
        return self.weights.nbytes + self.bias.nbytes


def fast_positive_mod137_float32(values: np.ndarray) -> np.ndarray:
    """Exact modulo 137 for non-negative integer-valued float32 arrays.

    The dot products in this benchmark stay far below 2^24, so float32 stores
    all intermediate integer sums exactly. Replacing `np.remainder` with this
    floor-based reduction avoids a surprisingly expensive generic ufunc path.
    """

    return (values - np.floor(values * INV_P_FIELD_FLOAT32) * P_FIELD).astype(np.uint8)


class FastExactModularLayer(ModularLayer):
    """GF(137) layer using float32 BLAS for exact small-integer matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        seed: int,
        modulus: int = P_FIELD,
    ) -> None:
        super().__init__(in_features, out_features, seed=seed, modulus=modulus)
        self._refresh_float_cache()

    def _refresh_float_cache(self) -> None:
        self.weights_f32 = self.weights.astype(np.float32)
        self.bias_f32 = self.bias.astype(np.float32)

    def set_weights_bias(self, weights: np.ndarray, bias: np.ndarray) -> None:
        super().set_weights_bias(weights, bias)
        self._refresh_float_cache()

    def forward_residue(self, x: np.ndarray) -> np.ndarray:
        x_float = x.astype(np.float32, copy=False)
        return fast_positive_mod137_float32(x_float @ self.weights_f32 + self.bias_f32)

    def memory_bytes(self) -> int:
        # Active inference memory includes compact residues plus the float32
        # compute cache. The uint8 copy could be the deployment storage format.
        return (
            self.weights.nbytes
            + self.bias.nbytes
            + self.weights_f32.nbytes
            + self.bias_f32.nbytes
        )


class ModularNetwork:
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        seed: int,
        *,
        layer_cls: type[ModularLayer] = ModularLayer,
    ) -> None:
        self.hidden = layer_cls(in_features, hidden_features, seed=seed)
        self.output = layer_cls(hidden_features, 1, seed=seed + 1)
        self.threshold = I5
        self.greater_is_prime = True

    def _hidden_features(self, x: np.ndarray) -> np.ndarray:
        return self.hidden.forward(x)

    def residues(self, x: np.ndarray) -> np.ndarray:
        hidden = self._hidden_features(x)
        return self.residues_from_hidden(hidden)

    def residues_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        return self.output.forward_residue(hidden).reshape(-1)

    @staticmethod
    def _pred_from_residues(
        residues: np.ndarray,
        *,
        threshold: int,
        greater_is_prime: bool,
    ) -> np.ndarray:
        pred = residues >= threshold
        if not greater_is_prime:
            pred = ~pred
        return pred.astype(np.float32)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        candidates: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        h_train = self._hidden_features(x_train)
        h_val = self._hidden_features(x_val)
        best_score = -1.0
        best_weights = self.output.weights.copy()
        best_bias = self.output.bias.copy()
        best_threshold = self.threshold
        best_direction = self.greater_is_prime

        thresholds = np.arange(P_FIELD, dtype=np.uint8)
        for _ in range(candidates):
            weights = rng.integers(0, P_FIELD, size=self.output.weights.shape, dtype=np.uint8)
            bias = rng.integers(0, P_FIELD, size=self.output.bias.shape, dtype=np.uint8)
            residues = ((h_val.astype(np.int64) @ weights.astype(np.int64) + bias.astype(np.int64)) % P_FIELD).reshape(-1)
            for direction in (True, False):
                scores = [
                    float(np.mean(self._pred_from_residues(residues, threshold=int(th), greater_is_prime=direction) == y_val))
                    for th in thresholds
                ]
                index = int(np.argmax(scores))
                score = scores[index]
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_bias = bias.copy()
                    best_threshold = int(thresholds[index])
                    best_direction = direction

        # A final light perceptron pass on the training set keeps the operation modular.
        self.output.set_weights_bias(best_weights, best_bias)
        self.threshold = best_threshold
        self.greater_is_prime = best_direction
        for row, target in zip(h_train, y_train, strict=True):
            residues = self.residues_from_hidden(row.reshape(1, -1))
            pred = self._pred_from_residues(
                residues,
                threshold=self.threshold,
                greater_is_prime=self.greater_is_prime,
            )[0]
            if pred != target:
                sign = 1 if target > pred else -1
                update = (sign * row.astype(np.int16)).reshape(-1, 1)
                weights = ((self.output.weights.astype(np.int16) + update) % P_FIELD).astype(np.uint8)
                bias = ((self.output.bias.astype(np.int16) + sign) % P_FIELD).astype(np.uint8)
                self.output.set_weights_bias(weights, bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._pred_from_residues(
            self.residues(x),
            threshold=self.threshold,
            greater_is_prime=self.greater_is_prime,
        )

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return self.hidden.memory_bytes() + self.output.memory_bytes() + 2


class FastExactModularNetwork(ModularNetwork):
    """Fused inference network for exact GF(137) residues via float32 BLAS."""

    def __init__(self, in_features: int, hidden_features: int, seed: int) -> None:
        super().__init__(
            in_features,
            hidden_features,
            seed,
            layer_cls=FastExactModularLayer,
        )

    def residues(self, x: np.ndarray) -> np.ndarray:
        hidden_layer = self.hidden
        output_layer = self.output
        if not isinstance(hidden_layer, FastExactModularLayer) or not isinstance(
            output_layer,
            FastExactModularLayer,
        ):
            return super().residues(x)

        x_float = x.astype(np.float32, copy=False)
        hidden_raw = x_float @ hidden_layer.weights_f32 + hidden_layer.bias_f32
        hidden_residue = hidden_raw - np.floor(hidden_raw * INV_P_FIELD_FLOAT32) * P_FIELD
        hidden_activation = (hidden_residue >= I5).astype(np.float32)
        output_raw = hidden_activation @ output_layer.weights_f32 + output_layer.bias_f32
        output_residue = output_raw - np.floor(output_raw * INV_P_FIELD_FLOAT32) * P_FIELD
        return output_residue.reshape(-1)


if NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def numba_modular_predict(
        x: np.ndarray,
        hidden_weights: np.ndarray,
        hidden_bias: np.ndarray,
        output_weights: np.ndarray,
        output_bias: np.ndarray,
        threshold: int,
        greater_is_prime: bool,
    ) -> np.ndarray:
        n_samples = x.shape[0]
        in_features = x.shape[1]
        hidden_features = hidden_weights.shape[1]
        out = np.empty(n_samples, dtype=np.float32)
        hidden = np.empty(hidden_features, dtype=np.int16)
        for sample in range(n_samples):
            for h in range(hidden_features):
                acc = int(hidden_bias[h])
                for feature in range(in_features):
                    acc += int(x[sample, feature]) * int(hidden_weights[feature, h])
                hidden[h] = 1 if (acc % P_FIELD) >= I5 else 0

            acc_out = int(output_bias[0])
            for h in range(hidden_features):
                acc_out += int(hidden[h]) * int(output_weights[h, 0])
            residue = acc_out % P_FIELD
            pred = residue >= threshold
            if not greater_is_prime:
                pred = not pred
            out[sample] = 1.0 if pred else 0.0
        return out


class NumbaModularPredictor:
    def __init__(self, model: ModularNetwork) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available.")
        self.hidden_weights = model.hidden.weights.astype(np.int16)
        self.hidden_bias = model.hidden.bias.astype(np.int16)
        self.output_weights = model.output.weights.astype(np.int16)
        self.output_bias = model.output.bias.astype(np.int16)
        self.threshold = model.threshold
        self.greater_is_prime = model.greater_is_prime

    def predict(self, x: np.ndarray) -> np.ndarray:
        return numba_modular_predict(  # type: ignore[misc]
            x,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            self.threshold,
            self.greater_is_prime,
        )

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return (
            self.hidden_weights.nbytes
            + self.hidden_bias.nbytes
            + self.output_weights.nbytes
            + self.output_bias.nbytes
            + 2
        )


class MlxModularPredictor:
    def __init__(self, model: ModularNetwork) -> None:
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX is not available.")
        self.hidden_weights = mx.array(model.hidden.weights.astype(np.float32))  # type: ignore[union-attr]
        self.hidden_bias = mx.array(model.hidden.bias.astype(np.float32))  # type: ignore[union-attr]
        self.output_weights = mx.array(model.output.weights.astype(np.float32))  # type: ignore[union-attr]
        self.output_bias = mx.array(model.output.bias.astype(np.float32))  # type: ignore[union-attr]
        self.threshold = model.threshold
        self.greater_is_prime = model.greater_is_prime

    @staticmethod
    def _mod137(values):
        return values - mx.floor(values * float(INV_P_FIELD_FLOAT32)) * P_FIELD  # type: ignore[union-attr]

    def predict_mx(self, x_mx):
        hidden_raw = x_mx @ self.hidden_weights + self.hidden_bias
        hidden_residue = self._mod137(hidden_raw)
        hidden = (hidden_residue >= I5).astype(mx.float32)  # type: ignore[union-attr]
        output_raw = hidden @ self.output_weights + self.output_bias
        residue = self._mod137(output_raw).reshape(-1)
        pred = residue >= self.threshold
        if not self.greater_is_prime:
            pred = mx.logical_not(pred)  # type: ignore[union-attr]
        return pred.astype(mx.float32)  # type: ignore[union-attr]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_mx = mx.array(x.astype(np.float32, copy=False))  # type: ignore[union-attr]
        pred = self.predict_mx(x_mx)
        mx.eval(pred)  # type: ignore[union-attr]
        return np.asarray(pred)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return int(
            np.prod(self.hidden_weights.shape) * 4
            + np.prod(self.hidden_bias.shape) * 4
            + np.prod(self.output_weights.shape) * 4
            + np.prod(self.output_bias.shape) * 4
        )


class TorchModularPredictor:
    def __init__(self, model: ModularNetwork, *, compile_model: bool = False) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is not available.")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore[name-defined]
        self.hidden_weights = torch.tensor(  # type: ignore[name-defined]
            model.hidden.weights.astype(np.float32),
            device=self.device,
        )
        self.hidden_bias = torch.tensor(model.hidden.bias.astype(np.float32), device=self.device)  # type: ignore[name-defined]
        self.output_weights = torch.tensor(  # type: ignore[name-defined]
            model.output.weights.astype(np.float32),
            device=self.device,
        )
        self.output_bias = torch.tensor(model.output.bias.astype(np.float32), device=self.device)  # type: ignore[name-defined]
        self.threshold = float(model.threshold)
        self.greater_is_prime = model.greater_is_prime
        self.compiled = False
        if compile_model and hasattr(torch, "compile"):  # type: ignore[name-defined]
            try:
                self._forward_impl = torch.compile(self._forward_impl, mode="reduce-overhead")  # type: ignore[name-defined,method-assign]
                self.compiled = True
            except Exception:
                self.compiled = False

    @staticmethod
    def _mod137(values):
        return values - torch.floor(values * float(INV_P_FIELD_FLOAT32)) * P_FIELD  # type: ignore[name-defined]

    def _forward_impl(self, x_tensor):
        hidden_raw = x_tensor @ self.hidden_weights + self.hidden_bias
        hidden_residue = self._mod137(hidden_raw)
        hidden = (hidden_residue >= I5).to(torch.float32)  # type: ignore[name-defined]
        output_raw = hidden @ self.output_weights + self.output_bias
        residue = self._mod137(output_raw).reshape(-1)
        pred = residue >= self.threshold
        if not self.greater_is_prime:
            pred = torch.logical_not(pred)  # type: ignore[name-defined]
        return pred.to(torch.float32)  # type: ignore[name-defined]

    def predict_tensor(self, x_tensor):
        return self._forward_impl(x_tensor)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x.astype(np.float32, copy=False), device=self.device)  # type: ignore[name-defined]
        pred = self.predict_tensor(x_tensor)
        if self.device == "mps":
            torch.mps.synchronize()  # type: ignore[name-defined]
        return pred.detach().cpu().numpy()

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return (
            self.hidden_weights.numel() * self.hidden_weights.element_size()
            + self.hidden_bias.numel() * self.hidden_bias.element_size()
            + self.output_weights.numel() * self.output_weights.element_size()
            + self.output_bias.numel() * self.output_bias.element_size()
        )


class JaxModularPredictor:
    def __init__(self, model: ModularNetwork) -> None:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available.")
        self.hidden_weights = jnp.asarray(model.hidden.weights.astype(np.float32))  # type: ignore[union-attr]
        self.hidden_bias = jnp.asarray(model.hidden.bias.astype(np.float32))  # type: ignore[union-attr]
        self.output_weights = jnp.asarray(model.output.weights.astype(np.float32))  # type: ignore[union-attr]
        self.output_bias = jnp.asarray(model.output.bias.astype(np.float32))  # type: ignore[union-attr]
        self.threshold = float(model.threshold)
        self.greater_is_prime = model.greater_is_prime
        self._forward_jit = jax.jit(self._forward_impl)  # type: ignore[union-attr]

    @staticmethod
    def _mod137(values):
        return values - jnp.floor(values * float(INV_P_FIELD_FLOAT32)) * P_FIELD  # type: ignore[union-attr]

    def _forward_impl(self, x_array):
        hidden_raw = x_array @ self.hidden_weights + self.hidden_bias
        hidden_residue = self._mod137(hidden_raw)
        hidden = (hidden_residue >= I5).astype(jnp.float32)  # type: ignore[union-attr]
        output_raw = hidden @ self.output_weights + self.output_bias
        residue = self._mod137(output_raw).reshape(-1)
        pred = residue >= self.threshold
        if not self.greater_is_prime:
            pred = jnp.logical_not(pred)  # type: ignore[union-attr]
        return pred.astype(jnp.float32)  # type: ignore[union-attr]

    def predict_jax(self, x_array):
        return self._forward_jit(x_array)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = jnp.asarray(x.astype(np.float32, copy=False))  # type: ignore[union-attr]
        pred = self.predict_jax(x_array)
        pred.block_until_ready()
        return np.asarray(pred)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return int(
            np.prod(self.hidden_weights.shape) * 4
            + np.prod(self.hidden_bias.shape) * 4
            + np.prod(self.output_weights.shape) * 4
            + np.prod(self.output_bias.shape) * 4
        )


class CppFusedModularPredictor:
    """Native uint8 GF(137) inference using the C++ fused kernel."""

    def __init__(self, model: ModularNetwork) -> None:
        if not CPP_KERNEL_AVAILABLE:
            raise RuntimeError("C++ fused kernel binding is not available.")
        compile_cpp_kernel()
        self.hidden_weights = np.ascontiguousarray(model.hidden.weights, dtype=np.uint8)
        self.hidden_bias = np.ascontiguousarray(model.hidden.bias, dtype=np.uint8)
        self.output_weights = np.ascontiguousarray(model.output.weights.reshape(-1), dtype=np.uint8)
        self.output_bias = np.ascontiguousarray(model.output.bias.reshape(-1), dtype=np.uint8)
        self.threshold = model.threshold
        self.greater_is_prime = model.greater_is_prime
        self.thread_count = gf137_thread_count()
        self._buffer: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return cpp_fused_two_layer_mod137_predict(
            x,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            hidden_threshold=I5,
            output_threshold=self.threshold,
            greater_is_prime=self.greater_is_prime,
        ).astype(np.float32)

    def predict_reused_buffer(self, x: np.ndarray) -> np.ndarray:
        if self._buffer is None or self._buffer.shape[0] != x.shape[0]:
            self._buffer = np.empty(x.shape[0], dtype=np.uint8)
        return cpp_fused_two_layer_mod137_predict_into(
            x,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            self._buffer,
            hidden_threshold=I5,
            output_threshold=self.threshold,
            greater_is_prime=self.greater_is_prime,
        )

    def time_repeated_native_ms(self, x: np.ndarray, iterations: int) -> float:
        if self._buffer is None or self._buffer.shape[0] != x.shape[0]:
            self._buffer = np.empty(x.shape[0], dtype=np.uint8)
        cpp_fused_two_layer_mod137_predict_repeated(
            x,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            self._buffer,
            1,
            hidden_threshold=I5,
            output_threshold=self.threshold,
            greater_is_prime=self.greater_is_prime,
        )
        start = time.perf_counter()
        cpp_fused_two_layer_mod137_predict_repeated(
            x,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            self._buffer,
            iterations,
            hidden_threshold=I5,
            output_threshold=self.threshold,
            greater_is_prime=self.greater_is_prime,
        )
        return (time.perf_counter() - start) * 1000.0

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def memory_bytes(self) -> int:
        return (
            self.hidden_weights.nbytes
            + self.hidden_bias.nbytes
            + self.output_weights.nbytes
            + self.output_bias.nbytes
            + 2
        )


def time_forward_ms(callable_forward, x: np.ndarray, iterations: int) -> float:
    callable_forward(x)
    start = time.perf_counter()
    for _ in range(iterations):
        callable_forward(x)
    return (time.perf_counter() - start) * 1000.0


def time_mlx_forward_ms(predictor: MlxModularPredictor, x: np.ndarray, iterations: int) -> float:
    x_mx = mx.array(x.astype(np.float32, copy=False))  # type: ignore[union-attr]
    pred = predictor.predict_mx(x_mx)
    mx.eval(pred)  # type: ignore[union-attr]
    start = time.perf_counter()
    for _ in range(iterations):
        pred = predictor.predict_mx(x_mx)
        mx.eval(pred)  # type: ignore[union-attr]
    return (time.perf_counter() - start) * 1000.0


def time_torch_forward_ms(
    predictor: TorchModularPredictor,
    x: np.ndarray,
    iterations: int,
) -> float:
    x_tensor = torch.tensor(x.astype(np.float32, copy=False), device=predictor.device)  # type: ignore[name-defined]
    pred = predictor.predict_tensor(x_tensor)
    if predictor.device == "mps":
        torch.mps.synchronize()  # type: ignore[name-defined]
    start = time.perf_counter()
    for _ in range(iterations):
        pred = predictor.predict_tensor(x_tensor)
    if predictor.device == "mps":
        torch.mps.synchronize()  # type: ignore[name-defined]
    return (time.perf_counter() - start) * 1000.0


def time_jax_forward_ms(predictor: JaxModularPredictor, x: np.ndarray, iterations: int) -> float:
    x_array = jnp.asarray(x.astype(np.float32, copy=False))  # type: ignore[union-attr]
    pred = predictor.predict_jax(x_array)
    pred.block_until_ready()
    start = time.perf_counter()
    for _ in range(iterations):
        pred = predictor.predict_jax(x_array)
    pred.block_until_ready()
    return (time.perf_counter() - start) * 1000.0


def benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    dataset = make_dataset(
        n_samples=args.samples,
        frac_digits=args.frac_digits,
        seed=args.seed,
        start=args.start,
        stop=args.stop,
        precision=args.precision,
    )

    float_model = Float32MLP(args.frac_digits, args.hidden, seed=args.seed + 10)
    float_model.fit(dataset.x_train, dataset.y_train, epochs=args.float_epochs, lr=args.float_lr)

    modular_model = ModularNetwork(args.frac_digits, args.hidden, seed=args.seed + 20)
    modular_model.fit(
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        candidates=args.modular_candidates,
        seed=args.seed + 30,
    )
    fast_modular_model = FastExactModularNetwork(
        args.frac_digits,
        args.hidden,
        seed=args.seed + 20,
    )
    fast_modular_model.fit(
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        candidates=args.modular_candidates,
        seed=args.seed + 30,
    )
    x_all_f32 = dataset.x_all.astype(np.float32)
    if not np.array_equal(modular_model.predict(dataset.x_all), fast_modular_model.predict(x_all_f32)):
        raise AssertionError("Fast exact modular path diverged from compact integer GF(137) path.")

    reference_pred = modular_model.predict(dataset.x_all)
    results = [
        BenchmarkResult(
            model="float32_mlp_numpy" if not TORCH_AVAILABLE else "float32_mlp_torch_available_but_numpy_used",
            memory_kb=float_model.memory_bytes() / 1024.0,
            forward_1000_ms=time_forward_ms(float_model.forward, dataset.x_all, args.forward_iterations),
            train_accuracy=float_model.accuracy(dataset.x_train, dataset.y_train),
            validation_accuracy=float_model.accuracy(dataset.x_val, dataset.y_val),
            test_accuracy=float_model.accuracy(dataset.x_test, dataset.y_test),
            notes="PyTorch not installed; NumPy float32 MLP baseline used."
            if not TORCH_AVAILABLE
            else "Torch is available, but this script keeps both models in NumPy for comparable timing.",
        ),
        BenchmarkResult(
            model="gf137_modular_network_compact_uint8",
            memory_kb=modular_model.memory_bytes() / 1024.0,
            forward_1000_ms=time_forward_ms(modular_model.predict, dataset.x_all, args.forward_iterations),
            train_accuracy=modular_model.accuracy(dataset.x_train, dataset.y_train),
            validation_accuracy=modular_model.accuracy(dataset.x_val, dataset.y_val),
            test_accuracy=modular_model.accuracy(dataset.x_test, dataset.y_test),
            notes=(
                f"GF({P_FIELD}) weights stored as uint8; activation is residue >= I5={I5}; "
                f"readout threshold={modular_model.threshold}, "
                f"greater_is_prime={modular_model.greater_is_prime}."
            ),
        ),
        BenchmarkResult(
            model="gf137_modular_network_fast_exact",
            memory_kb=fast_modular_model.memory_bytes() / 1024.0,
            forward_1000_ms=time_forward_ms(
                fast_modular_model.predict,
                x_all_f32,
                args.forward_iterations,
            ),
            train_accuracy=fast_modular_model.accuracy(dataset.x_train, dataset.y_train),
            validation_accuracy=fast_modular_model.accuracy(dataset.x_val, dataset.y_val),
            test_accuracy=fast_modular_model.accuracy(dataset.x_test, dataset.y_test),
            notes=(
                f"Exact GF({P_FIELD}) residues via float32 BLAS plus floor modulo; "
                "active memory includes uint8 storage and float32 compute cache."
            ),
        ),
    ]
    if CPP_KERNEL_AVAILABLE:
        cpp_predictor = CppFusedModularPredictor(modular_model)
        cpp_pred = cpp_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, cpp_pred):
            raise AssertionError("C++ fused modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model="gf137_modular_network_cpp_fused_uint8",
                memory_kb=cpp_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_forward_ms(
                    cpp_predictor.predict_reused_buffer,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=cpp_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=cpp_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=cpp_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes=(
                    "C++ ctypes fused kernel; shape-specialized ARM NEON fast path for H=32; "
                    f"threads={cpp_predictor.thread_count}. "
                    f"Compile log: {CPP_COMPILE_LOG}."
                ),
            )
        )
        results.append(
            BenchmarkResult(
                model="gf137_modular_network_cpp_neon_uint8_native_loop",
                memory_kb=cpp_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=cpp_predictor.time_repeated_native_ms(
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=cpp_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=cpp_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=cpp_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes=(
                    "C++ NEON throughput path; one native call runs all "
                    f"{args.forward_iterations} iterations; threads={cpp_predictor.thread_count}. "
                    "This measures production throughput rather than Python-call latency."
                ),
            )
        )
    if NUMBA_AVAILABLE:
        numba_predictor = NumbaModularPredictor(modular_model)
        numba_pred = numba_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, numba_pred):
            raise AssertionError("Numba modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model="gf137_modular_network_numba_fused",
                memory_kb=numba_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_forward_ms(
                    numba_predictor.predict,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=numba_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=numba_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=numba_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes="Fused Numba CPU kernel; matmul, modulo, threshold, and readout in one loop.",
            )
        )
    if MLX_AVAILABLE:
        mlx_predictor = MlxModularPredictor(modular_model)
        mlx_pred = mlx_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, mlx_pred):
            raise AssertionError("MLX modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model="gf137_modular_network_mlx_gpu",
                memory_kb=mlx_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_mlx_forward_ms(
                    mlx_predictor,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=mlx_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=mlx_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=mlx_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes="MLX GPU path on Apple Silicon; includes GPU synchronization each timed pass.",
            )
        )
    if TORCH_AVAILABLE:
        torch_predictor = TorchModularPredictor(modular_model, compile_model=False)
        torch_pred = torch_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, torch_pred):
            raise AssertionError("Torch modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model=f"gf137_modular_network_torch_{torch_predictor.device}_eager",
                memory_kb=torch_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_torch_forward_ms(
                    torch_predictor,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=torch_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=torch_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=torch_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes="Torch eager path using floor modulo; synchronization included for MPS.",
            )
        )
        compiled_torch_predictor = TorchModularPredictor(modular_model, compile_model=True)
        compiled_pred = compiled_torch_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, compiled_pred):
            raise AssertionError("Torch compiled modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model=(
                    f"gf137_modular_network_torch_{compiled_torch_predictor.device}_compile"
                    if compiled_torch_predictor.compiled
                    else f"gf137_modular_network_torch_{compiled_torch_predictor.device}_compile_unavailable"
                ),
                memory_kb=compiled_torch_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_torch_forward_ms(
                    compiled_torch_predictor,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=compiled_torch_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=compiled_torch_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=compiled_torch_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes=(
                    "Torch compile reduce-overhead path."
                    if compiled_torch_predictor.compiled
                    else "torch.compile unavailable or failed; fell back to eager implementation."
                ),
            )
        )
    if JAX_AVAILABLE:
        jax_predictor = JaxModularPredictor(modular_model)
        jax_pred = jax_predictor.predict(dataset.x_all)
        if not np.array_equal(reference_pred, jax_pred):
            raise AssertionError("JAX modular path diverged from compact integer GF(137) path.")
        results.append(
            BenchmarkResult(
                model="gf137_modular_network_jax_jit_cpu",
                memory_kb=jax_predictor.memory_bytes() / 1024.0,
                forward_1000_ms=time_jax_forward_ms(
                    jax_predictor,
                    dataset.x_all,
                    args.forward_iterations,
                ),
                train_accuracy=jax_predictor.accuracy(dataset.x_train, dataset.y_train),
                validation_accuracy=jax_predictor.accuracy(dataset.x_val, dataset.y_val),
                test_accuracy=jax_predictor.accuracy(dataset.x_test, dataset.y_test),
                notes="JAX JIT CPU path; JAX reports CPU only in this environment.",
            )
        )
    frame = pd.DataFrame([result.__dict__ for result in results])
    dataset_note = (
        f"Balanced dataset: {args.samples // 2} primes and {args.samples // 2} composites; "
        f"features are first {args.frac_digits} fractional base-e digits; "
        f"split is 70/10/20 train/validation/test."
    )
    return frame, dataset_note


def render_report(args: argparse.Namespace, metrics: pd.DataFrame, dataset_note: str) -> str:
    winner_memory = metrics.sort_values("memory_kb").iloc[0]
    winner_time = metrics.sort_values("forward_1000_ms").iloc[0]
    latency_metrics = metrics[
        ~metrics["model"].str.contains("native_loop", regex=False)
    ]
    winner_latency = latency_metrics.sort_values("forward_1000_ms").iloc[0]
    winner_accuracy = metrics.sort_values("test_accuracy", ascending=False).iloc[0]
    memory_min = float(winner_memory["memory_kb"])
    memory_winners = ", ".join(
        f"`{model}`"
        for model in metrics.loc[
            np.isclose(metrics["memory_kb"], memory_min),
            "model",
        ].tolist()
    )
    return f"""# Modular AGI Core Benchmark

## Scope

This is a small simulator comparing a conventional float32 MLP baseline with a
network whose weights and matrix products live in `Z/137Z`. It is not an AGI
demonstration. The classification task is intentionally difficult: previous
base-e digit audits did not find a robust prime/composite separator.

## Dataset

{dataset_note}

Sampling range: [{args.start}, {args.stop}], seed: {args.seed}.

## Modular Layer

```python
class ModularLayer(NN.Module):
    def forward_residue(self, x):
        residue = x @ W + b
        return residue % 137

    def forward(self, x):
        return (forward_residue(x) >= 42)
```

The optimized NumPy path uses the same residues but computes the small integer
matrix products through exact float32 BLAS and a fast floor reduction:

```python
y = x_float32 @ W_float32 + b_float32
residue = y - floor(y / 137) * 137
```

The sums in this benchmark are below `2^24`, so float32 represents the integer
intermediates exactly.

The C++ path stores only the compact `uint8` residues and fuses matrix
multiplication, Barrett reduction modulo 137, and threshold activation in one
native loop. The current C++ backend uses a persistent standard-library worker
pool to split rows across CPU threads without paying thread-spawn cost on every
forward pass. On ARM64, the `H=32` benchmark shape uses a specialized NEON path
for hidden accumulation, modulo-thresholding, and active output-weight summation.

The `cpp_neon_uint8_native_loop` row is a production-throughput check: it runs
all timed iterations inside one native C++ call and therefore removes the
per-call Python/ctypes boundary from the measurement.

Backend availability in this run:

- `numba_available = {NUMBA_AVAILABLE}`
- `mlx_available = {MLX_AVAILABLE}`
- `torch_available = {TORCH_AVAILABLE}`
- `jax_available = {JAX_AVAILABLE}`
- `cpp_kernel_available = {CPP_KERNEL_AVAILABLE}`
- `cpp_compile_log = {CPP_COMPILE_LOG}`

The float32 baseline remains a NumPy MLP so that the baseline does not depend
on optional runtime installation.

## Metrics

{metrics.to_markdown(index=False, floatfmt=".6g")}

## Winners

- Lowest memory footprint: {memory_winners} ({winner_memory['memory_kb']:.6g} KB)
- Fastest 1000 forward passes, throughput mode included: `{winner_time['model']}` ({winner_time['forward_1000_ms']:.6g} ms)
- Fastest Python-call latency path: `{winner_latency['model']}` ({winner_latency['forward_1000_ms']:.6g} ms)
- Highest test accuracy: `{winner_accuracy['model']}` ({winner_accuracy['test_accuracy']:.6g})

## Interpretation

The modular model is much smaller because its weights are stored as `uint8`
residues. The multithreaded C++ fused path keeps that compact storage and
removes most of the Python/NumPy integer-loop overhead, but the fastest backend
in this run may still be a JIT/BLAS path if vectorization dominates threading
overhead for this small dense workload. Accuracy should be read cautiously: if
the models are near chance, that is consistent with the earlier null result that
base-e fractional digits do not expose a clean primality invariant.

## What Would Be Faster Than NumPy Here?

For this exact GF(137) workload, the next realistic speed steps are not more
Python wrappers but a SIMD/tiled/multithreaded C++ kernel or a custom XLA-style
integer kernel. This script now records all optional backends that are
importable in the active environment.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--frac-digits", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--stop", type=int, default=5000)
    parser.add_argument("--precision", type=int, default=120)
    parser.add_argument("--float-epochs", type=int, default=500)
    parser.add_argument("--float-lr", type=float, default=0.06)
    parser.add_argument("--modular-candidates", type=int, default=2000)
    parser.add_argument("--forward-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Modular AGI Core benchmark simulator")
    print(
        "samples=%d frac_digits=%d hidden=%d field=GF(%d) I5=%d torch_available=%s"
        % (args.samples, args.frac_digits, args.hidden, P_FIELD, I5, TORCH_AVAILABLE)
    )
    print("generating balanced base-e digit dataset...")
    metrics, dataset_note = benchmark(args)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_PATH, index=False)
    report_text = render_report(args, metrics, dataset_note)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    CPP_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"wrote {TABLE_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {CPP_REPORT_PATH}")


if __name__ == "__main__":
    main()
