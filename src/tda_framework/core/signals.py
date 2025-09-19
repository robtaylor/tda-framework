"""Signal representation and management for continuous-time simulation."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


class SignalType(Enum):
    """Types of signals in the simulation."""
    VOLTAGE = "voltage"
    CURRENT = "current"
    DIGITAL = "digital"
    ANALOG = "analog"


@dataclass
class SignalSample:
    """A single sample of a signal at a specific time."""
    timestamp: Decimal
    value: float
    derivative: Optional[float] = None


class Signal:
    """Represents a continuous-time signal with timestamp-based state changes."""

    def __init__(
        self,
        name: str,
        signal_type: SignalType = SignalType.ANALOG,
        initial_value: float = 0.0,
        units: str = "V"
    ) -> None:
        """Initialize a signal with given parameters."""
        self.name = name
        self.signal_type = signal_type
        self.units = units
        self._samples: List[SignalSample] = []
        self._current_value = initial_value
        self._current_derivative: Optional[float] = None
        self._observers: List[Callable[[Signal, float], None]] = []

    @property
    def current_value(self) -> float:
        """Get current signal value."""
        return self._current_value

    @property
    def current_derivative(self) -> Optional[float]:
        """Get current signal derivative."""
        return self._current_derivative

    def set_value(
        self,
        timestamp: Decimal,
        value: float,
        derivative: Optional[float] = None
    ) -> None:
        """Set signal value at specified timestamp."""
        self._current_value = value
        self._current_derivative = derivative

        sample = SignalSample(timestamp, value, derivative)
        self._samples.append(sample)

        # Notify observers of signal change
        for observer in self._observers:
            observer(self, value)

    def get_value_at(self, timestamp: Decimal) -> float:
        """Get interpolated signal value at specific timestamp."""
        if not self._samples:
            return self._current_value

        # Find samples before and after timestamp
        before_sample = None
        after_sample = None

        for sample in self._samples:
            if sample.timestamp <= timestamp:
                before_sample = sample
            elif sample.timestamp > timestamp and after_sample is None:
                after_sample = sample
                break

        if before_sample is None:
            return self._samples[0].value

        if after_sample is None:
            return before_sample.value

        # Linear interpolation
        dt = float(after_sample.timestamp - before_sample.timestamp)
        if dt == 0:
            return before_sample.value

        t_frac = float(timestamp - before_sample.timestamp) / dt
        return before_sample.value + t_frac * (after_sample.value - before_sample.value)

    def get_history(self) -> List[SignalSample]:
        """Get complete signal history."""
        return self._samples.copy()

    def get_samples_in_range(
        self,
        start_time: Decimal,
        end_time: Decimal
    ) -> List[SignalSample]:
        """Get signal samples within specified time range."""
        return [
            sample for sample in self._samples
            if start_time <= sample.timestamp <= end_time
        ]

    def add_observer(self, observer: Callable[[Signal, float], None]) -> None:
        """Add observer for signal changes."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[Signal, float], None]) -> None:
        """Remove signal observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def clear_history(self) -> None:
        """Clear signal history."""
        self._samples.clear()

    def to_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert signal history to numpy arrays for plotting."""
        if not self._samples:
            return np.array([]), np.array([])

        times = np.array([float(sample.timestamp) for sample in self._samples])
        values = np.array([sample.value for sample in self._samples])

        return times, values

    def compute_rms(self, start_time: Decimal, end_time: Decimal) -> float:
        """Compute RMS value over specified time interval."""
        samples = self.get_samples_in_range(start_time, end_time)
        if len(samples) < 2:
            return abs(self._current_value)

        # Trapezoidal integration for RMS calculation
        total_energy = 0.0
        total_time = 0.0

        for i in range(len(samples) - 1):
            dt = float(samples[i + 1].timestamp - samples[i].timestamp)
            avg_value = (samples[i].value + samples[i + 1].value) / 2
            total_energy += avg_value * avg_value * dt
            total_time += dt

        if total_time == 0:
            return abs(self._current_value)

        return np.sqrt(total_energy / total_time)

    def __repr__(self) -> str:
        return f"Signal(name='{self.name}', type={self.signal_type}, value={self._current_value})"


class SignalBus:
    """Manages multiple related signals."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._signals: Dict[str, Signal] = {}

    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the bus."""
        self._signals[signal.name] = signal

    def get_signal(self, name: str) -> Signal:
        """Get signal by name."""
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' not found in bus '{self.name}'")
        return self._signals[name]

    def remove_signal(self, name: str) -> None:
        """Remove signal from bus."""
        if name in self._signals:
            del self._signals[name]

    def list_signals(self) -> List[str]:
        """List all signal names in the bus."""
        return list(self._signals.keys())

    def clear_all_history(self) -> None:
        """Clear history for all signals in the bus."""
        for signal in self._signals.values():
            signal.clear_history()

    def __len__(self) -> int:
        return len(self._signals)

    def __contains__(self, name: str) -> bool:
        return name in self._signals