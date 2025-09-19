"""Standard analog computation components."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import List, Optional, TYPE_CHECKING

from .base import Component, IntegratingComponent, NonlinearComponent

if TYPE_CHECKING:
    from ..hardware.models import HardwareModel


class Integrator(IntegratingComponent):
    """Analog integrator component implementing ∫x(t)dt."""

    def __init__(
        self,
        name: str,
        gain: float = 1.0,
        initial_condition: float = 0.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, initial_condition, hardware_model)
        self.gain = gain

        # Ports: input, output, reset
        self.add_port("input", is_input=True)
        self.add_port("output", is_input=False)
        self.add_port("reset", is_input=True)

    def update(self) -> None:
        """Update integrator output."""
        if not self._simulator:
            return

        # Check for reset signal
        reset_value = self.get_input_value("reset")
        if reset_value > 0.5:  # Digital high threshold
            self.reset_integrator()
            self.set_output_value("output", self._integrator_state)
            return

        # Perform integration
        input_value = self.get_input_value("input")
        current_time = self._simulator.current_time

        integrated_value = self._integrate(input_value * self.gain, current_time)
        self.set_output_value("output", integrated_value)


class SummingAmplifier(Component):
    """Multi-input summing amplifier with programmable gains."""

    def __init__(
        self,
        name: str,
        gains: Optional[List[float]] = None,
        num_inputs: int = 2,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.num_inputs = num_inputs
        self.gains = gains or [1.0] * num_inputs

        if len(self.gains) != num_inputs:
            raise ValueError(f"Number of gains ({len(self.gains)}) must match num_inputs ({num_inputs})")

        # Create input ports
        for i in range(num_inputs):
            self.add_port(f"input_{i}", is_input=True)

        self.add_port("output", is_input=False)

    def set_gain(self, input_index: int, gain: float) -> None:
        """Set gain for specific input."""
        if 0 <= input_index < self.num_inputs:
            self.gains[input_index] = gain

    def update(self) -> None:
        """Update summing amplifier output."""
        total = 0.0
        for i in range(self.num_inputs):
            input_value = self.get_input_value(f"input_{i}")
            total += input_value * self.gains[i]

        self.set_output_value("output", total)


class CoefficientMultiplier(Component):
    """Multiplies input signal by a constant coefficient."""

    def __init__(
        self,
        name: str,
        coefficient: float = 1.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.coefficient = coefficient

        self.add_port("input", is_input=True)
        self.add_port("output", is_input=False)

    def set_coefficient(self, coefficient: float) -> None:
        """Update the multiplication coefficient."""
        self.coefficient = coefficient

    def update(self) -> None:
        """Update multiplier output."""
        input_value = self.get_input_value("input")
        output_value = input_value * self.coefficient
        self.set_output_value("output", output_value)


class FunctionGenerator(NonlinearComponent):
    """Generates nonlinear functions like sine, cosine, exponential."""

    def __init__(
        self,
        name: str,
        function_type: str = "sine",
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, None, hardware_model)
        self.function_type = function_type.lower()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

        self.add_port("input", is_input=True)  # Optional time input
        self.add_port("output", is_input=False)

        # Validate function type
        valid_functions = {"sine", "cosine", "exponential", "logarithm", "square", "triangle", "sawtooth"}
        if self.function_type not in valid_functions:
            raise ValueError(f"Invalid function type: {self.function_type}. Must be one of {valid_functions}")

    def update(self) -> None:
        """Update function generator output."""
        if not self._simulator:
            return

        # Use simulation time for function generation
        time_value = float(self._simulator.current_time)

        # Calculate function value
        output_value = self._calculate_function(time_value)
        self.set_output_value("output", output_value)

    def _calculate_function(self, t: float) -> float:
        """Calculate function value for given time."""
        arg = 2 * math.pi * self.frequency * t + self.phase

        if self.function_type == "sine":
            return self.amplitude * math.sin(arg) + self.offset
        elif self.function_type == "cosine":
            return self.amplitude * math.cos(arg) + self.offset
        elif self.function_type == "exponential":
            return self.amplitude * math.exp(self.frequency * t) + self.offset
        elif self.function_type == "logarithm":
            if t <= 0:
                return self.offset
            return self.amplitude * math.log(t * self.frequency + 1) + self.offset
        elif self.function_type == "square":
            return self.amplitude * (1 if math.sin(arg) >= 0 else -1) + self.offset
        elif self.function_type == "triangle":
            phase_norm = (arg / (2 * math.pi)) % 1.0
            if phase_norm < 0.5:
                return self.amplitude * (4 * phase_norm - 1) + self.offset
            else:
                return self.amplitude * (3 - 4 * phase_norm) + self.offset
        elif self.function_type == "sawtooth":
            phase_norm = (arg / (2 * math.pi)) % 1.0
            return self.amplitude * (2 * phase_norm - 1) + self.offset
        else:
            return self.offset


class Comparator(Component):
    """Voltage comparator with optional hysteresis."""

    def __init__(
        self,
        name: str,
        threshold: float = 0.0,
        hysteresis: float = 0.0,
        output_high: float = 1.0,
        output_low: float = 0.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.output_high = output_high
        self.output_low = output_low

        self._state["output_state"] = False  # Track current output state

        self.add_port("input_positive", is_input=True)
        self.add_port("input_negative", is_input=True)
        self.add_port("output", is_input=False)

    def update(self) -> None:
        """Update comparator output."""
        pos_input = self.get_input_value("input_positive")
        neg_input = self.get_input_value("input_negative")

        diff = pos_input - neg_input
        current_state = self._state.get("output_state", False)

        # Apply hysteresis
        if not current_state:  # Currently low
            if diff > (self.threshold + self.hysteresis / 2):
                new_state = True
            else:
                new_state = False
        else:  # Currently high
            if diff < (self.threshold - self.hysteresis / 2):
                new_state = False
            else:
                new_state = True

        self._state["output_state"] = new_state
        output_value = self.output_high if new_state else self.output_low
        self.set_output_value("output", output_value)


class SampleHold(Component):
    """Sample and hold circuit."""

    def __init__(
        self,
        name: str,
        hold_time: float = 1e-6,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.hold_time = hold_time

        self._state["held_value"] = 0.0
        self._state["sample_time"] = None

        self.add_port("input", is_input=True)
        self.add_port("sample_trigger", is_input=True)
        self.add_port("output", is_input=False)

    def update(self) -> None:
        """Update sample and hold output."""
        if not self._simulator:
            return

        trigger_value = self.get_input_value("sample_trigger")
        current_time = self._simulator.current_time

        # Trigger on rising edge
        if trigger_value > 0.5:  # Digital high threshold
            if self._state["sample_time"] is None:
                # Sample the input
                input_value = self.get_input_value("input")
                self._state["held_value"] = input_value
                self._state["sample_time"] = current_time

                # Schedule hold period end
                self._simulator.schedule_event(
                    Decimal(str(self.hold_time)),
                    self._end_hold_period
                )

        self.set_output_value("output", self._state["held_value"])

    def _end_hold_period(self) -> None:
        """End the hold period."""
        self._state["sample_time"] = None


class TimeScaler(Component):
    """Scales simulation time by a constant factor."""

    def __init__(
        self,
        name: str,
        scale_factor: float = 1.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.scale_factor = scale_factor

        self.add_port("input", is_input=True)
        self.add_port("output", is_input=False)

    def update(self) -> None:
        """Update time scaler output."""
        if not self._simulator:
            return

        input_value = self.get_input_value("input")
        scaled_time = float(self._simulator.current_time) * self.scale_factor

        # Output is typically the input delayed by the scaled time
        self.set_output_value("output", input_value)