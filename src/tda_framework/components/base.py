"""Base classes for analog computation components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..core.simulation import Simulator
    from ..core.signals import Signal
    from ..hardware.models import HardwareModel


@dataclass
class Port:
    """Represents an input/output port of a component."""
    name: str
    signal: Optional[Signal] = None
    is_input: bool = True


class Component(ABC):
    """Abstract base class for all analog computation components."""

    def __init__(
        self,
        name: str,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        """Initialize component with name and optional hardware model."""
        self.name = name
        self.hardware_model = hardware_model
        self._simulator: Optional[Simulator] = None
        self._ports: Dict[str, Port] = {}
        self._state: Dict[str, Any] = {}
        self._initialized = False

    def set_simulator(self, simulator: Simulator) -> None:
        """Set the simulator reference."""
        self._simulator = simulator

    def add_port(self, name: str, is_input: bool = True) -> Port:
        """Add an input/output port to the component."""
        port = Port(name=name, is_input=is_input)
        self._ports[name] = port
        return port

    def connect_signal(self, port_name: str, signal: Signal) -> None:
        """Connect a signal to a specific port."""
        if port_name not in self._ports:
            raise ValueError(f"Port '{port_name}' not found in component '{self.name}'")

        self._ports[port_name].signal = signal

        # Add signal observer for input ports
        if self._ports[port_name].is_input:
            signal.add_observer(self._on_input_change)

    def get_port(self, name: str) -> Port:
        """Get port by name."""
        if name not in self._ports:
            raise ValueError(f"Port '{name}' not found in component '{self.name}'")
        return self._ports[name]

    def get_input_value(self, port_name: str) -> float:
        """Get current value of input port."""
        port = self.get_port(port_name)
        if not port.is_input:
            raise ValueError(f"Port '{port_name}' is not an input port")
        if port.signal is None:
            return 0.0
        return port.signal.current_value

    def set_output_value(
        self,
        port_name: str,
        value: float,
        timestamp: Optional[Decimal] = None
    ) -> None:
        """Set value of output port."""
        port = self.get_port(port_name)
        if port.is_input:
            raise ValueError(f"Port '{port_name}' is not an output port")
        if port.signal is None:
            raise ValueError(f"No signal connected to output port '{port_name}'")

        if timestamp is None:
            timestamp = self._simulator.current_time if self._simulator else Decimal('0')

        port.signal.set_value(timestamp, value)

    def _on_input_change(self, signal: Signal, new_value: float) -> None:
        """Called when an input signal changes."""
        if self._initialized:
            self._schedule_update()

    def _schedule_update(self) -> None:
        """Schedule component update based on hardware timing."""
        if not self._simulator:
            return

        # Use hardware model delay if available, otherwise immediate
        delay = Decimal('0')
        if self.hardware_model:
            delay = self.hardware_model.get_propagation_delay()

        self._simulator.schedule_event(
            delay,
            self._perform_update,
            priority=1
        )

    def _perform_update(self) -> None:
        """Perform the component update calculation."""
        try:
            self.update()
        except Exception as e:
            raise RuntimeError(f"Error updating component '{self.name}': {e}") from e

    @abstractmethod
    def update(self) -> None:
        """Update component outputs based on current inputs."""
        pass

    def initialize(self) -> None:
        """Initialize component for simulation."""
        self._initialized = True
        # Schedule immediate update instead of direct call
        if self._simulator:
            self._simulator.schedule_event(Decimal('0'), self._perform_update, priority=0)

    def finalize(self) -> None:
        """Finalize component after simulation."""
        pass

    def reset(self) -> None:
        """Reset component to initial state."""
        self._state.clear()
        self._initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get component state for debugging."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "ports": {name: {"is_input": port.is_input, "connected": port.signal is not None}
                     for name, port in self._ports.items()},
            "state": self._state.copy()
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class IntegratingComponent(Component):
    """Base class for components that perform integration."""

    def __init__(
        self,
        name: str,
        initial_condition: float = 0.0,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.initial_condition = initial_condition
        self._integrator_state = initial_condition
        self._last_update_time: Optional[Decimal] = None

    def reset_integrator(self, value: Optional[float] = None) -> None:
        """Reset integrator to initial condition or specified value."""
        reset_value = value if value is not None else self.initial_condition
        self._integrator_state = reset_value
        self._last_update_time = None

    def _integrate(self, input_value: float, current_time: Decimal) -> float:
        """Perform numerical integration using trapezoidal rule."""
        if self._last_update_time is None:
            self._last_update_time = current_time
            return self._integrator_state

        dt = float(current_time - self._last_update_time)
        if dt <= 0:
            return self._integrator_state

        # Get previous input value for trapezoidal integration
        # For now, assume input was constant (rectangular integration)
        self._integrator_state += input_value * dt
        self._last_update_time = current_time

        return self._integrator_state

    def get_integrator_state(self) -> float:
        """Get current integrator state."""
        return self._integrator_state

    def reset(self) -> None:
        """Reset component and integrator state."""
        super().reset()
        self.reset_integrator()


class NonlinearComponent(Component):
    """Base class for components with nonlinear transfer functions."""

    def __init__(
        self,
        name: str,
        function_table: Optional[List[tuple[float, float]]] = None,
        hardware_model: Optional[HardwareModel] = None
    ) -> None:
        super().__init__(name, hardware_model)
        self.function_table = function_table or []

    def _lookup_function(self, input_value: float) -> float:
        """Perform lookup table interpolation for nonlinear function."""
        if not self.function_table:
            return input_value  # Linear passthrough

        # Find bounding points
        if input_value <= self.function_table[0][0]:
            return self.function_table[0][1]

        if input_value >= self.function_table[-1][0]:
            return self.function_table[-1][1]

        # Linear interpolation between points
        for i in range(len(self.function_table) - 1):
            x1, y1 = self.function_table[i]
            x2, y2 = self.function_table[i + 1]

            if x1 <= input_value <= x2:
                if x2 == x1:
                    return y1
                t = (input_value - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

        return input_value  # Fallback