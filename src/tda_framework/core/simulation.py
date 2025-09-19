"""Main simulation engine coordinating events, signals, and components."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from .events import EventScheduler
from .signals import Signal, SignalBus


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation."""
    time_resolution: Decimal = Decimal('1e-12')  # 1 picosecond
    max_simulation_time: Decimal = Decimal('1e-3')  # 1 millisecond
    max_events: int = 1_000_000
    convergence_tolerance: float = 1e-9


class Simulator:
    """Main simulation engine for time domain analog computation."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        """Initialize simulator with configuration."""
        self.config = config or SimulationConfig()
        self.scheduler = EventScheduler(self.config.time_resolution)
        self.signal_bus = SignalBus("main")

        self._components: List[Any] = []  # Will be Component instances
        self._nodes: Dict[str, Set[Signal]] = {}
        self._simulation_state = "idle"
        self._event_count = 0

    @property
    def current_time(self) -> Decimal:
        """Get current simulation time."""
        return self.scheduler.current_time

    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self._simulation_state == "running"

    def add_component(self, component: Any) -> None:
        """Add a component to the simulation."""
        self._components.append(component)
        component.set_simulator(self)

    def create_signal(
        self,
        name: str,
        initial_value: float = 0.0,
        units: str = "V"
    ) -> Signal:
        """Create and register a new signal."""
        signal = Signal(name, initial_value=initial_value, units=units)
        self.signal_bus.add_signal(signal)
        return signal

    def connect_signals(self, node_name: str, *signals: Signal) -> None:
        """Connect multiple signals to the same electrical node."""
        if node_name not in self._nodes:
            self._nodes[node_name] = set()

        for signal in signals:
            self._nodes[node_name].add(signal)

    def schedule_event(
        self,
        delay: Decimal | float,
        callback: callable,
        priority: int = 0
    ) -> None:
        """Schedule a simulation event."""
        if self._event_count >= self.config.max_events:
            raise RuntimeError(f"Maximum event count ({self.config.max_events}) exceeded")

        self.scheduler.schedule_event(delay, callback, priority)
        self._event_count += 1

    def run_simulation(
        self,
        duration: Optional[Decimal | float] = None
    ) -> Dict[str, Any]:
        """Run the simulation for specified duration."""
        if self._simulation_state == "error":
            raise RuntimeError("Simulation is in error state, reset required")

        # Allow running again from completed state
        if self._simulation_state == "completed":
            self._simulation_state = "idle"

        self._simulation_state = "running"

        if duration is None:
            duration = self.config.max_simulation_time

        if isinstance(duration, float):
            duration = Decimal(str(duration))

        try:
            # Initialize all components
            for component in self._components:
                component.initialize()

            # Run main simulation loop
            end_time = self.scheduler.current_time + duration

            while (self.scheduler.has_events() and
                   self.scheduler.peek_next_time() <= end_time and
                   self._event_count < self.config.max_events):

                self.scheduler.execute_next()
                self._event_count += 1

            # Advance time to end if no more events
            if self.scheduler.current_time < end_time:
                self.scheduler._current_time = end_time

            # Finalize components
            for component in self._components:
                component.finalize()

            self._simulation_state = "completed"

            return self._generate_results()

        except Exception as e:
            self._simulation_state = "error"
            raise e

    def reset_simulation(self) -> None:
        """Reset simulation to initial state."""
        self.scheduler.clear()
        self.signal_bus.clear_all_history()
        self._event_count = 0
        self._simulation_state = "idle"

        # Reset all components
        for component in self._components:
            component.reset()

    def pause_simulation(self) -> None:
        """Pause the simulation."""
        if self._simulation_state == "running":
            self._simulation_state = "paused"

    def resume_simulation(self) -> None:
        """Resume paused simulation."""
        if self._simulation_state == "paused":
            self._simulation_state = "running"

    def step_simulation(self) -> bool:
        """Execute one simulation step. Returns True if more events exist."""
        if not self.scheduler.has_events():
            return False

        self.scheduler.execute_next()
        self._event_count += 1

        return self.scheduler.has_events()

    def get_signal_history(self, signal_name: str) -> List[tuple]:
        """Get complete history of a signal as (time, value) tuples."""
        signal = self.signal_bus.get_signal(signal_name)
        history = signal.get_history()
        return [(float(sample.timestamp), sample.value) for sample in history]

    def _generate_results(self) -> Dict[str, Any]:
        """Generate simulation results summary."""
        results = {
            "simulation_time": float(self.scheduler.current_time),
            "events_executed": self._event_count,
            "final_state": self._simulation_state,
            "signals": {},
            "components": len(self._components),
            "nodes": len(self._nodes)
        }

        # Collect final signal values
        for signal_name in self.signal_bus.list_signals():
            signal = self.signal_bus.get_signal(signal_name)
            results["signals"][signal_name] = {
                "final_value": signal.current_value,
                "samples": len(signal.get_history()),
                "units": signal.units
            }

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            "current_time": float(self.scheduler.current_time),
            "events_scheduled": self.scheduler.event_count(),
            "events_executed": self._event_count,
            "pending_events": len(self.scheduler._event_queue),
            "simulation_state": self._simulation_state,
            "components": len(self._components),
            "signals": len(self.signal_bus),
            "nodes": len(self._nodes)
        }