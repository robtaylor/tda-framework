"""Circuit construction and connection management."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

from .signals import Signal, SignalBus
from .simulation import Simulator
from ..components.base import Component
from ..hardware.models import HardwareModel, HardwareLibrary


@dataclass
class Connection:
    """Represents a connection between component ports."""
    from_component: str
    from_port: str
    to_component: str
    to_port: str
    signal_name: Optional[str] = None


@dataclass
class Node:
    """Represents an electrical node connecting multiple signals."""
    name: str
    signals: Set[str] = field(default_factory=set)
    voltage_level: float = 0.0


class Circuit:
    """Container for components with connection topology."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._components: Dict[str, Component] = {}
        self._signals: Dict[str, Signal] = {}
        self._connections: List[Connection] = []
        self._nodes: Dict[str, Node] = {}
        self._simulator: Optional[Simulator] = None
        self._hardware_library = HardwareLibrary()

    def add_component(
        self,
        component: Component,
        hardware_model: Optional[str | HardwareModel] = None
    ) -> Component:
        """Add a component to the circuit."""
        if component.name in self._components:
            raise ValueError(f"Component '{component.name}' already exists in circuit")

        # Set hardware model if specified
        if hardware_model is not None:
            if isinstance(hardware_model, str):
                hardware_model = self._hardware_library.get_model(hardware_model)
            component.hardware_model = hardware_model

        self._components[component.name] = component
        return component

    def get_component(self, name: str) -> Component:
        """Get component by name."""
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in circuit")
        return self._components[name]

    def create_signal(
        self,
        name: str,
        initial_value: float = 0.0,
        units: str = "V"
    ) -> Signal:
        """Create a new signal in the circuit."""
        if name in self._signals:
            raise ValueError(f"Signal '{name}' already exists in circuit")

        signal = Signal(name, initial_value=initial_value, units=units)
        self._signals[name] = signal
        return signal

    def get_signal(self, name: str) -> Signal:
        """Get signal by name."""
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' not found in circuit")
        return self._signals[name]

    def connect(
        self,
        from_component: str,
        from_port: str,
        to_component: str,
        to_port: str,
        signal_name: Optional[str] = None
    ) -> Signal:
        """Connect two component ports with a signal."""
        # Validate components exist
        from_comp = self.get_component(from_component)
        to_comp = self.get_component(to_component)

        # Validate ports exist
        from_port_obj = from_comp.get_port(from_port)
        to_port_obj = to_comp.get_port(to_port)

        # Validate connection direction
        if from_port_obj.is_input:
            raise ValueError(f"Cannot connect from input port {from_component}.{from_port}")
        if not to_port_obj.is_input:
            raise ValueError(f"Cannot connect to output port {to_component}.{to_port}")

        # Create or get signal
        if signal_name is None:
            signal_name = f"{from_component}_{from_port}_to_{to_component}_{to_port}"

        if signal_name in self._signals:
            signal = self._signals[signal_name]
        else:
            signal = self.create_signal(signal_name)

        # Connect signal to ports
        from_comp.connect_signal(from_port, signal)
        to_comp.connect_signal(to_port, signal)

        # Record connection
        connection = Connection(from_component, from_port, to_component, to_port, signal_name)
        self._connections.append(connection)

        return signal

    def connect_to_node(
        self,
        component: str,
        port: str,
        node_name: str
    ) -> Signal:
        """Connect a component port to an electrical node."""
        comp = self.get_component(component)
        port_obj = comp.get_port(port)

        # Create node if it doesn't exist
        if node_name not in self._nodes:
            self._nodes[node_name] = Node(node_name)

        # Create or get node signal
        node_signal_name = f"node_{node_name}"
        if node_signal_name not in self._signals:
            signal = self.create_signal(node_signal_name)
        else:
            signal = self._signals[node_signal_name]

        # Connect component to signal
        comp.connect_signal(port, signal)

        # Add signal to node
        self._nodes[node_name].signals.add(signal.name)

        return signal

    def set_input_value(
        self,
        component: str,
        port: str,
        value: float
    ) -> None:
        """Set the value of an input signal."""
        comp = self.get_component(component)
        port_obj = comp.get_port(port)

        if not port_obj.is_input:
            raise ValueError(f"Port {component}.{port} is not an input port")

        if port_obj.signal is None:
            # Create a signal for this input
            signal_name = f"{component}_{port}_input"
            signal = self.create_signal(signal_name, initial_value=value)
            comp.connect_signal(port, signal)
        else:
            # Update existing signal
            port_obj.signal.set_value(
                self._simulator.current_time if self._simulator else 0,
                value
            )

    def get_output_value(self, component: str, port: str) -> float:
        """Get the current value of an output signal."""
        comp = self.get_component(component)
        port_obj = comp.get_port(port)

        if port_obj.is_input:
            raise ValueError(f"Port {component}.{port} is not an output port")

        if port_obj.signal is None:
            return 0.0

        return port_obj.signal.current_value

    def validate_connections(self) -> List[str]:
        """Validate circuit connections and return list of issues."""
        issues = []

        for comp_name, component in self._components.items():
            for port_name, port in component._ports.items():
                if port.signal is None:
                    if port.is_input:
                        issues.append(f"Unconnected input port: {comp_name}.{port_name}")
                    else:
                        issues.append(f"Unconnected output port: {comp_name}.{port_name}")

        return issues

    def get_connection_graph(self) -> Dict[str, List[str]]:
        """Get circuit connectivity as adjacency list."""
        graph = {}

        for connection in self._connections:
            from_node = f"{connection.from_component}.{connection.from_port}"
            to_node = f"{connection.to_component}.{connection.to_port}"

            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append(to_node)

        return graph

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit statistics."""
        return {
            "components": len(self._components),
            "signals": len(self._signals),
            "connections": len(self._connections),
            "nodes": len(self._nodes),
            "component_types": list(set(type(comp).__name__ for comp in self._components.values())),
            "validation_issues": len(self.validate_connections())
        }

    def to_simulator(self, config: Optional[Any] = None) -> Simulator:
        """Create and configure a simulator for this circuit."""
        simulator = Simulator(config)

        # Add all components
        for component in self._components.values():
            simulator.add_component(component)

        # Add all signals
        for signal in self._signals.values():
            simulator.signal_bus.add_signal(signal)

        # Connect nodes
        for node_name, node in self._nodes.items():
            signals = [self._signals[sig_name] for sig_name in node.signals]
            simulator.connect_signals(node_name, *signals)

        self._simulator = simulator
        return simulator


class CircuitBuilder:
    """Fluent API for easy circuit construction."""

    def __init__(self, name: str) -> None:
        self._circuit = Circuit(name)
        self._current_component: Optional[str] = None

    def add_component(
        self,
        component: Component,
        hardware_model: Optional[str | HardwareModel] = None
    ) -> CircuitBuilder:
        """Add component and make it current for chaining."""
        self._circuit.add_component(component, hardware_model)
        self._current_component = component.name
        return self

    def connect_to(
        self,
        to_component: str,
        from_port: str = "output",
        to_port: str = "input",
        signal_name: Optional[str] = None
    ) -> CircuitBuilder:
        """Connect current component to another component."""
        if self._current_component is None:
            raise ValueError("No current component set")

        self._circuit.connect(
            self._current_component,
            from_port,
            to_component,
            to_port,
            signal_name
        )
        self._current_component = to_component
        return self

    def connect_node(
        self,
        node_name: str,
        port: str = "output"
    ) -> CircuitBuilder:
        """Connect current component to a node."""
        if self._current_component is None:
            raise ValueError("No current component set")

        self._circuit.connect_to_node(self._current_component, port, node_name)
        return self

    def set_input(
        self,
        port: str,
        value: float
    ) -> CircuitBuilder:
        """Set input value for current component."""
        if self._current_component is None:
            raise ValueError("No current component set")

        self._circuit.set_input_value(self._current_component, port, value)
        return self

    def select_component(self, name: str) -> CircuitBuilder:
        """Select component as current for chaining."""
        self._current_component = name
        return self

    def build(self) -> Circuit:
        """Build and return the circuit."""
        return self._circuit