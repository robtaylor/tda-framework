"""Timing analysis tools for hardware performance evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics
from collections import defaultdict

from ..core.circuit import Circuit
from ..core.simulation import Simulator
from ..hardware.models import HardwareModel


@dataclass
class TimingResult:
    """Results from timing analysis."""
    component_name: str
    hardware_model: str
    propagation_delay: Decimal
    power_consumption: float
    accuracy: float
    execution_count: int = 0
    total_time: Decimal = Decimal('0')


@dataclass
class CircuitTimingReport:
    """Complete timing report for a circuit."""
    circuit_name: str
    total_execution_time: Decimal
    critical_path_delay: Decimal
    total_power_consumption: float
    component_results: List[TimingResult] = field(default_factory=list)
    timing_violations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TimingAnalyzer:
    """Analyzes execution time for different hardware implementations."""

    def __init__(self) -> None:
        self._timing_data: Dict[str, List[Decimal]] = defaultdict(list)
        self._power_data: Dict[str, List[float]] = defaultdict(list)
        self._execution_counts: Dict[str, int] = defaultdict(int)

    def analyze_circuit(
        self,
        circuit: Circuit,
        simulation_time: Decimal = Decimal('1e-3')
    ) -> CircuitTimingReport:
        """Analyze timing characteristics of a circuit."""
        # Create simulator and run
        simulator = circuit.to_simulator()
        results = simulator.run_simulation(simulation_time)

        # Collect timing data for each component
        component_results = []
        total_power = 0.0
        critical_path_delay = Decimal('0')

        for comp_name, component in circuit._components.items():
            if component.hardware_model:
                prop_delay = component.hardware_model.get_propagation_delay()
                power = component.hardware_model.get_power_consumption()
                accuracy = component.hardware_model.get_accuracy()

                result = TimingResult(
                    component_name=comp_name,
                    hardware_model=component.hardware_model.name,
                    propagation_delay=prop_delay,
                    power_consumption=power,
                    accuracy=accuracy,
                    execution_count=1  # Simplified for now
                )

                component_results.append(result)
                total_power += power
                critical_path_delay = max(critical_path_delay, prop_delay)

        # Calculate performance metrics
        performance_metrics = {
            "events_per_second": float(simulator.get_statistics()["events_executed"]) / float(simulation_time),
            "average_component_delay": float(statistics.mean([float(r.propagation_delay) for r in component_results]) if component_results else 0),
            "power_efficiency": total_power / float(simulation_time) if simulation_time > 0 else 0.0,
            "accuracy_range": (
                min(r.accuracy for r in component_results),
                max(r.accuracy for r in component_results)
            ) if component_results else (0.0, 0.0)
        }

        return CircuitTimingReport(
            circuit_name=circuit.name,
            total_execution_time=results["simulation_time"],
            critical_path_delay=critical_path_delay,
            total_power_consumption=total_power,
            component_results=component_results,
            performance_metrics=performance_metrics
        )

    def compare_hardware_implementations(
        self,
        circuit: Circuit,
        hardware_alternatives: Dict[str, List[str]]
    ) -> Dict[str, CircuitTimingReport]:
        """Compare different hardware implementations for components."""
        results = {}

        for scenario_name, component_models in hardware_alternatives.items():
            # Create copy of circuit with alternative hardware
            test_circuit = self._create_circuit_variant(circuit, component_models)
            report = self.analyze_circuit(test_circuit)
            results[scenario_name] = report

        return results

    def _create_circuit_variant(
        self,
        original_circuit: Circuit,
        hardware_mapping: List[str]
    ) -> Circuit:
        """Create circuit variant with different hardware models."""
        # This is a simplified implementation
        # In practice, would need more sophisticated hardware substitution
        variant_circuit = Circuit(f"{original_circuit.name}_variant")

        # Copy components and update hardware models
        for i, (comp_name, component) in enumerate(original_circuit._components.items()):
            if i < len(hardware_mapping):
                hardware_name = hardware_mapping[i]
                try:
                    hardware_model = original_circuit._hardware_library.get_model(hardware_name)
                    variant_circuit.add_component(component, hardware_model)
                except KeyError:
                    variant_circuit.add_component(component)
            else:
                variant_circuit.add_component(component)

        return variant_circuit

    def generate_timing_constraints(
        self,
        report: CircuitTimingReport,
        margin_factor: float = 1.2
    ) -> Dict[str, Decimal]:
        """Generate timing constraints based on analysis results."""
        constraints = {}

        for result in report.component_results:
            max_delay = result.propagation_delay * Decimal(str(margin_factor))
            constraints[result.component_name] = max_delay

        constraints["circuit_total"] = report.critical_path_delay * Decimal(str(margin_factor))

        return constraints

    def identify_bottlenecks(
        self,
        report: CircuitTimingReport,
        threshold_percentile: float = 0.8
    ) -> List[TimingResult]:
        """Identify timing bottlenecks in the circuit."""
        if not report.component_results:
            return []

        delays = [float(r.propagation_delay) for r in report.component_results]
        threshold = statistics.quantiles(delays, n=10)[int(threshold_percentile * 10) - 1]

        bottlenecks = [
            result for result in report.component_results
            if float(result.propagation_delay) >= threshold
        ]

        return sorted(bottlenecks, key=lambda x: x.propagation_delay, reverse=True)


class PerformanceProfiler:
    """Profiles component-level performance during simulation."""

    def __init__(self) -> None:
        self._component_timings: Dict[str, List[Decimal]] = defaultdict(list)
        self._component_calls: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, Decimal] = {}

    def start_timing(self, component_name: str, timestamp: Decimal) -> None:
        """Start timing a component operation."""
        self._start_times[component_name] = timestamp

    def end_timing(self, component_name: str, timestamp: Decimal) -> None:
        """End timing a component operation."""
        if component_name in self._start_times:
            duration = timestamp - self._start_times[component_name]
            self._component_timings[component_name].append(duration)
            self._component_calls[component_name] += 1
            del self._start_times[component_name]

    def get_profile_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate performance profile report."""
        report = {}

        for component_name, timings in self._component_timings.items():
            if timings:
                float_timings = [float(t) for t in timings]
                report[component_name] = {
                    "total_time": sum(float_timings),
                    "average_time": statistics.mean(float_timings),
                    "min_time": min(float_timings),
                    "max_time": max(float_timings),
                    "call_count": self._component_calls[component_name],
                    "time_per_call": sum(float_timings) / self._component_calls[component_name]
                }

        return report

    def get_hotspots(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N performance hotspots by total time."""
        total_times = []

        for component_name, timings in self._component_timings.items():
            if timings:
                total_time = sum(float(t) for t in timings)
                total_times.append((component_name, total_time))

        return sorted(total_times, key=lambda x: x[1], reverse=True)[:top_n]


class AccuracyAnalyzer:
    """Analyzes precision and accuracy with hardware noise models."""

    def __init__(self) -> None:
        self._accuracy_data: Dict[str, List[float]] = defaultdict(list)

    def analyze_signal_accuracy(
        self,
        circuit: Circuit,
        reference_values: Dict[str, float],
        simulation_time: Decimal = Decimal('1e-3')
    ) -> Dict[str, Dict[str, float]]:
        """Analyze signal accuracy compared to reference values."""
        simulator = circuit.to_simulator()
        simulator.run_simulation(simulation_time)

        accuracy_report = {}

        for signal_name, reference_value in reference_values.items():
            try:
                signal = circuit.get_signal(signal_name)
                actual_value = signal.current_value

                absolute_error = abs(actual_value - reference_value)
                relative_error = absolute_error / abs(reference_value) if reference_value != 0 else float('inf')

                accuracy_report[signal_name] = {
                    "reference_value": reference_value,
                    "actual_value": actual_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "accuracy_bits": -math.log2(relative_error) if relative_error > 0 else float('inf')
                }

            except KeyError:
                accuracy_report[signal_name] = {
                    "error": f"Signal '{signal_name}' not found in circuit"
                }

        return accuracy_report

    def monte_carlo_analysis(
        self,
        circuit: Circuit,
        num_runs: int = 100,
        simulation_time: Decimal = Decimal('1e-3')
    ) -> Dict[str, Dict[str, Any]]:
        """Perform Monte Carlo analysis for statistical accuracy assessment."""
        results: Dict[str, List[float]] = defaultdict(list)

        for run in range(num_runs):
            simulator = circuit.to_simulator()
            simulator.run_simulation(simulation_time)

            # Collect signal values
            for signal_name in circuit._signals:
                signal = circuit.get_signal(signal_name)
                # Add hardware noise if models are present
                value = signal.current_value
                for component in circuit._components.values():
                    if component.hardware_model:
                        value = component.hardware_model.add_noise(value)
                        break  # Use first hardware model found

                results[signal_name].append(value)

        # Calculate statistics
        stats_report = {}
        for signal_name, values in results.items():
            stats_report[signal_name] = {
                "mean": statistics.mean(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "cv": statistics.stdev(values) / abs(statistics.mean(values)) if len(values) > 1 and statistics.mean(values) != 0 else 0.0
            }

        return stats_report