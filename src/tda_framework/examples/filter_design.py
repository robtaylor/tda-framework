"""Example: Active filter design using analog computation."""

from decimal import Decimal
from typing import List, Tuple
import math
from ..core.circuit import Circuit, CircuitBuilder
from ..components.analog import Integrator, SummingAmplifier, FunctionGenerator
from ..analysis.visualization import SignalTracer


def create_lowpass_filter(
    cutoff_frequency: float = 1000.0,
    order: int = 2,
    hardware_model: str = "LF356"
) -> Circuit:
    """Create an active low-pass filter using analog computation."""

    if order < 1 or order > 4:
        raise ValueError("Filter order must be between 1 and 4")

    circuit_name = f"Lowpass_Filter_Order_{order}"
    builder = CircuitBuilder(circuit_name)

    # Input stage
    input_buffer = SummingAmplifier("input_buffer", gains=[1.0], num_inputs=1)
    builder.add_component(input_buffer, hardware_model)

    # Create integrator stages for the filter
    prev_component = "input_buffer"
    for i in range(order):
        # Calculate time constant for Butterworth response
        # This is simplified - real filter design would use proper pole placement
        time_constant = 1.0 / (2 * math.pi * cutoff_frequency)
        gain = 1.0 / time_constant

        integrator_name = f"integrator_{i+1}"
        integrator = Integrator(integrator_name, gain=gain)

        builder.add_component(integrator, hardware_model)
        builder.select_component(prev_component)
        builder.connect_to(integrator_name, "output", "input")

        prev_component = integrator_name

    # Output buffer
    output_buffer = SummingAmplifier("output_buffer", gains=[1.0], num_inputs=1)
    builder.add_component(output_buffer, hardware_model)
    builder.select_component(prev_component)
    builder.connect_to("output_buffer", "output", "input_0")

    return builder.build()


def create_bandpass_filter(
    center_frequency: float = 1000.0,
    q_factor: float = 10.0,
    hardware_model: str = "LF356"
) -> Circuit:
    """Create a bandpass filter using analog computation."""

    # Calculate component values
    omega = 2 * math.pi * center_frequency
    bandwidth = center_frequency / q_factor

    # Use state-variable filter topology
    builder = CircuitBuilder("Bandpass_Filter")

    # Input summer
    input_summer = SummingAmplifier("input_summer", gains=[1.0, -q_factor], num_inputs=2)
    builder.add_component(input_summer, hardware_model)

    # First integrator (gives -90° phase shift)
    integrator1 = Integrator("integrator1", gain=omega)
    builder.add_component(integrator1, hardware_model)
    builder.select_component("input_summer")
    builder.connect_to("integrator1", "output", "input")

    # Second integrator (gives another -90° phase shift)
    integrator2 = Integrator("integrator2", gain=omega)
    builder.add_component(integrator2, hardware_model)
    builder.select_component("integrator1")
    builder.connect_to("integrator2", "output", "input")

    # Feedback from integrator2 to input_summer
    builder.select_component("integrator2")
    builder.connect_to("input_summer", "output", "input_1")

    # Output buffer for bandpass response (taken from integrator1)
    output_buffer = SummingAmplifier("output_buffer", gains=[1.0], num_inputs=1)
    builder.add_component(output_buffer, hardware_model)
    builder.select_component("integrator1")
    builder.connect_to("output_buffer", "output", "input_0")

    return builder.build()


def test_filter_response(
    circuit: Circuit,
    test_frequencies: List[float],
    simulation_time: Decimal = Decimal('0.01')
) -> dict:
    """Test filter frequency response."""

    results = {}
    tracer = SignalTracer()

    for freq in test_frequencies:
        print(f"Testing frequency: {freq} Hz")

        # Create test signal generator
        signal_gen = FunctionGenerator(
            "test_signal",
            function_type="sine",
            amplitude=1.0,
            frequency=freq,
            hardware_model="LM741"
        )

        # Add signal generator to circuit temporarily
        test_circuit = Circuit(f"{circuit.name}_test_{freq}Hz")

        # Copy components from original circuit
        for comp_name, component in circuit._components.items():
            test_circuit.add_component(component)

        # Add signal generator
        test_circuit.add_component(signal_gen)

        # Connect signal generator to filter input
        # This is simplified - would need proper port mapping in real implementation
        test_circuit.connect("test_signal", "output", list(circuit._components.keys())[0], "input_0")

        # Run simulation
        simulator = test_circuit.to_simulator()
        sim_results = simulator.run_simulation(simulation_time)

        # Record results
        results[freq] = {
            "simulation_time": float(sim_results["simulation_time"]),
            "final_signals": sim_results["signals"]
        }

    return {"frequency_responses": results, "tracer": tracer}


def design_filter_comparison() -> dict:
    """Compare different filter implementations."""

    filters = {
        "1st_Order_LP": create_lowpass_filter(cutoff_frequency=1000, order=1),
        "2nd_Order_LP": create_lowpass_filter(cutoff_frequency=1000, order=2),
        "3rd_Order_LP": create_lowpass_filter(cutoff_frequency=1000, order=3),
        "Bandpass_Q10": create_bandpass_filter(center_frequency=1000, q_factor=10),
        "Bandpass_Q2": create_bandpass_filter(center_frequency=1000, q_factor=2)
    }

    # Test frequencies from 10 Hz to 100 kHz
    test_frequencies = [10.0 * (10 ** (i/10)) for i in range(41)]  # Logarithmic spacing

    comparison_results = {}

    for filter_name, filter_circuit in filters.items():
        print(f"\nTesting {filter_name}...")

        # Test with a subset of frequencies for demo
        test_freqs = [100, 1000, 10000]  # Hz
        response = test_filter_response(filter_circuit, test_freqs)

        comparison_results[filter_name] = {
            "circuit": filter_circuit,
            "frequency_response": response["frequency_responses"],
            "component_count": len(filter_circuit._components),
            "connection_count": len(filter_circuit._connections)
        }

    return comparison_results


if __name__ == "__main__":
    print("Creating filter designs...")

    # Create example filters
    lp_filter = create_lowpass_filter(cutoff_frequency=1000, order=2)
    bp_filter = create_bandpass_filter(center_frequency=1000, q_factor=5)

    print(f"Low-pass filter: {len(lp_filter._components)} components")
    print(f"Band-pass filter: {len(bp_filter._components)} components")

    # Test frequency response
    print("\nTesting frequency responses...")
    test_frequencies = [100, 500, 1000, 2000, 5000]  # Hz

    lp_response = test_filter_response(lp_filter, test_frequencies)
    print(f"Low-pass filter tested at {len(test_frequencies)} frequencies")

    # Full comparison
    print("\nRunning filter comparison...")
    comparison = design_filter_comparison()

    for filter_name, results in comparison.items():
        print(f"\n{filter_name}:")
        print(f"  Components: {results['component_count']}")
        print(f"  Connections: {results['connection_count']}")
        print(f"  Test frequencies: {len(results['frequency_response'])}")