"""Simple demonstration of the TDA framework capabilities."""

from decimal import Decimal
from ..core.circuit import CircuitBuilder
from ..components.analog import Integrator, SummingAmplifier, CoefficientMultiplier, FunctionGenerator
from ..analysis.timing import TimingAnalyzer
from ..analysis.visualization import SignalTracer


def demo_basic_integrator():
    """Demonstrate basic integrator functionality."""
    print("=== Basic Integrator Demo ===")

    # Create an integrator with unity gain
    integrator = Integrator("demo_integrator", gain=1.0, initial_condition=0.0)

    # Build simple circuit
    circuit = (CircuitBuilder("Basic_Integrator_Demo")
              .add_component(integrator, "LM741")
              .build())

    # Create necessary signals manually
    input_signal = circuit.create_signal("integrator_input", initial_value=1.0)
    output_signal = circuit.create_signal("integrator_output")
    reset_signal = circuit.create_signal("integrator_reset", initial_value=0.0)

    # Connect signals to integrator
    integrator.connect_signal("input", input_signal)
    integrator.connect_signal("output", output_signal)
    integrator.connect_signal("reset", reset_signal)

    # Set up simulation
    simulator = circuit.to_simulator()

    # Run simulation
    results = simulator.run_simulation(Decimal('0.001'))  # 1ms

    final_output = output_signal.current_value

    print(f"  Input: 1.0V step")
    print(f"  Simulation time: {results['simulation_time']:.2e}s")
    print(f"  Final output: {final_output:.6f}V")
    print(f"  Expected (∫1dt): {float(results['simulation_time']):.6f}V")
    print(f"  Events executed: {results['events_executed']}")
    print()


def demo_pid_controller():
    """Demonstrate PID controller construction and analysis."""
    print("=== PID Controller Demo ===")

    # PID gains
    kp, ki, kd = 2.0, 0.5, 0.1

    # Error summer (setpoint - feedback)
    error_summer = SummingAmplifier("error_summer", gains=[1.0, -1.0])

    # Proportional path
    p_gain = CoefficientMultiplier("p_gain", coefficient=kp)

    # Integral path
    integrator = Integrator("integrator", gain=ki)

    # Derivative path (simplified)
    d_gain = CoefficientMultiplier("d_gain", coefficient=kd)

    # Output summer
    output_summer = SummingAmplifier("output_summer", gains=[1.0, 1.0, 1.0], num_inputs=3)

    # Build circuit
    circuit = (CircuitBuilder("PID_Demo")
              .add_component(error_summer, "LF356")
              .connect_to("p_gain", "output", "input")
              .add_component(p_gain, "LF356")
              .connect_to("output_summer", "output", "input_0")
              .add_component(output_summer, "LF356")
              .select_component("error_summer")
              .connect_to("integrator", "output", "input")
              .add_component(integrator, "LF356")
              .connect_to("output_summer", "output", "input_1")
              .select_component("error_summer")
              .connect_to("d_gain", "output", "input")
              .add_component(d_gain, "LF356")
              .connect_to("output_summer", "output", "input_2")
              .build())

    print(f"  PID gains: Kp={kp}, Ki={ki}, Kd={kd}")
    print(f"  Components: {len(circuit._components)}")
    print(f"  Connections: {len(circuit._connections)}")

    # Analyze timing with different hardware
    analyzer = TimingAnalyzer()

    hardware_options = {
        "Standard OpAmps": ["LM741"] * 5,
        "High-Speed OpAmps": ["LF356"] * 5,
        "Ultra-Fast OpAmps": ["OPA847"] * 5
    }

    print("\n  Hardware Implementation Comparison:")
    for impl_name, hw_list in hardware_options.items():
        # Create variant circuit (simplified approach)
        test_circuit = (CircuitBuilder(f"PID_{impl_name}")
                       .add_component(error_summer, hw_list[0])
                       .add_component(p_gain, hw_list[1])
                       .add_component(integrator, hw_list[2])
                       .add_component(d_gain, hw_list[3])
                       .add_component(output_summer, hw_list[4])
                       .build())

        report = analyzer.analyze_circuit(test_circuit)
        print(f"    {impl_name}:")
        print(f"      Critical path delay: {report.critical_path_delay:.2e}s")
        print(f"      Power consumption: {report.total_power_consumption:.4f}W")

    print()


def demo_signal_generation():
    """Demonstrate signal generation and analysis."""
    print("=== Signal Generation Demo ===")

    # Create function generator
    sine_gen = FunctionGenerator(
        "sine_generator",
        function_type="sine",
        amplitude=1.0,
        frequency=1000.0  # 1kHz
    )

    square_gen = FunctionGenerator(
        "square_generator",
        function_type="square",
        amplitude=1.0,
        frequency=1000.0
    )

    # Build circuit
    circuit = (CircuitBuilder("Signal_Generator_Demo")
              .add_component(sine_gen, "LM741")
              .add_component(square_gen, "LM741")
              .build())

    # Set up simulation
    simulator = circuit.to_simulator()
    tracer = SignalTracer()

    # Run simulation for 2 periods
    simulation_time = Decimal('0.002')  # 2ms = 2 periods at 1kHz
    results = simulator.run_simulation(simulation_time)

    print(f"  Signal frequency: 1000 Hz")
    print(f"  Simulation time: {float(simulation_time):.3f}s ({float(simulation_time) * 1000:.1f} periods)")
    print(f"  Events executed: {results['events_executed']}")

    # Get final values
    for signal_name in circuit._signals:
        signal = circuit.get_signal(signal_name)
        if signal.name.endswith("_output"):
            print(f"  Final {signal.name}: {signal.current_value:.3f}V")

    print()


def demo_hardware_comparison():
    """Demonstrate comprehensive hardware comparison."""
    print("=== Hardware Comparison Demo ===")

    # Create a simple amplifier chain
    amp1 = CoefficientMultiplier("amp1", coefficient=2.0)
    amp2 = CoefficientMultiplier("amp2", coefficient=3.0)
    amp3 = CoefficientMultiplier("amp3", coefficient=0.5)

    base_circuit = (CircuitBuilder("Amplifier_Chain")
                   .add_component(amp1)
                   .connect_to("amp2", "output", "input")
                   .add_component(amp2)
                   .connect_to("amp3", "output", "input")
                   .add_component(amp3)
                   .build())

    # Hardware alternatives
    hardware_scenarios = {
        "Precision OpAmps": ["OPA847", "OPA847", "OPA847"],
        "General Purpose": ["LF356", "LF356", "LF356"],
        "Low Cost": ["LM741", "LM741", "LM741"],
        "FPGA Implementation": ["Virtex-7", "Virtex-7", "Virtex-7"],
        "Microcontroller": ["STM32F4", "STM32F4", "STM32F4"]
    }

    analyzer = TimingAnalyzer()

    print("  Analyzing different hardware implementations...")
    print(f"  Circuit: {len(base_circuit._components)} amplifier stages")
    print(f"  Overall gain: {2.0 * 3.0 * 0.5} = 3.0x")

    comparison_results = {}

    # Simplified comparison (in real implementation would use proper hardware substitution)
    for scenario_name, hw_models in hardware_scenarios.items():
        # Create test circuit with hardware models
        test_circuit = (CircuitBuilder(f"Test_{scenario_name.replace(' ', '_')}")
                       .add_component(CoefficientMultiplier("amp1", 2.0), hw_models[0])
                       .connect_to("amp2", "output", "input")
                       .add_component(CoefficientMultiplier("amp2", 3.0), hw_models[1])
                       .connect_to("amp3", "output", "input")
                       .add_component(CoefficientMultiplier("amp3", 0.5), hw_models[2])
                       .build())

        report = analyzer.analyze_circuit(test_circuit)
        comparison_results[scenario_name] = report

        print(f"\n  {scenario_name}:")
        print(f"    Execution time: {report.total_execution_time:.2e}s")
        print(f"    Critical path delay: {report.critical_path_delay:.2e}s")
        print(f"    Power consumption: {report.total_power_consumption:.4f}W")
        print(f"    Components analyzed: {len(report.component_results)}")

    # Find best and worst performers
    best_speed = min(comparison_results.values(), key=lambda r: r.critical_path_delay)
    best_power = min(comparison_results.values(), key=lambda r: r.total_power_consumption)

    print(f"\n  Performance Summary:")
    print(f"    Fastest: {[k for k, v in comparison_results.items() if v == best_speed][0]}")
    print(f"    Lowest Power: {[k for k, v in comparison_results.items() if v == best_power][0]}")

    print()


if __name__ == "__main__":
    print("TDA Framework Demonstration")
    print("=" * 50)
    print()

    # Run all demonstrations
    demo_basic_integrator()
    demo_pid_controller()
    demo_signal_generation()
    demo_hardware_comparison()

    print("Demo complete! Try running individual examples:")
    print("  - tda_framework.examples.pid_controller")
    print("  - tda_framework.examples.filter_design")