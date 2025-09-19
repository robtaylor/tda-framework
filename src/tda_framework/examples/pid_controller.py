"""Example: PID Controller implementation using analog computation."""

from decimal import Decimal
from ..core.circuit import Circuit, CircuitBuilder
from ..components.analog import Integrator, SummingAmplifier, CoefficientMultiplier
from ..analysis.timing import TimingAnalyzer
from ..analysis.visualization import SignalTracer


def create_pid_controller(
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.01,
    hardware_model: str = "LM741"
) -> Circuit:
    """Create a PID controller using analog computation components."""

    # Create components
    error_input = SummingAmplifier("error_summer", gains=[1.0, -1.0])  # setpoint - feedback

    # Proportional path
    p_gain = CoefficientMultiplier("proportional_gain", coefficient=kp)

    # Integral path
    integrator = Integrator("integrator", gain=ki)

    # Derivative path (simplified using differentiator approximation)
    d_gain = CoefficientMultiplier("derivative_gain", coefficient=kd)

    # Output summer
    output_summer = SummingAmplifier("output_summer", gains=[1.0, 1.0, 1.0], num_inputs=3)

    # Build circuit using CircuitBuilder
    circuit = (CircuitBuilder("PID_Controller")
              .add_component(error_input, hardware_model)
              .connect_to("proportional_gain", "output", "input")
              .add_component(p_gain, hardware_model)
              .connect_to("output_summer", "output", "input_0")
              .add_component(output_summer, hardware_model)
              .select_component("error_summer")
              .connect_to("integrator", "output", "input")
              .add_component(integrator, hardware_model)
              .connect_to("output_summer", "output", "input_1")
              .select_component("error_summer")
              .connect_to("derivative_gain", "output", "input")
              .add_component(d_gain, hardware_model)
              .connect_to("output_summer", "output", "input_2")
              .build())

    return circuit


def simulate_pid_response(
    circuit: Circuit,
    setpoint: float = 1.0,
    simulation_time: Decimal = Decimal('0.01')
) -> dict:
    """Simulate PID controller response to step input."""

    # Set up simulation
    simulator = circuit.to_simulator()
    tracer = SignalTracer()

    # Set initial conditions
    circuit.set_input_value("error_summer", "input_0", setpoint)  # setpoint
    circuit.set_input_value("error_summer", "input_1", 0.0)      # initial feedback

    # Run simulation
    results = simulator.run_simulation(simulation_time)

    # Trace key signals
    for signal_name in ["error_summer_output_to_proportional_gain_input",
                       "integrator_output_to_output_summer_input_1",
                       "output_summer_output"]:
        try:
            signal = circuit.get_signal(signal_name)
            tracer.trace_signal(signal)
        except KeyError:
            pass  # Signal might not exist with exact name

    return {
        "simulation_results": results,
        "signal_tracer": tracer
    }


def analyze_pid_implementations() -> dict:
    """Analyze different hardware implementations of PID controller."""

    analyzer = TimingAnalyzer()

    # Test different hardware implementations
    hardware_alternatives = {
        "High-Speed OpAmps": ["OPA847", "LF356", "OPA847", "LF356", "OPA847"],
        "Standard OpAmps": ["LM741", "LM741", "LM741", "LM741", "LM741"],
        "FPGA Implementation": ["Virtex-7", "Virtex-7", "Virtex-7", "Virtex-7", "Virtex-7"],
        "Microcontroller": ["STM32F4", "STM32F4", "STM32F4", "STM32F4", "STM32F4"]
    }

    base_circuit = create_pid_controller()

    comparison_results = analyzer.compare_hardware_implementations(
        base_circuit,
        hardware_alternatives
    )

    return comparison_results


if __name__ == "__main__":
    # Example usage
    print("Creating PID Controller...")
    pid_circuit = create_pid_controller(kp=2.0, ki=0.5, kd=0.1)

    print("Running simulation...")
    results = simulate_pid_response(pid_circuit)

    print("Simulation completed:")
    print(f"  Final time: {results['simulation_results']['simulation_time']:.2e}s")
    print(f"  Events executed: {results['simulation_results']['events_executed']}")

    print("\nAnalyzing different implementations...")
    comparison = analyze_pid_implementations()

    for impl_name, report in comparison.items():
        print(f"\n{impl_name}:")
        print(f"  Execution time: {report.total_execution_time:.2e}s")
        print(f"  Power consumption: {report.total_power_consumption:.4f}W")
        print(f"  Critical path delay: {report.critical_path_delay:.2e}s")