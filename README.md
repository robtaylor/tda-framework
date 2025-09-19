# TDA Framework - Time Domain Analog Computation Simulation

A Python framework for simulating time domain analog computation systems with timestamp-based pulse modeling and hardware performance analysis.

## Features

- **Event-Driven Simulation**: Precise timestamp-based simulation with picosecond resolution
- **Analog Components**: Standard analog computation building blocks (integrators, summers, multipliers, function generators)
- **Hardware Models**: Realistic timing and power models for op-amps, FPGAs, microcontrollers, and discrete components
- **Circuit Construction**: Fluent API for building complex analog computation circuits
- **Performance Analysis**: Compare execution times and power consumption across different hardware implementations
- **Visualization**: Signal plotting, frequency analysis, and timing reports

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TDA

# Install with PDM (recommended)
pdm install

# Or install with pip in development mode
pip install -e .
```

### Basic Usage

```python
from tda_framework import *
from tda_framework.components.analog import Integrator, SummingAmplifier
from tda_framework.core.circuit import CircuitBuilder

# Create a simple integrator circuit
integrator = Integrator("main_integrator", gain=1.0, initial_condition=0.0)
circuit = (CircuitBuilder("Simple_Integrator")
          .add_component(integrator, "LM741")  # Use LM741 op-amp model
          .build())

# Set up simulation
simulator = circuit.to_simulator()
circuit.set_input_value("main_integrator", "input", 1.0)  # Step input

# Run simulation
results = simulator.run_simulation(duration=0.001)  # 1ms simulation
print(f"Final output: {circuit.get_output_value('main_integrator', 'output')}")
```

### PID Controller Example

```python
from tda_framework.examples.pid_controller import create_pid_controller, analyze_pid_implementations

# Create PID controller
pid = create_pid_controller(kp=2.0, ki=0.5, kd=0.1)

# Analyze different hardware implementations
comparison = analyze_pid_implementations()

for impl_name, report in comparison.items():
    print(f"{impl_name}: {report.total_execution_time:.2e}s, {report.total_power_consumption:.4f}W")
```

## Core Components

### Simulation Engine
- `EventScheduler`: Priority queue-based event scheduling
- `Simulator`: Main simulation coordinator
- `Signal`: Continuous-time signal representation

### Analog Components
- `Integrator`: Implements ∫x(t)dt with initial conditions
- `SummingAmplifier`: Multi-input adder with programmable gains
- `CoefficientMultiplier`: Signal scaling
- `FunctionGenerator`: Sine, cosine, exponential, and other nonlinear functions
- `Comparator`: Threshold detection with hysteresis
- `SampleHold`: Sample and hold circuits

### Hardware Models
- `OpAmpModel`: Operational amplifier characteristics (slew rate, bandwidth, noise)
- `FPGAModel`: FPGA timing and power consumption
- `MicrocontrollerModel`: MCU execution timing
- `DiscreteAnalogModel`: Discrete component models

### Analysis Tools
- `TimingAnalyzer`: Hardware performance comparison
- `SignalTracer`: Waveform recording and plotting
- `AccuracyAnalyzer`: Precision analysis with Monte Carlo methods

## Project Structure

```
src/tda_framework/
├── core/              # Core simulation engine
│   ├── events.py      # Event scheduling
│   ├── signals.py     # Signal representation
│   ├── simulation.py  # Main simulator
│   └── circuit.py     # Circuit construction
├── components/        # Analog computation components
│   ├── base.py        # Base component classes
│   └── analog.py      # Standard analog components
├── hardware/          # Hardware models
│   └── models.py      # Timing and power models
├── analysis/          # Performance analysis tools
│   ├── timing.py      # Timing analysis
│   └── visualization.py # Plotting and reports
└── examples/          # Example implementations
    ├── pid_controller.py
    └── filter_design.py
```

## Examples

The framework includes several example implementations:

### PID Controller
```python
from tda_framework.examples.pid_controller import create_pid_controller

pid = create_pid_controller(kp=1.0, ki=0.1, kd=0.01)
```

### Active Filters
```python
from tda_framework.examples.filter_design import create_lowpass_filter, create_bandpass_filter

# 2nd order Butterworth low-pass filter
lpf = create_lowpass_filter(cutoff_frequency=1000, order=2)

# Bandpass filter with Q=10
bpf = create_bandpass_filter(center_frequency=1000, q_factor=10)
```

## Hardware Performance Analysis

Compare different implementations:

```python
from tda_framework.analysis.timing import TimingAnalyzer

analyzer = TimingAnalyzer()

hardware_alternatives = {
    "High-Speed": ["OPA847", "LF356", "OPA847"],
    "Low-Power": ["LM741", "LM741", "LM741"],
    "Digital": ["STM32F4", "STM32F4", "STM32F4"]
}

results = analyzer.compare_hardware_implementations(circuit, hardware_alternatives)
```

## Development

### Running Tests

```bash
pdm run pytest
```

### Code Style

```bash
pdm run ruff check .  # Linting
pdm run black .       # Formatting
pdm run pyright       # Type checking
```

## Key Features

### Timestamp-Based Precision
- Nanosecond to picosecond time resolution
- Precise pulse timing and width modeling
- Event-driven simulation for efficiency

### Hardware-Agnostic Design
- Easy swapping of hardware models
- Performance comparison across implementations
- Power consumption analysis

### Extensible Architecture
- Plugin architecture for custom components
- Flexible hardware model system
- Modular design for easy extension

### Analysis and Visualization
- Signal waveform plotting
- Frequency domain analysis
- Timing constraint generation
- HTML report generation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citations

When using this framework in academic work, please cite:

```
TDA Framework: Time Domain Analog Computation Simulation
[Your institution/publication details]
```