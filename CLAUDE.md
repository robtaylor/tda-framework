# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### PDM Commands
This project uses PDM for dependency management. Always use PDM to run commands and manage the Python environment:

- **Install dependencies**: `pdm install` (or `pdm install -G dev` for development dependencies)
- **Run tests**: `pdm run test` or `pdm run pytest`
- **Code linting**: `pdm run lint` (uses ruff)
- **Code formatting**: `pdm run format` (uses black)
- **Type checking**: `pdm run typecheck` (uses pyright)
- **Clean build artifacts**: `pdm run clean`

### Testing
- **Run all tests**: `pdm run pytest`
- **Run specific test file**: `pdm run pytest tests/test_components.py`
- **Run single test**: `pdm run pytest tests/test_components.py::TestIntegrator::test_integrator_creation`
- **Run with coverage**: `pdm run pytest --cov=tda_framework --cov-report=term-missing`

## Architecture Overview

### Core Simulation Engine
The framework implements an event-driven simulation engine with picosecond precision:

- **EventScheduler**: Priority queue-based event scheduling with Decimal timestamps
- **Simulator**: Main coordinator that manages components, signals, and the event loop
- **SignalBus**: Manages signal routing and connections between components
- **Signal**: Continuous-time signal representation with interpolation and history tracking

### Component Architecture
Components follow a port-based connection model:

- **Component (base class)**: Abstract base with port management, signal connection, and hardware model integration
- **Port**: Represents input/output connections with signal references
- **Analog Components**: Integrator, SummingAmplifier, CoefficientMultiplier, FunctionGenerator, Comparator, SampleHold
- Components connect via signals and update through the simulator's event loop

### Hardware Modeling
Realistic hardware performance modeling:

- **HardwareModel**: Abstract base for timing, power, and accuracy characteristics
- **TimingCharacteristics**: Propagation delays, setup/hold times, slew rates, bandwidth
- **Hardware-specific models**: OpAmp, FPGA, Microcontroller, and discrete component models
- Components can be instantiated with different hardware models for performance comparison

### Signal Processing
Time-domain analog signal representation:

- **Signal types**: VOLTAGE, CURRENT, TIME, FREQUENCY, DIGITAL
- **Timestamp-based updates**: Precise timing with Decimal precision
- **Value interpolation**: Linear interpolation between timestamp points
- **Observer pattern**: Components can subscribe to signal changes
- **Signal history**: Complete waveform capture for analysis

## Project Structure

```
src/tda_framework/
├── core/                   # Event-driven simulation engine
│   ├── events.py          # EventScheduler and Event classes
│   ├── signals.py         # Signal, SignalType, SignalBus
│   ├── simulation.py      # Simulator and SimulationConfig
│   └── circuit.py         # Circuit construction utilities
├── components/            # Analog computation building blocks
│   ├── base.py           # Component, Port base classes
│   └── analog.py         # Standard analog components
├── hardware/             # Hardware implementation models
│   └── models.py         # HardwareModel, TimingCharacteristics
├── analysis/             # Performance and visualization tools
│   ├── timing.py         # Hardware performance comparison
│   └── visualization.py  # Signal plotting and reports
└── examples/             # Example implementations
    ├── pid_controller.py # PID control system
    ├── filter_design.py  # Active filter designs
    └── simple_demo.py    # Basic usage demonstration
```

## Key Design Patterns

### Event-Driven Simulation
- All component updates are scheduled as events
- Precise timestamp ordering with priority handling
- Components schedule their own update events based on hardware timing models

### Hardware-Agnostic Components
- Components are independent of hardware implementation
- Hardware models provide timing, power, and accuracy characteristics
- Easy swapping between different hardware implementations (analog, digital, mixed)

### Signal Connection Model
- Components have named ports (inputs/outputs)
- Signals connect multiple component ports
- Signal updates trigger component recalculation via events

### Time Precision
- Decimal arithmetic for exact timestamp representation
- Configurable time resolution (default: 1 picosecond)
- Hardware models provide realistic propagation delays

## Testing Guidelines

- Tests use pytest framework with coverage reporting
- Component tests verify both creation and simulation behavior
- Test naming follows `test_<component>_<functionality>` pattern
- Simulation tests check signal propagation and timing accuracy
- Use `Decimal` for precise time comparisons in tests