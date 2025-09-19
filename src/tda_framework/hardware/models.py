"""Hardware models for different implementation technologies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional
import random


@dataclass
class TimingCharacteristics:
    """Timing characteristics for hardware components."""
    propagation_delay_min: Decimal
    propagation_delay_max: Decimal
    setup_time: Decimal = Decimal('0')
    hold_time: Decimal = Decimal('0')
    slew_rate: Optional[float] = None  # V/s
    bandwidth: Optional[float] = None  # Hz
    noise_floor: float = 0.0  # RMS noise voltage


class HardwareModel(ABC):
    """Abstract base class for hardware implementation models."""

    def __init__(self, name: str, characteristics: TimingCharacteristics) -> None:
        self.name = name
        self.characteristics = characteristics

    @abstractmethod
    def get_propagation_delay(self) -> Decimal:
        """Get propagation delay for this instance."""
        pass

    @abstractmethod
    def get_power_consumption(self, switching_frequency: float = 0.0) -> float:
        """Get power consumption in watts."""
        pass

    @abstractmethod
    def get_accuracy(self) -> float:
        """Get accuracy specification as relative error."""
        pass

    def add_noise(self, signal_value: float) -> float:
        """Add hardware noise to signal value."""
        if self.characteristics.noise_floor > 0:
            noise = random.gauss(0, self.characteristics.noise_floor)
            return signal_value + noise
        return signal_value


class OpAmpModel(HardwareModel):
    """Operational amplifier hardware model."""

    def __init__(
        self,
        name: str,
        slew_rate: float,  # V/μs
        gbw_product: float,  # GHz
        input_offset_voltage: float = 1e-3,  # mV
        input_bias_current: float = 1e-12,  # pA
        supply_current: float = 1e-3,  # mA
        supply_voltage: float = 5.0  # V
    ) -> None:
        # Convert to base units
        slew_rate_vs = slew_rate * 1e6  # V/s
        bandwidth_hz = gbw_product * 1e9  # Hz
        noise_floor = input_offset_voltage / 1000  # Convert to V RMS

        characteristics = TimingCharacteristics(
            propagation_delay_min=Decimal('1e-9'),  # 1ns minimum
            propagation_delay_max=Decimal('100e-9'),  # 100ns maximum
            slew_rate=slew_rate_vs,
            bandwidth=bandwidth_hz,
            noise_floor=noise_floor
        )

        super().__init__(name, characteristics)
        self.input_offset_voltage = input_offset_voltage
        self.input_bias_current = input_bias_current
        self.supply_current = supply_current
        self.supply_voltage = supply_voltage

    def get_propagation_delay(self) -> Decimal:
        """Calculate propagation delay based on slew rate and bandwidth."""
        # Delay dominated by slew rate or bandwidth
        if self.characteristics.slew_rate and self.characteristics.bandwidth:
            slew_delay = Decimal('1') / Decimal(str(self.characteristics.slew_rate))
            bandwidth_delay = Decimal('1') / (Decimal('2') * Decimal(str(self.characteristics.bandwidth)))
            return max(slew_delay, bandwidth_delay, self.characteristics.propagation_delay_min)
        return self.characteristics.propagation_delay_min

    def get_power_consumption(self, switching_frequency: float = 0.0) -> float:
        """Calculate power consumption."""
        static_power = self.supply_current * self.supply_voltage
        # Dynamic power is typically negligible for op-amps
        return static_power

    def get_accuracy(self) -> float:
        """Get accuracy based on offset voltage."""
        return abs(self.input_offset_voltage) / self.supply_voltage


class FPGAModel(HardwareModel):
    """FPGA implementation model."""

    def __init__(
        self,
        name: str,
        logic_levels: int,
        clock_frequency: float,  # MHz
        lookup_tables: int = 1,
        flip_flops: int = 1,
        dsp_blocks: int = 0,
        power_per_mhz: float = 1e-3  # W/MHz
    ) -> None:
        # Calculate timing from clock frequency
        clock_period = Decimal('1') / Decimal(str(clock_frequency * 1e6))

        characteristics = TimingCharacteristics(
            propagation_delay_min=clock_period / 4,
            propagation_delay_max=clock_period,
            setup_time=clock_period / 10,
            hold_time=clock_period / 20,
            noise_floor=0.001  # Digital noise floor
        )

        super().__init__(name, characteristics)
        self.logic_levels = logic_levels
        self.clock_frequency = clock_frequency
        self.lookup_tables = lookup_tables
        self.flip_flops = flip_flops
        self.dsp_blocks = dsp_blocks
        self.power_per_mhz = power_per_mhz

    def get_propagation_delay(self) -> Decimal:
        """Calculate propagation delay based on logic levels."""
        base_delay = self.characteristics.propagation_delay_min
        logic_delay = base_delay * Decimal(str(self.logic_levels))
        return logic_delay

    def get_power_consumption(self, switching_frequency: float = 0.0) -> float:
        """Calculate FPGA power consumption."""
        static_power = 0.1  # Base static power in watts
        dynamic_power = self.power_per_mhz * switching_frequency * 1e-6
        resource_power = (self.lookup_tables * 0.001 +
                         self.flip_flops * 0.0005 +
                         self.dsp_blocks * 0.01)
        return static_power + dynamic_power + resource_power

    def get_accuracy(self) -> float:
        """Get accuracy based on digital precision."""
        # Accuracy depends on bit width used in implementation
        return 2 ** (-16)  # Assume 16-bit precision


class MicrocontrollerModel(HardwareModel):
    """Microcontroller implementation model."""

    def __init__(
        self,
        name: str,
        cpu_frequency: float,  # MHz
        instruction_cycles: int = 1,
        adc_resolution: int = 12,
        dac_resolution: int = 12,
        power_consumption: float = 0.01  # W
    ) -> None:
        # Calculate timing from CPU frequency and instruction cycles
        instruction_time = Decimal(str(instruction_cycles)) / Decimal(str(cpu_frequency * 1e6))

        characteristics = TimingCharacteristics(
            propagation_delay_min=instruction_time,
            propagation_delay_max=instruction_time * 10,  # Complex operations
            setup_time=instruction_time / 4,
            hold_time=instruction_time / 4,
            noise_floor=0.01  # ADC quantization noise
        )

        super().__init__(name, characteristics)
        self.cpu_frequency = cpu_frequency
        self.instruction_cycles = instruction_cycles
        self.adc_resolution = adc_resolution
        self.dac_resolution = dac_resolution
        self.base_power_consumption = power_consumption

    def get_propagation_delay(self) -> Decimal:
        """Get instruction execution delay."""
        return self.characteristics.propagation_delay_min * Decimal(str(self.instruction_cycles))

    def get_power_consumption(self, switching_frequency: float = 0.0) -> float:
        """Calculate microcontroller power consumption."""
        # Power scales with CPU frequency
        frequency_factor = self.cpu_frequency / 100.0  # Normalize to 100 MHz
        dynamic_factor = 1.0 + (switching_frequency / 1e6) * 0.1  # 10% increase per MHz
        return self.base_power_consumption * frequency_factor * dynamic_factor

    def get_accuracy(self) -> float:
        """Get accuracy based on ADC/DAC resolution."""
        return 2 ** (-min(self.adc_resolution, self.dac_resolution))


class DiscreteAnalogModel(HardwareModel):
    """Discrete analog component model."""

    def __init__(
        self,
        name: str,
        component_type: str,  # "resistor", "capacitor", "inductor", "transistor"
        tolerance: float = 0.01,  # 1% tolerance
        temperature_coefficient: float = 100e-6,  # ppm/°C
        parasitic_capacitance: float = 1e-12,  # pF
        thermal_noise: float = 4.14e-21  # Johnson noise constant
    ) -> None:
        # Calculate timing based on parasitics
        delay = Decimal(str(parasitic_capacitance * 1e6))  # Rough RC delay

        characteristics = TimingCharacteristics(
            propagation_delay_min=delay,
            propagation_delay_max=delay * 10,
            noise_floor=thermal_noise ** 0.5
        )

        super().__init__(name, characteristics)
        self.component_type = component_type
        self.tolerance = tolerance
        self.temperature_coefficient = temperature_coefficient
        self.parasitic_capacitance = parasitic_capacitance

    def get_propagation_delay(self) -> Decimal:
        """Get propagation delay based on component type."""
        base_delay = self.characteristics.propagation_delay_min

        # Different components have different speed characteristics
        if self.component_type == "transistor":
            return base_delay
        elif self.component_type == "resistor":
            return base_delay * 2
        elif self.component_type in ["capacitor", "inductor"]:
            return base_delay * 5
        else:
            return base_delay

    def get_power_consumption(self, switching_frequency: float = 0.0) -> float:
        """Calculate discrete component power consumption."""
        # Most discrete components don't consume DC power
        if self.component_type == "transistor":
            return 1e-6  # 1μW for active devices
        return 0.0

    def get_accuracy(self) -> float:
        """Get accuracy based on component tolerance."""
        return self.tolerance


class HardwareLibrary:
    """Library of predefined hardware models."""

    def __init__(self) -> None:
        self._models: Dict[str, HardwareModel] = {}
        self._initialize_standard_models()

    def _initialize_standard_models(self) -> None:
        """Initialize library with standard hardware models."""
        # Op-amp models
        self.register_model(OpAmpModel("LM741", slew_rate=0.5, gbw_product=1.0))
        self.register_model(OpAmpModel("LF356", slew_rate=12.0, gbw_product=5.0))
        self.register_model(OpAmpModel("OPA847", slew_rate=950.0, gbw_product=3.9))

        # FPGA models
        self.register_model(FPGAModel("Spartan-6", logic_levels=4, clock_frequency=100))
        self.register_model(FPGAModel("Virtex-7", logic_levels=8, clock_frequency=500))
        self.register_model(FPGAModel("Zynq-7000", logic_levels=6, clock_frequency=200))

        # Microcontroller models
        self.register_model(MicrocontrollerModel("ATmega328", cpu_frequency=16))
        self.register_model(MicrocontrollerModel("STM32F4", cpu_frequency=168))
        self.register_model(MicrocontrollerModel("ESP32", cpu_frequency=240))

        # Discrete analog models
        self.register_model(DiscreteAnalogModel("1% Resistor", "resistor", tolerance=0.01))
        self.register_model(DiscreteAnalogModel("C0G Capacitor", "capacitor", tolerance=0.05))
        self.register_model(DiscreteAnalogModel("2N2222 Transistor", "transistor", tolerance=0.20))

    def register_model(self, model: HardwareModel) -> None:
        """Register a hardware model in the library."""
        self._models[model.name] = model

    def get_model(self, name: str) -> HardwareModel:
        """Get hardware model by name."""
        if name not in self._models:
            raise KeyError(f"Hardware model '{name}' not found in library")
        return self._models[name]

    def list_models(self) -> Dict[str, str]:
        """List all available models with their types."""
        return {name: model.__class__.__name__ for name, model in self._models.items()}

    def get_models_by_type(self, model_type: str) -> Dict[str, HardwareModel]:
        """Get all models of a specific type."""
        type_map = {
            "opamp": OpAmpModel,
            "fpga": FPGAModel,
            "microcontroller": MicrocontrollerModel,
            "discrete": DiscreteAnalogModel
        }

        if model_type.lower() not in type_map:
            raise ValueError(f"Unknown model type: {model_type}")

        target_type = type_map[model_type.lower()]
        return {name: model for name, model in self._models.items()
                if isinstance(model, target_type)}