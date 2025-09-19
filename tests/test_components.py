"""Tests for analog computation components."""

import pytest
from decimal import Decimal

from tda_framework.components.analog import (
    Integrator, SummingAmplifier, CoefficientMultiplier,
    FunctionGenerator, Comparator, SampleHold
)
from tda_framework.components.base import Component
from tda_framework.core.simulation import Simulator
from tda_framework.core.signals import Signal


class TestIntegrator:
    """Test integrator component."""

    def test_integrator_creation(self):
        """Test basic integrator creation."""
        integrator = Integrator("test_int", gain=2.0, initial_condition=1.0)

        assert integrator.name == "test_int"
        assert integrator.gain == 2.0
        assert integrator.initial_condition == 1.0
        assert "input" in integrator._ports
        assert "output" in integrator._ports
        assert "reset" in integrator._ports

    def test_integrator_simulation(self):
        """Test integrator in simulation."""
        simulator = Simulator()
        integrator = Integrator("test_int", gain=1.0, initial_condition=0.0)

        # Create signals
        input_signal = simulator.create_signal("input", initial_value=1.0)
        output_signal = simulator.create_signal("output")
        reset_signal = simulator.create_signal("reset", initial_value=0.0)

        # Connect integrator
        integrator.connect_signal("input", input_signal)
        integrator.connect_signal("output", output_signal)
        integrator.connect_signal("reset", reset_signal)

        simulator.add_component(integrator)

        # Run simulation
        simulator.run_simulation(Decimal('1e-3'))

        # Check that integration occurred (output should be positive)
        assert output_signal.current_value > 0

    def test_integrator_reset(self):
        """Test integrator reset functionality."""
        integrator = Integrator("test_int", initial_condition=5.0)
        integrator._integrator_state = 10.0

        integrator.reset_integrator()

        assert integrator._integrator_state == 5.0


class TestSummingAmplifier:
    """Test summing amplifier component."""

    def test_summing_amplifier_creation(self):
        """Test basic summing amplifier creation."""
        summer = SummingAmplifier("test_summer", gains=[1.0, -1.0], num_inputs=2)

        assert summer.name == "test_summer"
        assert summer.gains == [1.0, -1.0]
        assert summer.num_inputs == 2
        assert "input_0" in summer._ports
        assert "input_1" in summer._ports
        assert "output" in summer._ports

    def test_summing_amplifier_simulation(self):
        """Test summing amplifier in simulation."""
        simulator = Simulator()
        summer = SummingAmplifier("test_summer", gains=[2.0, -1.0], num_inputs=2)

        # Create signals
        input0 = simulator.create_signal("input0", initial_value=3.0)
        input1 = simulator.create_signal("input1", initial_value=2.0)
        output = simulator.create_signal("output")

        # Connect signals
        summer.connect_signal("input_0", input0)
        summer.connect_signal("input_1", input1)
        summer.connect_signal("output", output)

        simulator.add_component(summer)

        # Run simulation
        simulator.run_simulation(Decimal('1e-6'))

        # Check output: 2.0 * 3.0 + (-1.0) * 2.0 = 6.0 - 2.0 = 4.0
        assert abs(output.current_value - 4.0) < 1e-6

    def test_gain_modification(self):
        """Test gain modification."""
        summer = SummingAmplifier("test_summer", gains=[1.0, 1.0], num_inputs=2)

        summer.set_gain(0, 5.0)
        assert summer.gains[0] == 5.0

        summer.set_gain(1, -3.0)
        assert summer.gains[1] == -3.0


class TestCoefficientMultiplier:
    """Test coefficient multiplier component."""

    def test_multiplier_creation(self):
        """Test basic multiplier creation."""
        mult = CoefficientMultiplier("test_mult", coefficient=3.14)

        assert mult.name == "test_mult"
        assert mult.coefficient == 3.14
        assert "input" in mult._ports
        assert "output" in mult._ports

    def test_multiplier_simulation(self):
        """Test multiplier in simulation."""
        simulator = Simulator()
        mult = CoefficientMultiplier("test_mult", coefficient=2.5)

        # Create signals
        input_signal = simulator.create_signal("input", initial_value=4.0)
        output_signal = simulator.create_signal("output")

        # Connect signals
        mult.connect_signal("input", input_signal)
        mult.connect_signal("output", output_signal)

        simulator.add_component(mult)

        # Run simulation
        simulator.run_simulation(Decimal('1e-6'))

        # Check output: 4.0 * 2.5 = 10.0
        assert abs(output_signal.current_value - 10.0) < 1e-6


class TestFunctionGenerator:
    """Test function generator component."""

    def test_function_generator_creation(self):
        """Test basic function generator creation."""
        gen = FunctionGenerator(
            "test_gen",
            function_type="sine",
            amplitude=2.0,
            frequency=1000.0
        )

        assert gen.name == "test_gen"
        assert gen.function_type == "sine"
        assert gen.amplitude == 2.0
        assert gen.frequency == 1000.0

    def test_sine_function(self):
        """Test sine function generation."""
        simulator = Simulator()
        gen = FunctionGenerator("sine_gen", "sine", amplitude=1.0, frequency=1.0)

        output_signal = simulator.create_signal("output")
        gen.connect_signal("output", output_signal)

        simulator.add_component(gen)

        # Run simulation for 1/4 period (should be at peak)
        simulator.run_simulation(Decimal('0.25'))

        # At t=0.25s, sin(2π*1*0.25) = sin(π/2) = 1.0
        # Allow for some numerical precision
        assert abs(output_signal.current_value - 1.0) < 0.1

    def test_invalid_function_type(self):
        """Test invalid function type."""
        with pytest.raises(ValueError):
            FunctionGenerator("bad_gen", function_type="invalid")


class TestComparator:
    """Test comparator component."""

    def test_comparator_creation(self):
        """Test basic comparator creation."""
        comp = Comparator(
            "test_comp",
            threshold=0.0,
            hysteresis=0.1,
            output_high=5.0,
            output_low=0.0
        )

        assert comp.name == "test_comp"
        assert comp.threshold == 0.0
        assert comp.hysteresis == 0.1

    def test_comparator_operation(self):
        """Test comparator operation."""
        simulator = Simulator()
        comp = Comparator("test_comp", threshold=2.0, output_high=5.0, output_low=0.0)

        # Create signals
        pos_input = simulator.create_signal("pos_input", initial_value=3.0)
        neg_input = simulator.create_signal("neg_input", initial_value=1.0)
        output = simulator.create_signal("output")

        # Connect signals
        comp.connect_signal("input_positive", pos_input)
        comp.connect_signal("input_negative", neg_input)
        comp.connect_signal("output", output)

        simulator.add_component(comp)

        # Run simulation
        simulator.run_simulation(Decimal('1e-6'))

        # pos_input (3.0) - neg_input (1.0) = 2.0, which equals threshold
        # Should be high
        assert abs(output.current_value - 5.0) < 1e-6


class TestSampleHold:
    """Test sample and hold component."""

    def test_sample_hold_creation(self):
        """Test basic sample and hold creation."""
        sh = SampleHold("test_sh", hold_time=1e-6)

        assert sh.name == "test_sh"
        assert sh.hold_time == 1e-6
        assert "input" in sh._ports
        assert "sample_trigger" in sh._ports
        assert "output" in sh._ports

    def test_sample_hold_operation(self):
        """Test sample and hold operation."""
        simulator = Simulator()
        sh = SampleHold("test_sh", hold_time=1e-6)

        # Create signals
        input_signal = simulator.create_signal("input", initial_value=3.14)
        trigger_signal = simulator.create_signal("trigger", initial_value=0.0)
        output_signal = simulator.create_signal("output")

        # Connect signals
        sh.connect_signal("input", input_signal)
        sh.connect_signal("sample_trigger", trigger_signal)
        sh.connect_signal("output", output_signal)

        simulator.add_component(sh)

        # Initially, output should be 0 (no sample taken)
        simulator.run_simulation(Decimal('1e-8'))
        assert output_signal.current_value == 0.0

        # Trigger sampling
        trigger_signal.set_value(simulator.current_time, 1.0)  # High trigger
        simulator.run_simulation(Decimal('2e-6'))

        # Output should now hold the input value
        assert abs(output_signal.current_value - 3.14) < 1e-6