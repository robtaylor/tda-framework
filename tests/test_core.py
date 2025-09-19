"""Tests for core simulation components."""

import pytest
from decimal import Decimal

from tda_framework.core.events import Event, EventScheduler
from tda_framework.core.signals import Signal, SignalType
from tda_framework.core.simulation import Simulator, SimulationConfig


class TestEventScheduler:
    """Test event scheduling functionality."""

    def test_event_creation(self):
        """Test basic event creation."""
        callback = lambda: 42
        event = Event(Decimal('1.0'), 0, callback)

        assert event.timestamp == Decimal('1.0')
        assert event.priority == 0
        assert event.execute() == 42

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = EventScheduler(time_resolution=Decimal('1e-9'))

        assert scheduler.current_time == Decimal('0')
        assert scheduler.time_resolution == Decimal('1e-9')
        assert not scheduler.has_events()

    def test_event_scheduling(self):
        """Test event scheduling and execution."""
        scheduler = EventScheduler()
        executed_values = []

        def callback1():
            executed_values.append(1)

        def callback2():
            executed_values.append(2)

        # Schedule events
        scheduler.schedule_event(Decimal('1.0'), callback1)
        scheduler.schedule_event(Decimal('0.5'), callback2)

        assert scheduler.has_events()
        assert scheduler.peek_next_time() == Decimal('0.5')

        # Execute events
        scheduler.execute_next()
        assert executed_values == [2]
        assert scheduler.current_time == Decimal('0.5')

        scheduler.execute_next()
        assert executed_values == [2, 1]
        assert scheduler.current_time == Decimal('1.0')

    def test_execute_until(self):
        """Test executing events until specific time."""
        scheduler = EventScheduler()
        executed_count = [0]

        def increment():
            executed_count[0] += 1

        # Schedule multiple events
        for i in range(5):
            scheduler.schedule_event(Decimal(str(i + 1)), increment)

        # Execute until time 3
        scheduler.execute_until(Decimal('3.0'))

        assert executed_count[0] == 3
        assert scheduler.current_time == Decimal('3.0')


class TestSignal:
    """Test signal functionality."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal("test_signal", SignalType.VOLTAGE, initial_value=5.0)

        assert signal.name == "test_signal"
        assert signal.signal_type == SignalType.VOLTAGE
        assert signal.current_value == 5.0

    def test_signal_value_setting(self):
        """Test setting signal values."""
        signal = Signal("test_signal")

        signal.set_value(Decimal('1.0'), 3.14)
        assert signal.current_value == 3.14

        signal.set_value(Decimal('2.0'), 2.71)
        assert signal.current_value == 2.71

        # Check history
        history = signal.get_history()
        assert len(history) == 2
        assert history[0].timestamp == Decimal('1.0')
        assert history[0].value == 3.14

    def test_signal_interpolation(self):
        """Test signal value interpolation."""
        signal = Signal("test_signal")

        signal.set_value(Decimal('0.0'), 0.0)
        signal.set_value(Decimal('2.0'), 10.0)

        # Test interpolation at midpoint
        interpolated = signal.get_value_at(Decimal('1.0'))
        assert interpolated == 5.0

    def test_signal_observers(self):
        """Test signal observer functionality."""
        signal = Signal("test_signal")
        observed_values = []

        def observer(sig, value):
            observed_values.append(value)

        signal.add_observer(observer)
        signal.set_value(Decimal('1.0'), 42.0)

        assert observed_values == [42.0]


class TestSimulator:
    """Test simulation functionality."""

    def test_simulator_creation(self):
        """Test basic simulator creation."""
        config = SimulationConfig(time_resolution=Decimal('1e-9'))
        simulator = Simulator(config)

        assert simulator.current_time == Decimal('0')
        assert not simulator.is_running

    def test_signal_creation(self):
        """Test signal creation in simulator."""
        simulator = Simulator()

        signal = simulator.create_signal("test_signal", initial_value=1.5)

        assert signal.name == "test_signal"
        assert signal.current_value == 1.5
        assert "test_signal" in simulator.signal_bus.list_signals()

    def test_event_scheduling(self):
        """Test event scheduling in simulator."""
        simulator = Simulator()
        executed = [False]

        def test_callback():
            executed[0] = True

        simulator.schedule_event(Decimal('1e-6'), test_callback)

        # Run simulation
        simulator.run_simulation(Decimal('2e-6'))

        assert executed[0]

    def test_simulation_reset(self):
        """Test simulation reset functionality."""
        simulator = Simulator()

        # Schedule an event and create a signal
        simulator.schedule_event(Decimal('1.0'), lambda: None)
        simulator.create_signal("test_signal")

        # Reset simulation
        simulator.reset_simulation()

        assert simulator.current_time == Decimal('0')
        assert not simulator.scheduler.has_events()

    def test_simulation_statistics(self):
        """Test simulation statistics."""
        simulator = Simulator()

        stats = simulator.get_statistics()

        assert "current_time" in stats
        assert "events_scheduled" in stats
        assert "simulation_state" in stats
        assert stats["simulation_state"] == "idle"