"""Event scheduling and management for time domain simulation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
from decimal import Decimal

T = TypeVar("T")


@dataclass(frozen=True)
class Event(Generic[T]):
    """Represents a simulation event at a specific timestamp."""

    timestamp: Decimal
    priority: int
    callback: Callable[[], T]
    data: Any = None

    def __lt__(self, other: Event[T]) -> bool:
        """Compare events for priority queue ordering."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority

    def execute(self) -> T:
        """Execute the event callback."""
        return self.callback()


class EventScheduler:
    """Priority queue-based event scheduler for precise timestamp management."""

    def __init__(self, time_resolution: Decimal = Decimal('1e-12')) -> None:
        """Initialize scheduler with specified time resolution (default: 1ps)."""
        self._event_queue: list[Event[Any]] = []
        self._current_time = Decimal('0')
        self._time_resolution = time_resolution
        self._event_counter = 0

    @property
    def current_time(self) -> Decimal:
        """Get current simulation time."""
        return self._current_time

    @property
    def time_resolution(self) -> Decimal:
        """Get simulation time resolution."""
        return self._time_resolution

    def schedule_event(
        self,
        delay: Decimal | float,
        callback: Callable[[], T],
        priority: int = 0,
        data: Any = None
    ) -> Event[T]:
        """Schedule an event to occur after specified delay."""
        if isinstance(delay, float):
            delay = Decimal(str(delay))

        # Round delay to time resolution
        delay_steps = int(delay / self._time_resolution)
        rounded_delay = delay_steps * self._time_resolution

        timestamp = self._current_time + rounded_delay
        event = Event(timestamp, priority, callback, data)

        heapq.heappush(self._event_queue, event)
        self._event_counter += 1

        return event

    def schedule_absolute(
        self,
        timestamp: Decimal | float,
        callback: Callable[[], T],
        priority: int = 0,
        data: Any = None
    ) -> Event[T]:
        """Schedule an event at an absolute timestamp."""
        if isinstance(timestamp, float):
            timestamp = Decimal(str(timestamp))

        if timestamp < self._current_time:
            raise ValueError(f"Cannot schedule event in past: {timestamp} < {self._current_time}")

        # Round timestamp to time resolution
        timestamp_steps = int(timestamp / self._time_resolution)
        rounded_timestamp = timestamp_steps * self._time_resolution

        event = Event(rounded_timestamp, priority, callback, data)
        heapq.heappush(self._event_queue, event)
        self._event_counter += 1

        return event

    def has_events(self) -> bool:
        """Check if there are pending events."""
        return len(self._event_queue) > 0

    def peek_next_time(self) -> Decimal | None:
        """Get timestamp of next event without executing it."""
        if not self._event_queue:
            return None
        return self._event_queue[0].timestamp

    def execute_next(self) -> Any:
        """Execute the next event and advance simulation time."""
        if not self._event_queue:
            raise RuntimeError("No events to execute")

        event = heapq.heappop(self._event_queue)
        self._current_time = event.timestamp

        return event.execute()

    def execute_until(self, end_time: Decimal | float) -> None:
        """Execute all events until specified time."""
        if isinstance(end_time, float):
            end_time = Decimal(str(end_time))

        while self.has_events():
            next_time = self.peek_next_time()
            if next_time is None or next_time > end_time:
                break
            self.execute_next()

        self._current_time = max(self._current_time, end_time)

    def clear(self) -> None:
        """Clear all pending events and reset time."""
        self._event_queue.clear()
        self._current_time = Decimal('0')
        self._event_counter = 0

    def event_count(self) -> int:
        """Get total number of events scheduled."""
        return self._event_counter