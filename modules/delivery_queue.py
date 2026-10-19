"""
delivery_queue.py — Multi-goal delivery dispatch system.

Manages a queue of delivery destinations with priorities.
The robot navigates to each goal in order, returning to the
dock when the queue is empty or battery is low.

Usage in main.py:
    queue.add_goal((400, 200), "Ward A")
    goal = queue.current_goal  # returns (400, 200) or dock if empty
    queue.mark_complete()      # advance to next
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class DeliveryGoal:
    """A single delivery destination."""
    position: Tuple[int, int]
    label: str = "Unnamed"
    priority: int = 0                   # higher = more urgent
    added_time: float = 0.0
    completed: bool = False
    completed_time: float = 0.0


class DeliveryQueue:
    """Multi-goal delivery dispatch with priority ordering.

    Goals are served in priority order (highest first), with FIFO
    ordering within the same priority level.  When the queue is
    empty, `current_goal` returns None — the main loop should
    then navigate to the dock.
    """

    def __init__(self) -> None:
        self._queue: List[DeliveryGoal] = []
        self._completed: List[DeliveryGoal] = []
        self._current_index: int = 0
        self.total_deliveries: int = 0

    # ── Public API ─────────────────────────────

    def add_goal(
        self,
        position: Tuple[int, int],
        label: str = "Unnamed",
        priority: int = 0,
    ) -> None:
        """Add a delivery goal to the queue."""
        goal = DeliveryGoal(
            position=position,
            label=label,
            priority=priority,
            added_time=time.time(),
        )
        self._queue.append(goal)
        # Sort by priority (descending), then by added_time (ascending)
        self._queue.sort(key=lambda g: (-g.priority, g.added_time))
        print(f"[Delivery] Added: {label} at {position} (priority={priority})")

    def mark_complete(self) -> None:
        """Mark the current goal as completed and advance."""
        if self._queue:
            done = self._queue.pop(0)
            done.completed = True
            done.completed_time = time.time()
            self._completed.append(done)
            self.total_deliveries += 1
            print(f"[Delivery] Completed: {done.label} "
                  f"({self.total_deliveries} total)")

    @property
    def current_goal(self) -> Optional[Tuple[int, int]]:
        """Get the current delivery destination, or None if empty."""
        if self._queue:
            return self._queue[0].position
        return None

    @property
    def current_label(self) -> str:
        """Label of the current goal."""
        if self._queue:
            return self._queue[0].label
        return "None"

    @property
    def pending_count(self) -> int:
        """Number of goals remaining."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def status_text(self) -> str:
        """Short status string for HUD display."""
        if self._queue:
            return f"DELIVERY {self.total_deliveries + 1}: {self.current_label} ({self.pending_count} left)"
        return f"IDLE ({self.total_deliveries} done)"

    def get_summary(self) -> dict:
        """Return delivery statistics."""
        return {
            "total_deliveries": self.total_deliveries,
            "pending": self.pending_count,
            "completed": [
                {"label": g.label, "position": g.position,
                 "time": round(g.completed_time - g.added_time, 1)}
                for g in self._completed
            ],
        }

    def add_random_goal(
        self,
        map_width: int = 800,
        map_height: int = 600,
        is_free_fn=None,
    ) -> Tuple[int, int]:
        """Add a random valid delivery goal for testing."""
        import random
        names = ["Ward A", "Ward B", "ICU", "Pharmacy", "Lab",
                 "Radiology", "ER", "OT", "Reception", "Cafeteria"]
        label = random.choice(names)

        # Try to find a free position
        for _ in range(100):
            x = random.randint(100, map_width - 100)
            y = random.randint(100, map_height - 100)
            if is_free_fn is None or is_free_fn(x, y):
                self.add_goal((x, y), label, priority=random.randint(0, 2))
                return (x, y)

        # Fallback
        pos = (map_width // 2, map_height // 2)
        self.add_goal(pos, label)
        return pos


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    dq = DeliveryQueue()
    dq.add_goal((300, 200), "Ward A", priority=1)
    dq.add_goal((500, 100), "ICU", priority=2)
    dq.add_goal((400, 400), "Lab", priority=0)

    print(f"\nQueue status: {dq.status_text}")
    print(f"Current goal: {dq.current_goal} ({dq.current_label})")

    while not dq.is_empty:
        print(f"  Delivering to {dq.current_label}...")
        dq.mark_complete()

    print(f"\nFinal: {dq.status_text}")
    print(f"Summary: {dq.get_summary()}")
