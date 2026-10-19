"""
audio_sim.py — Simulated active buzzer for audible alerts.

Maps simulation events (obstacle warning, bump contact, low battery,
dock complete, emergency stop, junction approach) to Pygame mixer
tone patterns.  Each tone has a cooldown to prevent spam.

On systems without a sound device, falls back to console logging.
"""

from __future__ import annotations

import os
import sys
import time
from enum import Enum
from typing import Dict, Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try to import pygame mixer; fallback gracefully
_AUDIO_AVAILABLE: bool = False
try:
    import pygame
    import pygame.mixer
    import numpy as np
    _AUDIO_AVAILABLE = True
except ImportError:
    pass


class AudioEvent(Enum):
    """Buzzer audio events."""
    OBSTACLE_WARNING = "obstacle_warning"
    BUMP_CONTACT = "bump_contact"
    LOW_BATTERY = "low_battery"
    DOCK_COMPLETE = "dock_complete"
    EMERGENCY_STOP = "emergency_stop"
    JUNCTION_APPROACH = "junction_approach"


# Tone definitions: (frequency_hz, duration_ms, repeats)
_TONE_DEFS: Dict[AudioEvent, tuple] = {
    AudioEvent.OBSTACLE_WARNING:  (800, 150, 2),
    AudioEvent.BUMP_CONTACT:      (1200, 100, 3),
    AudioEvent.LOW_BATTERY:       (400, 300, 2),
    AudioEvent.DOCK_COMPLETE:     (600, 200, 1),
    AudioEvent.EMERGENCY_STOP:    (1000, 500, 1),
    AudioEvent.JUNCTION_APPROACH: (500, 100, 1),
}

# Cooldown per event (seconds)
_COOLDOWN: Dict[AudioEvent, float] = {
    AudioEvent.OBSTACLE_WARNING:  1.0,
    AudioEvent.BUMP_CONTACT:      0.5,
    AudioEvent.LOW_BATTERY:       5.0,
    AudioEvent.DOCK_COMPLETE:     3.0,
    AudioEvent.EMERGENCY_STOP:    2.0,
    AudioEvent.JUNCTION_APPROACH: 2.0,
}


class AudioSim:
    """Simulated active buzzer that plays tones via Pygame mixer.

    Each event maps to a frequency/duration/repeat pattern.
    Cooldown timers prevent the same alert from spamming.
    Falls back to console prints if audio is unavailable.
    """

    def __init__(self) -> None:
        self._last_play: Dict[AudioEvent, float] = {}
        self._initialized: bool = False
        self._current_event: Optional[AudioEvent] = None

        if _AUDIO_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                self._initialized = True
            except Exception:
                self._initialized = False

    def play(self, event: AudioEvent) -> None:
        """Play the tone for the given event if cooldown has elapsed."""
        now = time.time()
        last = self._last_play.get(event, 0.0)
        cooldown = _COOLDOWN.get(event, 1.0)

        if now - last < cooldown:
            return

        self._last_play[event] = now
        self._current_event = event
        freq, dur_ms, repeats = _TONE_DEFS[event]

        if self._initialized and _AUDIO_AVAILABLE:
            try:
                self._play_tone(freq, dur_ms, repeats)
            except Exception:
                pass
        else:
            print(f"[BUZZER] {event.value} ({freq}Hz x {repeats})")

    @property
    def current_event(self) -> Optional[AudioEvent]:
        """The most recently triggered event (for logging)."""
        return self._current_event

    def clear_event(self) -> None:
        """Clear the current event flag after logging."""
        self._current_event = None

    # ── tone generation ─────────────────────────

    @staticmethod
    def _play_tone(freq: int, duration_ms: int, repeats: int) -> None:
        """Generate and play a sine-wave tone via Pygame mixer."""
        if not _AUDIO_AVAILABLE:
            return

        sample_rate = 22050
        n_samples = int(sample_rate * duration_ms / 1000.0)

        t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

        # Pygame Sound expects (n_samples,) or (n_samples, 1) for mono
        sound = pygame.mixer.Sound(buffer=wave.tobytes())
        sound.set_volume(0.3)

        for _ in range(repeats):
            sound.play()
            pygame.time.wait(duration_ms + 30)


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    if _AUDIO_AVAILABLE:
        pygame.init()
    audio = AudioSim()
    for evt in AudioEvent:
        audio.play(evt)
        print(f"Played: {evt.value}")
        time.sleep(0.5)
