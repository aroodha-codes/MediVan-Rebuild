"""
battery_sim.py -- Simulated battery sensor.

Presents the same interface as hardware/battery_hw.BatteryHW, so
HARDWARE_MODE selects between them exactly as it already does for the
camera, IMU and motor driver. Requires no ADS1115, no INA219 and no I2C
bus, so the simulator and the unit tests run unchanged on any machine.

WHAT IS SIMULATED
-----------------
The percentage is authoritative here and voltage is derived from it, which
is the reverse of the hardware path where voltage is measured and the
percentage is estimated from it. That is deliberate: the existing
simulation already models charge as a percentage, and inventing a
plausible-looking voltage trace would add nothing while risking being
mistaken for a measurement.

Current is modelled from the drive state -- a small idle draw, more while
moving, negative while charging -- so that state classification, hysteresis
and the charging/discharging logic can be exercised without hardware. The
numbers are representative, NOT measured from the real vehicle.
"""

from __future__ import annotations

import random
from typing import Optional

from config import (
    BATTERY_CURVE_2S, BATTERY_START_PCT,
    INA219_SEES_CHARGE_CURRENT,
)
from modules.battery_manager import BatteryReading


def percent_to_voltage(pct: float) -> float:
    """Inverse of voltage_to_percent -- interpolate voltage from charge.

    Used only by the simulator, to synthesise a voltage consistent with the
    percentage the simulation is already tracking.
    """
    pct = max(0.0, min(100.0, pct))
    curve = BATTERY_CURVE_2S
    if pct <= curve[0][1]:
        return curve[0][0]
    if pct >= curve[-1][1]:
        return curve[-1][0]
    for i in range(len(curve) - 1):
        v0, p0 = curve[i]
        v1, p1 = curve[i + 1]
        if p0 <= pct <= p1:
            if p1 == p0:
                return v0
            return v0 + (v1 - v0) * (pct - p0) / (p1 - p0)
    return curve[0][0]


class BatterySim:
    """Simulated ADS1115 + INA219. Interface matches BatteryHW."""

    #: Representative draws, in amps. Not measured from the vehicle.
    IDLE_CURRENT_A = 0.45          # Pi 4 plus standby electronics
    MOVING_CURRENT_A = 1.30        # plus two DC geared motors
    CHARGE_CURRENT_A = -0.90       # negative = energy entering the pack
    NOISE_A = 0.03                 # measurement noise

    def __init__(self, start_pct: float = BATTERY_START_PCT,
                 seed: Optional[int] = None) -> None:
        self.percent = start_pct
        self.moving = False
        self.charging = False
        self.available = True
        self._rng = random.Random(seed)
        #: Set True to make read() report failure, for testing the
        #: UNKNOWN degradation path without unplugging anything.
        self.fail = False

    @property
    def source(self) -> str:
        return "sim"

    def set_state(self, percent: float, *, moving: bool = False,
                  charging: bool = False) -> None:
        """Drive the simulated sensor from the simulation's own battery."""
        self.percent = max(0.0, min(100.0, percent))
        self.moving = moving
        self.charging = charging

    def read(self) -> BatteryReading:
        if self.fail or not self.available:
            return BatteryReading(valid=False, source=self.source)

        voltage = percent_to_voltage(self.percent)

        if self.charging and INA219_SEES_CHARGE_CURRENT:
            current = self.CHARGE_CURRENT_A
            # Charge current tapers as the pack approaches full, as a CC/CV
            # charger does -- this is what makes the FULL state reachable.
            if self.percent > 90.0:
                current *= max(0.05, (100.0 - self.percent) / 10.0)
            voltage += 0.25                     # charger holds the pack up
        elif self.moving:
            current = self.MOVING_CURRENT_A
            voltage -= 0.18                     # sag under motor load
        else:
            current = self.IDLE_CURRENT_A
            voltage -= 0.04

        current += self._rng.uniform(-self.NOISE_A, self.NOISE_A)
        voltage += self._rng.uniform(-0.01, 0.01)
        return BatteryReading(voltage=voltage, current=current,
                              power=abs(voltage * current),
                              valid=True, source=self.source)

    def cleanup(self) -> None:
        self.available = False