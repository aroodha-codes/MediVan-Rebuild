"""
battery_manager.py -- Mission admission control, charging policy, and
battery state estimation from real sensors.

Implements the six operating rules as a single decision point, so mission
acceptance cannot drift out of step with charging behaviour.

  R1  battery >= ACCEPT_PCT              -> accept new missions
  R2  battery <  ACCEPT_PCT, idle        -> reject, return to dock
  R3  battery drops below mid-mission    -> finish CURRENT delivery only,
                                            then dock; accept nothing further
  R4  emergency missions override the threshold; dock immediately after
  R5  charge automatically once docked; expose %, status, time remaining
  R6  stop charging at CHARGE_STOP_PCT (95), never 100, then READY

Charging stops short of full deliberately: Li-ion cycled to 100 % ages faster,
and the 18650 pack has no balancing circuitry.

SENSOR-BACKED STATE (added for ADS1115 + INA219)
-----------------------------------------------
`update_from_sensor()` accepts a BatteryReading and derives a BatteryState.
The six rules above are unchanged and still operate on a percentage, so
simulation and hardware share one policy.

WHAT THE PERCENTAGE IS, AND IS NOT
----------------------------------
It is an ESTIMATE from pack voltage against a generic 2S Li-ion curve. It
is not coulomb counting. Two consequences worth stating plainly:

  * Li-ion voltage is nearly flat between roughly 40 % and 80 %, so a small
    voltage error becomes a large percentage error in the middle of the
    range.
  * Terminal voltage sags under load. Motor current can make a pack at 60 %
    read as 30 % while driving. IR compensation using the measured current
    reduces this, but the compensation resistance is itself an estimate
    until measured on the actual pack.

Accurate state of charge needs coulomb counting: integrate INA219 current
over time from a known starting point. The INA219 makes that possible and
it is the natural next step; it is NOT implemented here.

CHARGING IS DETECTED, NOT CONTROLLED
------------------------------------
There is no charger under Raspberry Pi control on this vehicle. ADS1115 and
INA219 are monitoring devices only. `charge_step()` remains a SIMULATION
model of charging progress; on hardware, charging state comes from measured
current direction and the actual charge is managed by whatever external
charger or BMS is fitted. Rule R6 ("stop charging at 95 %") is therefore a
simulator behaviour and a policy statement, not something this software can
enforce on the physical pack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

try:
    from config import (
        BATTERY_CURVE_2S, BATTERY_FULL_V, BATTERY_EMPTY_V,
        BATTERY_CHARGE_CURRENT_A, BATTERY_DISCHARGE_CURRENT_A,
        BATTERY_FULL_PCT, BATTERY_FULL_TAPER_A,
        BATTERY_LOW_PCT, BATTERY_CRITICAL_PCT,
        BATTERY_EMA_ALPHA, BATTERY_STATE_HYSTERESIS_POLLS,
        BATTERY_PCT_HYSTERESIS, BATTERY_IR_COMPENSATION_OHMS,
        BATTERY_REST_CURRENT_A, BATTERY_SENSOR_FAIL_LIMIT,
        INA219_SEES_CHARGE_CURRENT,
    )
except ImportError:                      # standalone import in a bare test
    BATTERY_CURVE_2S = ((6.0, 0.0), (7.4, 45.0), (8.4, 100.0))
    BATTERY_FULL_V, BATTERY_EMPTY_V = 8.4, 6.0
    BATTERY_CHARGE_CURRENT_A, BATTERY_DISCHARGE_CURRENT_A = -0.05, 0.05
    BATTERY_FULL_PCT, BATTERY_FULL_TAPER_A = 95.0, 0.10
    BATTERY_LOW_PCT, BATTERY_CRITICAL_PCT = 30.0, 12.0
    BATTERY_EMA_ALPHA, BATTERY_STATE_HYSTERESIS_POLLS = 0.20, 3
    BATTERY_PCT_HYSTERESIS, BATTERY_IR_COMPENSATION_OHMS = 3.0, 0.10
    BATTERY_REST_CURRENT_A, BATTERY_SENSOR_FAIL_LIMIT = 0.15, 3
    INA219_SEES_CHARGE_CURRENT = True


class BatteryState(Enum):
    """Reported pack condition.

    UNKNOWN is a first-class outcome, not an error code: it is what the
    system reports when sensors are unavailable or readings are not
    trustworthy. Reporting a fabricated percentage would be worse.
    """
    UNKNOWN = "unknown"
    DISCHARGING = "discharging"
    CHARGING = "charging"
    FULL = "full"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class BatteryReading:
    """One sample from the power-sensing hardware.

    `valid` False means the read failed; voltage/current/power are then
    meaningless and must not be displayed.
    """
    voltage: float = 0.0          # pack volts (2S: ~6.0 - 8.4)
    current: float = 0.0          # amps; POSITIVE = discharge
    power: float = 0.0            # watts
    valid: bool = False
    source: str = "none"          # "ads1115+ina219" | "ina219" | "sim" | "none"
    timestamp: float = field(default_factory=time.time)


def voltage_to_percent(voltage: float,
                       curve: Sequence[Tuple[float, float]] = BATTERY_CURVE_2S
                       ) -> float:
    """Interpolate state of charge from pack voltage on a 2S Li-ion curve.

    The curve is for a 2S pack: 8.4 V is 100 %, 6.0 V is 0 %. Applying a 1S
    curve (4.2 V = 100 %) to this pack would be wrong by a factor of two.

    Returns an ESTIMATE. See the module docstring for why voltage-based
    state of charge is imprecise in the mid range and under load.
    """
    if voltage <= curve[0][0]:
        return 0.0
    if voltage >= curve[-1][0]:
        return 100.0
    for i in range(len(curve) - 1):
        v0, p0 = curve[i]
        v1, p1 = curve[i + 1]
        if v0 <= voltage <= v1:
            if v1 == v0:
                return p0
            return p0 + (p1 - p0) * (voltage - v0) / (v1 - v0)
    return 0.0


def compensate_for_load(voltage: float, current: float,
                        r_internal: float = BATTERY_IR_COMPENSATION_OHMS
                        ) -> float:
    """Estimate resting voltage from a loaded measurement.

        V_rest ~= V_measured + I_discharge * R_internal

    Only applied while discharging (positive current). The resistance is an
    estimate until measured on the actual pack, so this reduces the load
    error rather than eliminating it.
    """
    if current <= 0.0 or r_internal <= 0.0:
        return voltage
    return voltage + current * r_internal


class BatteryVerdict(Enum):
    ACCEPT = "accept"
    REJECT_LOW = "reject_low"
    FINISH_THEN_DOCK = "finish_then_dock"
    ACCEPT_EMERGENCY = "accept_emergency"


@dataclass
class ChargeStatus:
    charging: bool = False
    percent: float = 0.0
    target: float = 95.0
    eta_seconds: Optional[float] = None
    complete: bool = False


class BatteryManager:
    ACCEPT_PCT = 30.0
    CHARGE_STOP_PCT = 95.0
    CRITICAL_PCT = 12.0

    def __init__(self, charge_rate_pct_per_s: float = 0.35) -> None:
        self.charge_rate = charge_rate_pct_per_s
        self.cycles = 0
        self._was_charging = False
        self.lockout = False          # set by R3/R4: no further missions

        # ── sensor-backed state (inert until update_from_sensor is called) ──
        self.state: BatteryState = BatteryState.UNKNOWN
        self.reading: BatteryReading = BatteryReading()
        self.sensed_pct: Optional[float] = None   # None = no sensor data yet
        self._v_ema: Optional[float] = None
        #: Voltage step, in volts, beyond which the smoothing filter is
        #: reset rather than ramped. A pack cannot jump 0.5 V between two
        #: one-second samples in normal operation, so such a step means the
        #: measurement context changed -- a charger connected, the load
        #: removed, or a fresh session. Ramping across it would report a
        #: percentage belonging to neither state for several seconds.
        self._ema_reset_delta: float = 0.5
        self._candidate: Optional[BatteryState] = None
        self._candidate_count: int = 0
        self._fail_count: int = 0
        self._was_low: bool = False

    # ── sensor-backed state estimation ─────────────────────────
    def update_from_sensor(self, reading: BatteryReading) -> BatteryState:
        """Fold one sensor sample into the estimated state.

        Returns the CONFIRMED state, which only changes after a candidate
        has held for BATTERY_STATE_HYSTERESIS_POLLS consecutive samples.
        Without that, PWM ripple and motor transients flip the reported
        state several times a second.
        """
        self.reading = reading

        # ── invalid reads ──────────────────────────────────────
        if not reading.valid:
            self._fail_count += 1
            if self._fail_count >= BATTERY_SENSOR_FAIL_LIMIT:
                # Do not guess. UNKNOWN is the honest answer.
                self.state = BatteryState.UNKNOWN
                self.sensed_pct = None
                self._candidate, self._candidate_count = None, 0
            return self.state
        self._fail_count = 0

        # ── smooth the voltage ─────────────────────────────────
        v = reading.voltage
        if self._v_ema is None or abs(v - self._v_ema) > self._ema_reset_delta:
            self._v_ema = v
        else:
            self._v_ema = (BATTERY_EMA_ALPHA * v
                           + (1.0 - BATTERY_EMA_ALPHA) * self._v_ema)

        # ── percentage, load-compensated while discharging ─────
        v_rest = compensate_for_load(self._v_ema, reading.current)
        pct = voltage_to_percent(v_rest)
        self.sensed_pct = pct

        # ── classify ───────────────────────────────────────────
        observed = self._classify(pct, reading.current)

        # ── hysteresis on the state itself ─────────────────────
        if observed == self.state:
            self._candidate, self._candidate_count = None, 0
            return self.state
        if observed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = observed, 1
        if self._candidate_count >= BATTERY_STATE_HYSTERESIS_POLLS:
            self.state = observed
            self._candidate, self._candidate_count = None, 0
        return self.state

    def _classify(self, pct: float, current: float) -> BatteryState:
        """Map percentage and current direction onto a state.

        Current direction decides charging, never voltage alone: a pack
        rising in voltage may simply be recovering after a load was
        removed, which is not charging.
        """
        charging = (INA219_SEES_CHARGE_CURRENT
                    and current <= BATTERY_CHARGE_CURRENT_A)

        if charging:
            # FULL only when charge current has tapered, which is what
            # actually indicates a finished charge on a CC/CV charger.
            if pct >= BATTERY_FULL_PCT and abs(current) <= BATTERY_FULL_TAPER_A:
                return BatteryState.FULL
            return BatteryState.CHARGING

        # Not charging: severity first, so a critical pack is never
        # reported merely as DISCHARGING.
        if pct <= BATTERY_CRITICAL_PCT:
            self._was_low = True
            return BatteryState.CRITICAL

        # Percentage hysteresis: once LOW, stay LOW until clearly recovered.
        low_exit = BATTERY_LOW_PCT + BATTERY_PCT_HYSTERESIS
        if pct <= BATTERY_LOW_PCT or (self._was_low and pct < low_exit):
            self._was_low = True
            return BatteryState.LOW
        self._was_low = False

        if pct >= BATTERY_FULL_PCT:
            return BatteryState.FULL
        if current >= BATTERY_DISCHARGE_CURRENT_A:
            return BatteryState.DISCHARGING
        # Valid data, but the pack is neither charging nor meaningfully
        # discharging -- powered but idle.
        return BatteryState.DISCHARGING

    @property
    def has_sensor_data(self) -> bool:
        """True when a trustworthy percentage is available from hardware."""
        return self.sensed_pct is not None and self.state != BatteryState.UNKNOWN

    # ── admission control ──────────────────────────────────────
    def evaluate(self, battery_pct: float, *, mission_active: bool,
                 emergency: bool = False) -> BatteryVerdict:
        if emergency:
            self.lockout = True                       # R4: dock straight after
            return BatteryVerdict.ACCEPT_EMERGENCY
        if battery_pct >= self.ACCEPT_PCT and not self.lockout:
            return BatteryVerdict.ACCEPT               # R1
        if mission_active:
            self.lockout = True                        # R3
            return BatteryVerdict.FINISH_THEN_DOCK
        return BatteryVerdict.REJECT_LOW               # R2

    def may_accept(self, battery_pct: float) -> bool:
        return battery_pct >= self.ACCEPT_PCT and not self.lockout

    # ── charging ───────────────────────────────────────────────
    def charge_step(self, battery_pct: float, dt: float) -> tuple:
        """Advance charging. Returns (new_pct, ChargeStatus)."""
        if battery_pct >= self.CHARGE_STOP_PCT:        # R6
            if self._was_charging:
                self.cycles += 1
                self._was_charging = False
            self.lockout = False                       # ready again
            return battery_pct, ChargeStatus(
                charging=False, percent=battery_pct,
                target=self.CHARGE_STOP_PCT, eta_seconds=0.0, complete=True)

        self._was_charging = True
        new = min(self.CHARGE_STOP_PCT, battery_pct + self.charge_rate * dt)
        remaining = max(0.0, self.CHARGE_STOP_PCT - new)
        eta = remaining / self.charge_rate if self.charge_rate > 0 else None
        return new, ChargeStatus(charging=True, percent=new,
                                 target=self.CHARGE_STOP_PCT,
                                 eta_seconds=eta, complete=False)