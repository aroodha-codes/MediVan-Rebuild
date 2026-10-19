"""
battery_hw.py -- Physical battery monitoring: ADS1115 + INA219.

Reads pack voltage, current and power from the two I2C monitoring devices
fitted to the vehicle, and returns them as a BatteryReading that
BatteryManager can fold into a state estimate.

    2 x 18650 SERIES (2S1P, 7.4 V nominal, 8.4 V full)
            |
          INA219  ------> current, power, bus voltage
            |
          divider ------> ADS1115 A0 ------> pack voltage
            |
          buck ---------> Raspberry Pi 4

THESE ARE MONITORING DEVICES, NOT CHARGERS
------------------------------------------
Neither part can charge the pack, and nothing here controls charging.
Charging is DETECTED from the direction of current through the INA219
shunt. Whatever charger or BMS is fitted manages the actual charge; this
module only observes it.

RAW REGISTER ACCESS, NO EXTRA DEPENDENCIES
------------------------------------------
Both devices are driven through smbus2, which the project already uses for
the MPU6050. This avoids adding adafruit-circuitpython-ads1x15 and
-ina219 to a Raspberry Pi that is already tight on install size, and keeps
the dependency list matching hardware/imu_hw.py.

FAILURE BEHAVIOUR
-----------------
Every read path returns `BatteryReading(valid=False)` on failure rather
than a plausible-looking number. A missing sensor, a disconnected I2C line
and a bus error are all indistinguishable to software, and reporting an
invented voltage for any of them would be worse than reporting nothing:
BatteryManager degrades to UNKNOWN after BATTERY_SENSOR_FAIL_LIMIT
consecutive failures.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ADS1115_ENABLED, ADS1115_I2C_BUS, ADS1115_I2C_ADDR, ADS1115_CHANNEL,
    ADS1115_GAIN_FS_VOLTS, ADS1115_DIVIDER_RATIO, ADS1115_CALIBRATION,
    INA219_ENABLED, INA219_I2C_BUS, INA219_I2C_ADDR,
    INA219_SHUNT_OHMS, INA219_MAX_EXPECTED_A, INA219_INVERT_CURRENT,
)
from modules.battery_manager import BatteryReading

try:
    import smbus2
    _SMBUS = True
except ImportError:                                  # noqa: BLE001
    _SMBUS = False


# ══════════════════════════════════════════════════════════════
# ADS1115 -- 16-bit ADC, pack voltage through a divider
# ══════════════════════════════════════════════════════════════

_ADS_REG_CONVERSION = 0x00
_ADS_REG_CONFIG = 0x01

# Config bits (see TI datasheet SBAS444)
_ADS_OS_SINGLE = 0x8000        # begin a single conversion
_ADS_MODE_SINGLE = 0x0100      # power-down single-shot mode
_ADS_DR_128SPS = 0x0080        # 128 samples/s
_ADS_COMP_DISABLE = 0x0003     # comparator off

_ADS_MUX_SINGLE = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
_ADS_PGA = {                   # full-scale range -> PGA bits
    6.144: 0x0000, 4.096: 0x0200, 2.048: 0x0400,
    1.024: 0x0600, 0.512: 0x0800, 0.256: 0x0A00,
}


class ADS1115:
    """Single-ended pack-voltage measurement through a resistive divider."""

    def __init__(self, bus: int = ADS1115_I2C_BUS,
                 addr: int = ADS1115_I2C_ADDR,
                 channel: int = ADS1115_CHANNEL,
                 fs_volts: float = ADS1115_GAIN_FS_VOLTS,
                 divider_ratio: float = ADS1115_DIVIDER_RATIO,
                 calibration: float = ADS1115_CALIBRATION) -> None:
        self.addr = addr
        self.channel = channel
        self.fs = fs_volts
        self.ratio = divider_ratio
        self.cal = calibration
        self.available = False
        self._bus = None

        if not _SMBUS:
            print("[Battery] smbus2 not installed -- ADS1115 unavailable")
            return
        try:
            self._bus = smbus2.SMBus(bus)
            self._bus.read_byte(self.addr)           # probe
            self.available = True
            print(f"[Battery] ADS1115 at 0x{self.addr:02X}, channel A{channel}, "
                  f"divider {self.ratio:.3f}x")
        except Exception as exc:                     # noqa: BLE001
            print(f"[Battery] ADS1115 not found at 0x{addr:02X}: {exc}")
            self._bus = None

    def read_voltage(self) -> Optional[float]:
        """Pack volts, or None if the read failed.

        Returns the voltage AT THE PACK: the ADC measurement multiplied by
        the divider ratio and the calibration trim.
        """
        if not self.available or self._bus is None:
            return None
        try:
            cfg = (_ADS_OS_SINGLE
                   | _ADS_MUX_SINGLE.get(self.channel, 0x4000)
                   | _ADS_PGA.get(self.fs, 0x0200)
                   | _ADS_MODE_SINGLE | _ADS_DR_128SPS | _ADS_COMP_DISABLE)
            # Config register is big-endian; SMBus word writes are little-
            # endian, so the bytes are swapped here.
            self._bus.write_word_data(
                self.addr, _ADS_REG_CONFIG,
                ((cfg & 0xFF) << 8) | (cfg >> 8))
            time.sleep(0.010)                        # 128 SPS -> ~8 ms

            raw = self._bus.read_word_data(self.addr, _ADS_REG_CONVERSION)
            raw = ((raw & 0xFF) << 8) | (raw >> 8)   # swap back
            if raw > 0x7FFF:
                raw -= 0x10000                       # two's complement

            adc_volts = (raw / 32767.0) * self.fs
            return adc_volts * self.ratio * self.cal
        except Exception:                            # noqa: BLE001
            return None

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:                        # noqa: BLE001
                pass
            self._bus = None
            self.available = False


# ══════════════════════════════════════════════════════════════
# INA219 -- current, power, bus voltage
# ══════════════════════════════════════════════════════════════

_INA_REG_CONFIG = 0x00
_INA_REG_SHUNT = 0x01
_INA_REG_BUS = 0x02
_INA_REG_POWER = 0x03
_INA_REG_CURRENT = 0x04
_INA_REG_CALIBRATION = 0x05

# 32 V bus range, +/-320 mV shunt (PGA 8), 12-bit, continuous both channels
_INA_CONFIG_32V_2A = 0x399F


class INA219:
    """Current and power measurement across a shunt resistor.

    SIGN CONVENTION: positive current means discharge, i.e. energy leaving
    the pack. If the board is wired with VIN+/VIN- reversed the sign
    inverts; set INA219_INVERT_CURRENT rather than rewiring.
    """

    def __init__(self, bus: int = INA219_I2C_BUS,
                 addr: int = INA219_I2C_ADDR,
                 shunt_ohms: float = INA219_SHUNT_OHMS,
                 max_amps: float = INA219_MAX_EXPECTED_A,
                 invert: bool = INA219_INVERT_CURRENT) -> None:
        self.addr = addr
        self.shunt = shunt_ohms
        self.invert = invert
        self.available = False
        self._bus = None

        # Calibration per the TI datasheet (SBOS448):
        #   current_LSB = max_expected_current / 2^15
        #   cal = trunc(0.04096 / (current_LSB * R_shunt))
        self._current_lsb = max_amps / 32768.0
        self._power_lsb = self._current_lsb * 20.0
        cal = int(0.04096 / (self._current_lsb * self.shunt))

        if not _SMBUS:
            print("[Battery] smbus2 not installed -- INA219 unavailable")
            return
        try:
            self._bus = smbus2.SMBus(bus)
            self._write16(_INA_REG_CALIBRATION, cal)
            self._write16(_INA_REG_CONFIG, _INA_CONFIG_32V_2A)
            self.available = True
            print(f"[Battery] INA219 at 0x{self.addr:02X}, "
                  f"shunt {self.shunt} ohm, range +/-{max_amps} A")
        except Exception as exc:                     # noqa: BLE001
            print(f"[Battery] INA219 not found at 0x{addr:02X}: {exc}")
            self._bus = None

    def _write16(self, reg: int, value: int) -> None:
        self._bus.write_i2c_block_data(
            self.addr, reg, [(value >> 8) & 0xFF, value & 0xFF])

    def _read16(self, reg: int) -> int:
        hi, lo = self._bus.read_i2c_block_data(self.addr, reg, 2)
        return (hi << 8) | lo

    def _read16_signed(self, reg: int) -> int:
        v = self._read16(reg)
        return v - 0x10000 if v > 0x7FFF else v

    def read_bus_voltage(self) -> Optional[float]:
        """Voltage at V- relative to ground, in volts."""
        if not self.available or self._bus is None:
            return None
        try:
            raw = self._read16(_INA_REG_BUS)
            return (raw >> 3) * 0.004            # bits 15-3, 4 mV LSB
        except Exception:                        # noqa: BLE001
            return None

    def read_current(self) -> Optional[float]:
        """Current in amps. Positive = discharge."""
        if not self.available or self._bus is None:
            return None
        try:
            amps = self._read16_signed(_INA_REG_CURRENT) * self._current_lsb
            return -amps if self.invert else amps
        except Exception:                        # noqa: BLE001
            return None

    def read_power(self) -> Optional[float]:
        """Power in watts (always positive -- the register is unsigned)."""
        if not self.available or self._bus is None:
            return None
        try:
            return self._read16(_INA_REG_POWER) * self._power_lsb
        except Exception:                        # noqa: BLE001
            return None

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:                    # noqa: BLE001
                pass
            self._bus = None
            self.available = False


# ══════════════════════════════════════════════════════════════
# Combined source
# ══════════════════════════════════════════════════════════════

class BatteryHW:
    """Physical battery monitor. Same interface as BatterySim.

    Voltage preference: ADS1115 first, since it measures the pack directly
    through a divider. The INA219 bus voltage is the fallback -- it reads
    at the sensor's own measurement point, which may sit after wiring drop
    and therefore read slightly low under load.
    """

    def __init__(self) -> None:
        self.ads: Optional[ADS1115] = ADS1115() if ADS1115_ENABLED else None
        self.ina: Optional[INA219] = INA219() if INA219_ENABLED else None

        self.available = bool(
            (self.ads is not None and self.ads.available)
            or (self.ina is not None and self.ina.available))

        if not self.available:
            print("[Battery] ERROR: no battery sensor responded. Voltage, "
                  "current and power will report UNKNOWN.")
            print("[Battery]        Check: i2cdetect -y 1  "
                  "(expect 0x48 for ADS1115, 0x40 for INA219)")

    @property
    def source(self) -> str:
        a = self.ads is not None and self.ads.available
        i = self.ina is not None and self.ina.available
        if a and i:
            return "ads1115+ina219"
        if a:
            return "ads1115"
        if i:
            return "ina219"
        return "none"

    def read(self) -> BatteryReading:
        """One sample. `valid` False whenever voltage could not be read."""
        voltage: Optional[float] = None
        current: Optional[float] = None
        power: Optional[float] = None

        if self.ads is not None and self.ads.available:
            voltage = self.ads.read_voltage()

        if self.ina is not None and self.ina.available:
            current = self.ina.read_current()
            power = self.ina.read_power()
            if voltage is None:
                voltage = self.ina.read_bus_voltage()

        if voltage is None:
            # No voltage means no percentage. Say so rather than guess.
            return BatteryReading(valid=False, source=self.source)

        if current is None:
            current = 0.0
        if power is None:
            power = abs(voltage * current)

        return BatteryReading(voltage=voltage, current=current, power=power,
                              valid=True, source=self.source)

    def cleanup(self) -> None:
        """Release both I2C handles. Safe to call twice."""
        if self.ads is not None:
            self.ads.close()
        if self.ina is not None:
            self.ina.close()
        self.available = False


if __name__ == "__main__":
    from modules.battery_manager import BatteryManager

    hw = BatteryHW()
    mgr = BatteryManager()
    print(f"\nsource: {hw.source}\n")
    try:
        for i in range(20):
            r = hw.read()
            state = mgr.update_from_sensor(r)
            if r.valid:
                print(f"[{i:2d}] {r.voltage:5.2f} V  {r.current:+6.3f} A  "
                      f"{r.power:5.2f} W  ->  {mgr.sensed_pct:5.1f} %  "
                      f"{state.value}")
            else:
                print(f"[{i:2d}] read failed  ->  {state.value}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        hw.cleanup()