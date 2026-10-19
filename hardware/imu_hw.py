"""
imu_hw.py -- Physical MPU6500-compatible IMU Interface via I2C.

Robust Raspberry Pi MPU6500-compatible interface using block reads.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    IMU_I2C_ADDR,
    IMU_ALPHA,
    IMU_DT,
    GYRO_BIAS,
    VIB_SAFE_RMS,
    VIB_WARNING_RMS,
    VIB_DANGER_RMS,
)

from modules.imu_sim import IMUData, VibrationLevel

try:
    import smbus2
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False
    print("[WARNING] smbus2 not found. Running IMUHW in stub mode.")


# MPU6050/MPU6500-compatible register map
WHO_AM_I = 0x75
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B
GYRO_ZOUT_H = 0x47


class IMUHW:
    """Reads the MPU6500-compatible IMU via I2C on Raspberry Pi."""

    def __init__(self) -> None:

        self.yaw: float = 0.0
        self._last_time = time.time()
        self._vib_buffer: list[float] = []

        self.bus = None
        self.sensor_id = None
        self.last_read_ok = False

        if not _SMBUS_AVAILABLE:
            return

        try:
            self.bus = smbus2.SMBus(1)

            # Read device identity
            self.sensor_id = self.bus.read_byte_data(
                IMU_I2C_ADDR,
                WHO_AM_I
            )

            print(
                f"[Hardware] IMU detected at "
                f"0x{IMU_I2C_ADDR:02X}, "
                f"WHO_AM_I=0x{self.sensor_id:02X}"
            )

            # Wake sensor
            self.bus.write_byte_data(
                IMU_I2C_ADDR,
                PWR_MGMT_1,
                0x00
            )

            time.sleep(0.1)

            # Accelerometer ±2g
            self.bus.write_byte_data(
                IMU_I2C_ADDR,
                ACCEL_CONFIG,
                0x00
            )

            # Gyroscope ±250 deg/s
            self.bus.write_byte_data(
                IMU_I2C_ADDR,
                GYRO_CONFIG,
                0x00
            )

            print("[Hardware] IMU initialized via I2C.")

        except Exception as e:
            print(f"[Hardware] IMU initialization failed: {e}")
            self.bus = None

    @staticmethod
    def _signed16(high: int, low: int) -> int:
        value = (high << 8) | low

        if value & 0x8000:
            value -= 65536

        return value

    def _read_sensor_block(self):
        """
        Read accelerometer + temperature + gyro in one I2C block.

        Registers:
        0x3B through 0x48
        """

        if self.bus is None:
            return None

        # MPU6050/MPU6500 register windows are a raw sequential burst.
        # i2c_rdwr performs the register-pointer write followed by a repeated-
        # start read; this matches the standalone 14-byte transaction that
        # was verified on the physical robot. SMBus block-read framing can
        # fail with EIO on some compatible IMUs.
        reg = smbus2.i2c_msg.write(IMU_I2C_ADDR, [ACCEL_XOUT_H])
        burst = smbus2.i2c_msg.read(IMU_I2C_ADDR, 14)
        self.bus.i2c_rdwr(reg, burst)
        data = list(burst)

        if len(data) != 14:
            raise IOError(
                f"Expected 14 bytes, received {len(data)}"
            )

        acc_x = self._signed16(data[0], data[1])
        acc_y = self._signed16(data[2], data[3])
        acc_z = self._signed16(data[4], data[5])

        gyro_x = self._signed16(data[8], data[9])
        gyro_y = self._signed16(data[10], data[11])
        gyro_z = self._signed16(data[12], data[13])

        return (
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
        )

    def get_latest(self) -> IMUData:

        # Fail closed for this sample; only the successful end of this method
        # flips the flag back to True. A stale previous success must not mask
        # a current I2C failure in the main control loop.
        self.last_read_ok = False
        now = time.time()

        dt = now - self._last_time
        self._last_time = now

        # Protect against abnormal timing
        dt = max(0.001, min(0.1, dt))

        if self.bus is None:
            self.last_read_ok = False
            return IMUData(
                yaw=self.yaw,
                tilt_fault=False,
                vib_level=VibrationLevel.SAFE,
                vib_rms=0.0,
            )

        try:

            values = self._read_sensor_block()

            if values is None:
                self.last_read_ok = False
                return IMUData(yaw=self.yaw)

            (
                raw_ax,
                raw_ay,
                raw_az,
                raw_gx,
                raw_gy,
                raw_gz,
            ) = values

            # MPU6050 sensitivity:
            # Accelerometer ±2g = 16384 LSB/g
            # Gyroscope ±250°/s = 131 LSB/(°/s)

            acc_x = raw_ax / 16384.0
            acc_y = raw_ay / 16384.0
            acc_z = raw_az / 16384.0

            gyro_z = raw_gz / 131.0

            # Gyroscope yaw integration
            d_yaw = math.radians(
                gyro_z - GYRO_BIAS
            ) * dt

            self.yaw += d_yaw

            # Normalize yaw to [-pi, pi]
            self.yaw = (
                self.yaw + math.pi
            ) % (2.0 * math.pi) - math.pi

            # Pitch and roll
            pitch = math.atan2(
                acc_x,
                math.sqrt(
                    acc_y * acc_y +
                    acc_z * acc_z
                )
            )

            roll = math.atan2(
                acc_y,
                math.sqrt(
                    acc_x * acc_x +
                    acc_z * acc_z
                )
            )

            tilt_fault = (
                abs(pitch) > 0.4 or
                abs(roll) > 0.4
            )

            # Acceleration magnitude
            acc_magnitude = math.sqrt(
                acc_x * acc_x +
                acc_y * acc_y +
                acc_z * acc_z
            )

            vibration = abs(
                acc_magnitude - 1.0
            ) * 9.81

            self._vib_buffer.append(vibration)

            if len(self._vib_buffer) > 10:
                self._vib_buffer.pop(0)

            rms = (
                sum(self._vib_buffer) /
                len(self._vib_buffer)
            )

            if rms > VIB_DANGER_RMS:
                lvl = VibrationLevel.DANGER
            elif rms > VIB_WARNING_RMS:
                lvl = VibrationLevel.WARNING
            else:
                lvl = VibrationLevel.SAFE

            self.last_read_ok = True
            return IMUData(
                pitch=pitch,
                roll=roll,
                yaw=self.yaw,

                accel_x=acc_x * 9.81,
                accel_y=acc_y * 9.81,
                accel_z=acc_z * 9.81,

                gyro_z=gyro_z,

                vib_rms=rms,
                vib_level=lvl,

                tilt_fault=tilt_fault,
                slope_warning=abs(pitch) > 0.25,
            )

        except Exception as e:

            self.last_read_ok = False
            print(
                f"[Hardware] IMU read error: {e}"
            )

            return IMUData(
                yaw=self.yaw
            )

    def snap_yaw(self, new_yaw: float) -> None:
        self.yaw = new_yaw

    def start(self) -> None:
        pass

    def cleanup(self) -> None:

        if self.bus:

            try:
                self.bus.close()
            except Exception:
                pass

            self.bus = None


if __name__ == "__main__":

    print("=== MPU6050 standalone test ===")

    imu = IMUHW()

    if imu.bus is None:
        print("ERROR: IMU could not be opened.")
        sys.exit(1)

    print()
    print("Reading IMU for 10 seconds...")
    print("Keep the sensor stationary.")
    print()

    try:

        for i in range(50):

            data = imu.get_latest()

            print(
                f"[{i:02d}] "
                f"Ax={data.accel_x:7.3f} "
                f"Ay={data.accel_y:7.3f} "
                f"Az={data.accel_z:7.3f} | "
                f"Gz={data.gyro_z:7.2f}°/s | "
                f"Yaw={math.degrees(data.yaw):7.2f}° | "
                f"Pitch={math.degrees(data.pitch):6.2f}° | "
                f"Roll={math.degrees(data.roll):6.2f}°"
            )

            time.sleep(0.2)

    except KeyboardInterrupt:

        print("\nStopped.")

    finally:

        imu.cleanup()