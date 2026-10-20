#!/usr/bin/env python3
"""verify_wheels.py -- Interactive 5-second wheel verification test.

Tests both wheels individually and together through the production
MotorDriverHW interface to verify wiring, direction polarity, and torque.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    PIN_MOTOR_ENA, PIN_MOTOR_IN1, PIN_MOTOR_IN2,
    PIN_MOTOR_ENB, PIN_MOTOR_IN3, PIN_MOTOR_IN4,
    MotorCommand, MotorDirection,
)
from hardware.motor_driver_hw import MotorDriverHW


def run_wheel_test(pwm: int = 128, duration: float = 5.0) -> None:
    print("=" * 60)
    print("        MEDIVAN 5-SECOND WHEEL HARDWARE VERIFICATION")
    print("=" * 60)
    print("Pin mapping:")
    print(f"  LEFT  motor: ENA=GPIO{PIN_MOTOR_ENA} (PWM), IN1=GPIO{PIN_MOTOR_IN1}, IN2=GPIO{PIN_MOTOR_IN2}")
    print(f"  RIGHT motor: ENB=GPIO{PIN_MOTOR_ENB} (PWM), IN3=GPIO{PIN_MOTOR_IN3}, IN4=GPIO{PIN_MOTOR_IN4}")
    print(f"Testing speed : PWM {pwm} / 255 ({(pwm/255.0)*100:.1f}% duty cycle)")
    print(f"Test duration : {duration:.1f} seconds per phase")
    print("=" * 60)
    print("\nSAFETY CHECK:")
    print("Place the MediVan chassis on a stand or block so the wheels are")
    print("off the ground and free to spin, or clear a 1-meter floor space.")
    print("You can press Ctrl+C at any time to immediately stop all motors.\n")

    input("Press Enter to begin the verification sequence... ")

    driver = MotorDriverHW()
    stop_cmd = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

    phases = [
        (
            "PHASE 1: LEFT WHEEL ONLY -> FORWARD",
            MotorCommand(pwm, 0, MotorDirection.FWD, MotorDirection.BRAKE),
            "Left wheel should spin FORWARD (towards front bumper/camera).",
        ),
        (
            "PHASE 2: LEFT WHEEL ONLY -> REVERSE",
            MotorCommand(pwm, 0, MotorDirection.REV, MotorDirection.BRAKE),
            "Left wheel should spin BACKWARD (away from front bumper/camera).",
        ),
        (
            "PHASE 3: RIGHT WHEEL ONLY -> FORWARD",
            MotorCommand(0, pwm, MotorDirection.BRAKE, MotorDirection.FWD),
            "Right wheel should spin FORWARD (towards front bumper/camera).",
        ),
        (
            "PHASE 4: RIGHT WHEEL ONLY -> REVERSE",
            MotorCommand(0, pwm, MotorDirection.BRAKE, MotorDirection.REV),
            "Right wheel should spin BACKWARD (away from front bumper/camera).",
        ),
        (
            "PHASE 5: BOTH WHEELS -> STRAIGHT FORWARD",
            MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.FWD),
            "Both wheels should spin FORWARD in unison at identical speed.",
        ),
        (
            "PHASE 6: BOTH WHEELS -> STRAIGHT REVERSE",
            MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV),
            "Both wheels should spin BACKWARD in unison at identical speed.",
        ),
        (
            "PHASE 7: PIVOT TURN LEFT (Counter-Clockwise)",
            MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.FWD),
            "Left wheel BACKWARD, Right wheel FORWARD (turns robot left).",
        ),
        (
            "PHASE 8: PIVOT TURN RIGHT (Clockwise)",
            MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.REV),
            "Left wheel FORWARD, Right wheel BACKWARD (turns robot right).",
        ),
    ]

    try:
        for idx, (title, cmd, expectation) in enumerate(phases, start=1):
            print("\n" + "-" * 60)
            print(f"[{idx}/{len(phases)}] {title}")
            print(f"Expected: {expectation}")
            print(f"Command : Left={cmd.dir_a.name} (PWM {cmd.pwm_a}) | Right={cmd.dir_b.name} (PWM {cmd.pwm_b})")
            print("Running for 5 seconds...", end="", flush=True)

            driver.set_pwm(cmd)
            t_end = time.time() + duration
            while time.time() < t_end:
                driver.update(0.05)
                time.sleep(0.05)

            # Brake for 1.5 seconds between phases
            driver.set_pwm(stop_cmd)
            driver.update(0.05)
            print(" STOPPED.")
            time.sleep(1.5)

        print("\n" + "=" * 60)
        print("           VERIFICATION SEQUENCE COMPLETE")
        print("=" * 60)
        print("Diagnostic summary:")
        print("  - If a wheel spun in the opposite direction from the description:")
        print("    Swap the two motor wires on the corresponding L298N terminal block")
        print("    (OUT1 & OUT2 for Left motor; OUT3 & OUT4 for Right motor).")
        print("  - If both wheels followed the expected directions correctly:")
        print("    Your motor wiring and driver polarity are 100% verified!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Interrupted by user; halting motors immediately.")
    finally:
        driver.emergency_stop()
        driver.cleanup()
        print("GPIO outputs safely cleaned up and released.")


if __name__ == "__main__":
    speed = 128
    if len(sys.argv) > 1:
        try:
            speed = int(sys.argv[1])
        except ValueError:
            pass
    run_wheel_test(pwm=speed, duration=5.0)
