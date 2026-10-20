#!/usr/bin/env python3
"""test_single_wheel.py -- Isolated single-wheel GPIO diagnostic tool.

Tests one wheel at a time for 5 seconds to isolate pin mismatches,
wiring polarity, and track down swapped Left/Right channels.
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
)

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False
    print("[WARNING] gpiozero not found. Running in simulation stub mode.")

DEFAULT_DUTY = 0.50  # 50% PWM duty cycle for strong starting torque
DURATION_SEC = 5.0


def run_single_test(
    name: str,
    enable_pin: int,
    in_a_pin: int,
    in_b_pin: int,
    direction: str,  # "fwd" or "rev"
    duty: float = DEFAULT_DUTY,
    duration: float = DURATION_SEC,
) -> None:
    print("\n" + "=" * 65)
    print(f" TESTING: {name} ({direction.upper()}) for {duration:.1f} seconds")
    print("=" * 65)
    print(f"  Active Pins: ENABLE=GPIO{enable_pin} (PWM duty {duty*100:.0f}%)")
    if direction == "fwd":
        print(f"               IN_A=GPIO{in_a_pin} (HIGH / 3.3V)")
        print(f"               IN_B=GPIO{in_b_pin} (LOW  / 0.0V)")
    else:
        print(f"               IN_A=GPIO{in_a_pin} (LOW  / 0.0V)")
        print(f"               IN_B=GPIO{in_b_pin} (HIGH / 3.3V)")
    print("-" * 65)

    if not _GPIO_AVAILABLE:
        print("[MOCK] Simulating 5 seconds of motor motion...")
        for remaining in range(int(duration), 0, -1):
            print(f"  {remaining}s remaining...", end="\r", flush=True)
            time.sleep(1.0)
        print("  Finished (MOCK).")
        return

    # Setup devices
    ena_dev = PWMOutputDevice(enable_pin, frequency=1000, initial_value=0)
    in_a_dev = DigitalOutputDevice(in_a_pin, initial_value=False)
    in_b_dev = DigitalOutputDevice(in_b_pin, initial_value=False)

    try:
        if direction == "fwd":
            in_a_dev.on()
            in_b_dev.off()
        else:
            in_a_dev.off()
            in_b_dev.on()

        ena_dev.value = duty

        for remaining in range(int(duration), 0, -1):
            print(f"  Motor RUNNING -> {remaining}s remaining... (Ctrl+C to stop early)", end="\r", flush=True)
            time.sleep(1.0)
    finally:
        ena_dev.value = 0
        in_a_dev.off()
        in_b_dev.off()
        ena_dev.close()
        in_a_dev.close()
        in_b_dev.close()
        print("\n  Motor STOPPED and pins safely released.")


def test_left_forward(duty=DEFAULT_DUTY, duration=DURATION_SEC):
    run_single_test("LEFT WHEEL (ENA=12, IN1=5, IN2=6)", PIN_MOTOR_ENA, PIN_MOTOR_IN1, PIN_MOTOR_IN2, "fwd", duty, duration)

def test_left_reverse(duty=DEFAULT_DUTY, duration=DURATION_SEC):
    run_single_test("LEFT WHEEL (ENA=12, IN1=5, IN2=6)", PIN_MOTOR_ENA, PIN_MOTOR_IN1, PIN_MOTOR_IN2, "rev", duty, duration)

def test_right_forward(duty=DEFAULT_DUTY, duration=DURATION_SEC):
    run_single_test("RIGHT WHEEL (ENB=18, IN3=13, IN4=19)", PIN_MOTOR_ENB, PIN_MOTOR_IN3, PIN_MOTOR_IN4, "fwd", duty, duration)

def test_right_reverse(duty=DEFAULT_DUTY, duration=DURATION_SEC):
    run_single_test("RIGHT WHEEL (ENB=18, IN3=13, IN4=19)", PIN_MOTOR_ENB, PIN_MOTOR_IN3, PIN_MOTOR_IN4, "rev", duty, duration)


def menu():
    while True:
        print("\n" + "=" * 65)
        print("         MEDIVAN SINGLE-WHEEL PIN DIAGNOSTIC MENU")
        print("=" * 65)
        print("Configured mapping in config.py:")
        print(f"  LEFT Channel : ENA=GPIO{PIN_MOTOR_ENA}, IN1=GPIO{PIN_MOTOR_IN1}, IN2=GPIO{PIN_MOTOR_IN2}")
        print(f"  RIGHT Channel: ENB=GPIO{PIN_MOTOR_ENB}, IN3=GPIO{PIN_MOTOR_IN3}, IN4=GPIO{PIN_MOTOR_IN4}")
        print("-" * 65)
        print("Select a single motor test to run for 5 seconds:")
        print("  1) Test LEFT wheel  -> FORWARD")
        print("  2) Test LEFT wheel  -> REVERSE")
        print("  3) Test RIGHT wheel -> FORWARD")
        print("  4) Test RIGHT wheel -> REVERSE")
        print("  5) Run Left then Right sequential test")
        print("  q) Quit")
        print("=" * 65)
        choice = input("Enter option [1-5, q]: ").strip().lower()

        if choice == "1":
            test_left_forward()
            diagnose("LEFT", "FORWARD")
        elif choice == "2":
            test_left_reverse()
            diagnose("LEFT", "REVERSE")
        elif choice == "3":
            test_right_forward()
            diagnose("RIGHT", "FORWARD")
        elif choice == "4":
            test_right_reverse()
            diagnose("RIGHT", "REVERSE")
        elif choice == "5":
            print("\nStarting sequential single wheel test...")
            test_left_forward()
            time.sleep(1.0)
            test_left_reverse()
            time.sleep(1.0)
            test_right_forward()
            time.sleep(1.0)
            test_right_reverse()
        elif choice in ("q", "quit", "exit"):
            print("Exiting diagnostic.")
            break
        else:
            print("Invalid option. Enter 1, 2, 3, 4, 5, or q.")


def diagnose(expected_wheel: str, expected_dir: str):
    print("\nOBSERVATION CHECK:")
    ans = input(f"Did the physical {expected_wheel} wheel spin {expected_dir}? (y/n): ").strip().lower()
    if ans == "y":
        print(f"[OK] {expected_wheel} channel and direction are correctly wired.")
    else:
        what_happened = input("What happened instead? (1=Opposite wheel spun, 2=Spun wrong direction, 3=Nothing): ").strip()
        if what_happened == "1":
            print("\n[DIAGNOSIS: LEFT/RIGHT CHANNEL SWAP]")
            print("Your Left and Right channels are physically swapped!")
            print("The code commanded the LEFT channel (ENA=12, IN1=5, IN2=6), but the other wheel spun.")
            print("Fix: Swap the motor wires between OUT1/OUT2 and OUT3/OUT4 on the L298N driver,")
            print("OR swap the pin definitions in config.py:")
            print("  PIN_MOTOR_ENA, PIN_MOTOR_IN1, PIN_MOTOR_IN2 <-> PIN_MOTOR_ENB, PIN_MOTOR_IN3, PIN_MOTOR_IN4")
        elif what_happened == "2":
            print(f"\n[DIAGNOSIS: {expected_wheel} POLARITY INVERTED]")
            print(f"The correct wheel turned, but in the reverse direction.")
            print(f"Fix: Invert the two wires for the {expected_wheel} motor on the L298N screw terminal block.")
        elif what_happened == "3":
            print(f"\n[DIAGNOSIS: NO MOTION]")
            print("Check that:")
            print("  1. The L298N 12V/GND power terminals are connected to battery and Pi GND.")
            print("  2. The ENA/ENB jumper has been removed so GPIO PWM can control speed.")
            print("  3. The battery is fully charged.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("left", "left-fwd", "1"):
            test_left_forward()
        elif cmd in ("left-rev", "2"):
            test_left_reverse()
        elif cmd in ("right", "right-fwd", "3"):
            test_right_forward()
        elif cmd in ("right-rev", "4"):
            test_right_reverse()
        else:
            print(f"Unknown argument: {cmd}. Usage: python3 test_single_wheel.py [left|right|left-rev|right-rev]")
    else:
        menu()
