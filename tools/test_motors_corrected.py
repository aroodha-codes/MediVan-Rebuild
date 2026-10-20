#!/usr/bin/env python3
"""test_motors_corrected.py -- Validates the corrected motor pins:
  Left : [6, 5, 12]   (IN1=6, IN2=5, ENA=12)  <- Polarity corrected!
  Right: [13, 19, 18] (IN3=13, IN4=19, ENB=18)
"""
from __future__ import annotations

import sys
import time

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
except ImportError:
    print("gpiozero library not found. Install or run on Raspberry Pi.")
    sys.exit(1)

# Corrected configuration:
LEFT_IN1, LEFT_IN2, LEFT_ENA = 6, 5, 12
RIGHT_IN3, RIGHT_IN4, RIGHT_ENB = 13, 19, 18

PWM_SPEED = 0.55  # 55% PWM for good starting torque
RUN_TIME = 5.0     # 5 seconds per test

print("=" * 60)
print("     TESTING CORRECTED CONFIG:")
print("       Left : [6, 5, 12]   (IN1=6,  IN2=5,  ENA=12)")
print("       Right: [13, 19, 18] (IN3=13, IN4=19, ENB=18)")
print("=" * 60)
print(f"Testing at {PWM_SPEED*100:.0f}% duty cycle for {RUN_TIME:.0f} seconds each.")
print("=" * 60)
input("Place robot on a clear floor or stand. Press Enter to start... ")

left_ena = PWMOutputDevice(LEFT_ENA, frequency=1000, initial_value=0)
left_in1 = DigitalOutputDevice(LEFT_IN1, initial_value=False)
left_in2 = DigitalOutputDevice(LEFT_IN2, initial_value=False)

right_enb = PWMOutputDevice(RIGHT_ENB, frequency=1000, initial_value=0)
right_in3 = DigitalOutputDevice(RIGHT_IN3, initial_value=False)
right_in4 = DigitalOutputDevice(RIGHT_IN4, initial_value=False)

def stop_all():
    left_ena.value = 0
    left_in1.off()
    left_in2.off()
    right_enb.value = 0
    right_in3.off()
    right_in4.off()

def countdown(label: str, duration: float):
    for sec in range(int(duration), 0, -1):
        print(f"  {label} -> {sec}s remaining...", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  {label} -> STOPPED.   ")

try:
    # ── Test 1: Left Wheel Forward ──────────────────────
    print("\n[TEST 1/6] LEFT Wheel FORWARD for 5 seconds...")
    print("  Expected: Left wheel drives forward (robot pivots to the right).")
    left_in1.on()
    left_in2.off()
    left_ena.value = PWM_SPEED
    countdown("LEFT FORWARD", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 2: Left Wheel Reverse ──────────────────────
    print("\n[TEST 2/6] LEFT Wheel REVERSE for 5 seconds...")
    print("  Expected: Left wheel drives backward (robot pivots to the left).")
    left_in1.off()
    left_in2.on()
    left_ena.value = PWM_SPEED
    countdown("LEFT REVERSE", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 3: Right Wheel Forward ─────────────────────
    print("\n[TEST 3/6] RIGHT Wheel FORWARD for 5 seconds...")
    print("  Expected: Right wheel drives forward (robot pivots to the left).")
    right_in3.on()
    right_in4.off()
    right_enb.value = PWM_SPEED
    countdown("RIGHT FORWARD", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 4: Right Wheel Reverse ─────────────────────
    print("\n[TEST 4/6] RIGHT Wheel REVERSE for 5 seconds...")
    print("  Expected: Right wheel drives backward (robot pivots to the right).")
    right_in3.off()
    right_in4.on()
    right_enb.value = PWM_SPEED
    countdown("RIGHT REVERSE", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 5: Both Wheels Forward ─────────────────────
    print("\n[TEST 5/6] BOTH Wheels FORWARD for 5 seconds...")
    print("  Expected: Robot drives STRAIGHT FORWARD!")
    left_in1.on()
    left_in2.off()
    left_ena.value = PWM_SPEED

    right_in3.on()
    right_in4.off()
    right_enb.value = PWM_SPEED
    countdown("STRAIGHT FORWARD", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 6: Both Wheels Reverse ─────────────────────
    print("\n[TEST 6/6] BOTH Wheels REVERSE for 5 seconds...")
    print("  Expected: Robot drives STRAIGHT BACKWARD!")
    left_in1.off()
    left_in2.on()
    left_ena.value = PWM_SPEED

    right_in3.off()
    right_in4.on()
    right_enb.value = PWM_SPEED
    countdown("STRAIGHT BACKWARD", RUN_TIME)
    stop_all()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE! If Test 5 drove straight forward,")
    print("your motor wiring and GPIO configuration are 100% verified!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n[STOP] Test stopped early by user.")
finally:
    stop_all()
    left_ena.close()
    left_in1.close()
    left_in2.close()
    right_enb.close()
    right_in3.close()
    right_in4.close()
    print("GPIO pins released.")
