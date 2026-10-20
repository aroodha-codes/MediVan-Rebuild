#!/usr/bin/env python3
"""test_motors_configured.py -- Direct test for:
  Left : [5, 6, 12]   (IN1=5, IN2=6, ENA=12)
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

# Exact configuration specified:
LEFT_IN1, LEFT_IN2, LEFT_ENA = 5, 6, 12
RIGHT_IN3, RIGHT_IN4, RIGHT_ENB = 13, 19, 18

PWM_SPEED = 0.55  # 55% PWM for good starting torque
RUN_TIME = 5.0     # 5 seconds per test

print("=" * 60)
print("     TESTING CONFIG: left=[5, 6, 12], right=[13, 19, 18]")
print("=" * 60)
print(f"LEFT MOTOR  -> IN1=GPIO{LEFT_IN1}, IN2=GPIO{LEFT_IN2}, ENA=GPIO{LEFT_ENA}")
print(f"RIGHT MOTOR -> IN3=GPIO{RIGHT_IN3}, IN4=GPIO{RIGHT_IN4}, ENB=GPIO{RIGHT_ENB}")
print(f"Testing at {PWM_SPEED*100:.0f}% duty cycle for {RUN_TIME:.0f} seconds each.")
print("=" * 60)
input("Ensure robot wheels are free or elevated. Press Enter to start... ")

# Initialize devices
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
    print(f"  {label} -> DONE.   ")

try:
    # ── Test 1: Left Wheel Forward ──────────────────────
    print("\n[TEST 1/5] LEFT Wheel FORWARD for 5 seconds...")
    print(f"  Driving: GPIO{LEFT_IN1}=HIGH, GPIO{LEFT_IN2}=LOW, ENA=GPIO{LEFT_ENA}={PWM_SPEED*100:.0f}%")
    left_in1.on()
    left_in2.off()
    left_ena.value = PWM_SPEED
    countdown("LEFT FORWARD", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 2: Left Wheel Reverse ──────────────────────
    print("\n[TEST 2/5] LEFT Wheel REVERSE for 5 seconds...")
    print(f"  Driving: GPIO{LEFT_IN1}=LOW, GPIO{LEFT_IN2}=HIGH, ENA=GPIO{LEFT_ENA}={PWM_SPEED*100:.0f}%")
    left_in1.off()
    left_in2.on()
    left_ena.value = PWM_SPEED
    countdown("LEFT REVERSE", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 3: Right Wheel Forward ─────────────────────
    print("\n[TEST 3/5] RIGHT Wheel FORWARD for 5 seconds...")
    print(f"  Driving: GPIO{RIGHT_IN3}=HIGH, GPIO{RIGHT_IN4}=LOW, ENB=GPIO{RIGHT_ENB}={PWM_SPEED*100:.0f}%")
    right_in3.on()
    right_in4.off()
    right_enb.value = PWM_SPEED
    countdown("RIGHT FORWARD", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 4: Right Wheel Reverse ─────────────────────
    print("\n[TEST 4/5] RIGHT Wheel REVERSE for 5 seconds...")
    print(f"  Driving: GPIO{RIGHT_IN3}=LOW, GPIO{RIGHT_IN4}=HIGH, ENB=GPIO{RIGHT_ENB}={PWM_SPEED*100:.0f}%")
    right_in3.off()
    right_in4.on()
    right_enb.value = PWM_SPEED
    countdown("RIGHT REVERSE", RUN_TIME)
    stop_all()
    time.sleep(1.5)

    # ── Test 5: Both Wheels Forward ─────────────────────
    print("\n[TEST 5/5] BOTH Wheels FORWARD for 5 seconds...")
    left_in1.on()
    left_in2.off()
    left_ena.value = PWM_SPEED

    right_in3.on()
    right_in4.off()
    right_enb.value = PWM_SPEED
    countdown("BOTH FORWARD", RUN_TIME)
    stop_all()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE! Motors safely stopped.")
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
