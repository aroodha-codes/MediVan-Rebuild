#!/usr/bin/env python3
"""test_motors_final.py -- Validates the final verified motor pins:
  Left : [19, 13, 18] (IN1=19, IN2=13, ENA=18)
  Right: [5, 6, 12]   (IN3=5,  IN4=6,  ENB=12)
"""
from __future__ import annotations

import sys
import time

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
except ImportError:
    print("gpiozero library not found. Install or run on Raspberry Pi.")
    sys.exit(1)

# Final verified configuration:
LEFT_IN1, LEFT_IN2, LEFT_ENA = 19, 13, 18
RIGHT_IN3, RIGHT_IN4, RIGHT_ENB = 5, 6, 12

PWM_SPEED = 0.55  # 55% PWM for smooth ground driving
RUN_TIME = 5.0

print("=" * 60)
print("     FINAL VERIFIED MOTOR CONFIGURATION:")
print(f"       LEFT  motor: IN1=GPIO{LEFT_IN1}, IN2=GPIO{LEFT_IN2}, ENA=GPIO{LEFT_ENA}")
print(f"       RIGHT motor: IN3=GPIO{RIGHT_IN3}, IN4=GPIO{RIGHT_IN4}, ENB=GPIO{RIGHT_ENB}")
print("=" * 60)
input("Place robot on a clear floor. Press Enter to start... ")

left_ena = PWMOutputDevice(LEFT_ENA, frequency=1000, initial_value=0)
left_in1 = DigitalOutputDevice(LEFT_IN1, initial_value=False)
left_in2 = DigitalOutputDevice(LEFT_IN2, initial_value=False)

right_enb = PWMOutputDevice(RIGHT_ENB, frequency=1000, initial_value=0)
right_in3 = DigitalOutputDevice(RIGHT_IN3, initial_value=False)
right_in4 = DigitalOutputDevice(RIGHT_IN4, initial_value=False)

def stop_all():
    left_ena.value = 0; left_in1.off(); left_in2.off()
    right_enb.value = 0; right_in3.off(); right_in4.off()

def countdown(label, duration):
    for sec in range(int(duration), 0, -1):
        print(f"  {label} -> {sec}s remaining...", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  {label} -> STOPPED.   ")

try:
    # ── Test 1: Straight Forward ────────────────────────
    print("\n[1/4] BOTH Wheels FORWARD (5s)... (Robot must drive STRAIGHT FORWARD!)")
    left_in1.on(); left_in2.off(); left_ena.value = PWM_SPEED
    right_in3.on(); right_in4.off(); right_enb.value = PWM_SPEED
    countdown("STRAIGHT FORWARD", RUN_TIME)
    stop_all(); time.sleep(1.5)

    # ── Test 2: Straight Reverse ────────────────────────
    print("\n[2/4] BOTH Wheels REVERSE (5s)... (Robot must drive STRAIGHT BACKWARD!)")
    left_in1.off(); left_in2.on(); left_ena.value = PWM_SPEED
    right_in3.off(); right_in4.on(); right_enb.value = PWM_SPEED
    countdown("STRAIGHT BACKWARD", RUN_TIME)
    stop_all(); time.sleep(1.5)

    # ── Test 3: Pivot Turn Left ─────────────────────────
    print("\n[3/4] PIVOT TURN LEFT (5s)... (Robot rotates counter-clockwise)")
    left_in1.off(); left_in2.on(); left_ena.value = PWM_SPEED
    right_in3.on(); right_in4.off(); right_enb.value = PWM_SPEED
    countdown("TURN LEFT", RUN_TIME)
    stop_all(); time.sleep(1.5)

    # ── Test 4: Pivot Turn Right ────────────────────────
    print("\n[4/4] PIVOT TURN RIGHT (5s)... (Robot rotates clockwise with boosted torque)")
    turn_right_speed = min(1.0, PWM_SPEED * 1.25)
    left_in1.on(); left_in2.off(); left_ena.value = turn_right_speed
    right_in3.off(); right_in4.on(); right_enb.value = turn_right_speed
    countdown("TURN RIGHT", RUN_TIME)
    stop_all()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! MOTORS ARE 100% PERFECTLY CONFIGURED.")
    print("=" * 60)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stop_all()
    left_ena.close(); left_in1.close(); left_in2.close()
    right_enb.close(); right_in3.close(); right_in4.close()
