# MediVan technical research notes

Research references relevant to this project's exact hardware and runtime design. These references explain upstream APIs and calibration concepts; they do not certify MediVan's physical implementation.

## Camera stack and frame handling

- Raspberry Pi's Picamera2 library is built on libcamera and is the supported replacement for the legacy PiCamera API for current CSI camera workflows: <https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf>.
- Raspberry Pi camera setup and ribbon-cable guidance: <https://www.raspberrypi.com/documentation/accessories/camera.html>.
- Raspberry Pi's camera software documents `rpicam-*` commands and libcamera use: <https://www.raspberrypi.com/documentation/computers/camera_software.html>.

MediVan's recorded hardware is an OV5647 CSI camera. The software must be configured against that sensor and the actual capture dimensions; do not copy intrinsics from a different camera module or resolution.

## Calibration and geometric range

- OpenCV calibration estimates camera intrinsic parameters and distortion from known target points: <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>.
- OpenCV's chessboard calibration tutorial describes finding corners and solving for camera parameters: <https://docs.opencv.org/4.12.0/dc/d43/tutorial_camera_calibration_square_chess.html>.

Intrinsic calibration is necessary but not sufficient for floor-plane inverse perspective mapping. Camera height, pitch, forward offset, lens parameters, floor appearance, lighting, robot frame alignment, and segmentation behavior all affect range estimates. MediVan's recent Pi log shows suspicious hit/clear ray patterns; physical range accuracy is therefore still a no-go until compared with known targets at measured distances on the actual floor.

## Motor interface and GPIO assumptions

- GPIO Zero's `Motor` interface drives a bidirectional H-bridge and supports a separate enable pin with PWM when configured accordingly: <https://gpiozero.readthedocs.io/en/latest/api_output.html#gpiozero.Motor>.

The motor GPIOs in `config.py` are preserved from the user's existing physical wiring: ENA=BCM GPIO12, left IN1/IN2=GPIO5/6, ENB=GPIO18, right IN3/IN4=GPIO13/19. Do not substitute generic diagrams for physical verification. Raised-wheel tests must independently confirm left/right directions, enable duty-cycle behavior and emergency stop. The software's open-loop velocity estimate is not encoder feedback.

## Dashboard and control design

The hardware dashboard uses Python's standard-library HTTP server for telemetry and a separate control server. Telemetry is read-only on port 5000; commands go to port 5001. The control state uses a 400 ms command lease for held manual-drive requests, explicit arm/disarm states, and a software E-stop latch. The independent hardware power cutoff remains necessary because software cannot guarantee stopping if the main loop or OS is blocked.

The separate Flask simulator/demo backend is not the hardware command process. Flask's official deployment notes warn that its development server is not intended for production deployment: <https://flask.palletsprojects.com/en/stable/deploying/>.

## Evidence boundaries

Software tests can validate interfaces, mode interlocks, command transformation, planner assumptions and synthetic vision behavior. They cannot prove real camera range accuracy, traction, PWM output, collision avoidance or successful physical delivery. MediVan has no verified physical wheel encoder feedback, no integrated physical medicine-release actuator, and no fully integrated precision docking/live-battery safety loop in this release candidate.
