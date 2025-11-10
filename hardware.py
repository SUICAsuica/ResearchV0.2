"""
Hardware abstraction for the OSOYOO Raspberry Pi Robot Car.

Lesson 6 では Flask 経由で操作するため本モジュールは必須ではないが、
ラズパイ上で直接 PCA9685 / RPi.GPIO を制御したい場合の補助として残している。
非 Raspberry Pi 環境で実行された場合は自動的に dry-run モードへフォールバックする。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotorPins:
    """BCM pin assignments for the dual H-bridge and PCA9685 channels."""

    in1: int = 23  # Left motor direction pin A
    in2: int = 24  # Left motor direction pin B
    in3: int = 27  # Right motor direction pin A
    in4: int = 22  # Right motor direction pin B
    ena_channel: int = 0  # PCA9685 channel for left motor speed
    enb_channel: int = 1  # PCA9685 channel for right motor speed
    servo_channel: Optional[int] = 15  # Optional pan servo channel


@dataclass(frozen=True)
class ServoConfig:
    """Simple min/max pulse settings for the pan servo."""

    min_pulse_us: int = 500
    max_pulse_us: int = 2500
    frequency_hz: int = 50


class MotorController:
    """
    Control wrapper for the two drive motors (via L298N + PCA9685) and the
    optional pan servo.

    The controller keeps track of the most recent command so that higher level
    code can decide whether a watchdog timeout needs to idle the motors.
    """

    _MAX_DUTY = 0xFFFF

    def __init__(
        self,
        pins: MotorPins | None = None,
        servo: ServoConfig | None = None,
        *,
        pwm_frequency_hz: int = 60,
        dry_run: bool | None = None,
    ) -> None:
        self.pins = pins or MotorPins()
        self.servo_cfg = servo or ServoConfig()
        self._pwm_frequency_hz = pwm_frequency_hz
        self._last_command: Tuple[float, float] = (0.0, 0.0)
        self._last_servo_angle: Optional[float] = None

        if dry_run is None:
            dry_run = False
        self._forced_dry_run = dry_run
        self._dry_run = dry_run

        self._gpio = None
        self._pca = None
        self._i2c = None

        self._setup()

    # --------------------------------------------------------------------- #
    # Initialisation helpers
    # --------------------------------------------------------------------- #
    def _setup(self) -> None:
        if self._forced_dry_run:
            LOG.info("MotorController running in dry-run mode (forced).")
            self._dry_run = True
            return

        try:
            import RPi.GPIO as GPIO  # type: ignore[import-not-found]
            import busio  # type: ignore[import-not-found]
            from board import SCL, SDA  # type: ignore[import-not-found]
            from adafruit_pca9685 import PCA9685  # type: ignore[import-not-found]
        except (ImportError, NotImplementedError, RuntimeError) as exc:
            LOG.warning(
                "GPIO/PCA9685 libraries unavailable (%s); enabling dry-run fallback.",
                exc,
            )
            self._dry_run = True
            return

        self._gpio = GPIO
        self._gpio.setmode(GPIO.BCM)
        self._gpio.setwarnings(False)
        for pin in (self.pins.in1, self.pins.in2, self.pins.in3, self.pins.in4):
            self._gpio.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        self._i2c = busio.I2C(SCL, SDA)
        self._pca = PCA9685(self._i2c)
        self._pca.frequency = self._pwm_frequency_hz

        LOG.info(
            "MotorController initialised (PCA9685 freq=%d Hz, servo channel=%s).",
            self._pwm_frequency_hz,
            self.pins.servo_channel,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def last_command(self) -> Tuple[float, float]:
        return self._last_command

    @property
    def last_servo_angle(self) -> Optional[float]:
        return self._last_servo_angle

    def set_speed(self, left: float, right: float) -> None:
        """
        Apply a new speed command.

        :param left:  Normalised speed for the left wheels (-1.0 .. 1.0).
        :param right: Normalised speed for the right wheels (-1.0 .. 1.0).
        """
        left = _clamp(left, -1.0, 1.0)
        right = _clamp(right, -1.0, 1.0)
        self._last_command = (left, right)

        if self._dry_run:
            LOG.debug("DRY-RUN motor command: left=%.2f right=%.2f", left, right)
            return

        assert self._gpio is not None
        assert self._pca is not None

        self._apply_motor(self.pins.in1, self.pins.in2, self.pins.ena_channel, left)
        self._apply_motor(self.pins.in3, self.pins.in4, self.pins.enb_channel, right)

    def stop(self) -> None:
        """Immediately halt both motors."""
        self.set_speed(0.0, 0.0)

    def set_servo_angle(self, angle_deg: float) -> None:
        """
        Point the camera pan servo to a new angle in degrees.
        The expected range is roughly -90 .. 90, but wider ranges are clipped.
        """
        self._last_servo_angle = _clamp(angle_deg, -90.0, 90.0)
        if self._dry_run or self.pins.servo_channel is None:
            LOG.debug("DRY-RUN servo angle %.1f°", self._last_servo_angle)
            return

        assert self._pca is not None
        duty = _servo_angle_to_duty(
            self._last_servo_angle,
            self.servo_cfg.min_pulse_us,
            self.servo_cfg.max_pulse_us,
            self.servo_cfg.frequency_hz,
        )
        self._pca.channels[self.pins.servo_channel].duty_cycle = duty

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if not self._dry_run and self._gpio is not None:
            self._gpio.cleanup()
        if self._pca is not None:
            try:
                self._pca.deinit()  # type: ignore[attr-defined]
            except AttributeError:
                pass

    # ------------------------------------------------------------------ #
    # Context manager helpers
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "MotorController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_motor(self, pin_a: int, pin_b: int, channel: int, value: float) -> None:
        assert self._gpio is not None
        assert self._pca is not None

        if value > 0.0:
            self._gpio.output(pin_a, self._gpio.HIGH)
            self._gpio.output(pin_b, self._gpio.LOW)
        elif value < 0.0:
            self._gpio.output(pin_a, self._gpio.LOW)
            self._gpio.output(pin_b, self._gpio.HIGH)
        else:
            self._gpio.output(pin_a, self._gpio.LOW)
            self._gpio.output(pin_b, self._gpio.LOW)

        duty = int(abs(value) * self._MAX_DUTY)
        self._pca.channels[channel].duty_cycle = duty


# ---------------------------------------------------------------------- #
# Utility helpers
# ---------------------------------------------------------------------- #
def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _servo_angle_to_duty(
    angle_deg: float,
    min_pulse_us: int,
    max_pulse_us: int,
    frequency_hz: int,
) -> int:
    """Convert an angle into a 16-bit duty cycle for the PCA9685."""
    # Map -90..90 degrees into [0, 1].
    normalised = (angle_deg + 90.0) / 180.0
    normalised = _clamp(normalised, 0.0, 1.0)
    pulse_range = max_pulse_us - min_pulse_us
    pulse_us = min_pulse_us + normalised * pulse_range
    period_us = 1_000_000.0 / frequency_hz
    duty_cycle = int(_clamp(pulse_us / period_us, 0.0, 1.0) * MotorController._MAX_DUTY)
    return duty_cycle
