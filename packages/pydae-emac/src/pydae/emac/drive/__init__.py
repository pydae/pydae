"""The switched drive: a multirate CasADi solver (DSP every T_s, PWM carrier,
switching edges, dead time) for what a mean-value DAE cannot show."""

from pydae.emac.drive.controllers import PredictiveController
from pydae.emac.drive.drive_model import (
    PIController,
    build_control_period,
    build_machine,
    build_window,
    dq_to_alpha_beta,
    leg_voltages_to_alpha_beta,
    limit_voltage,
    pwm_pieces,
    simulate,
    time_average,
)
from pydae.emac.drive.modulation import abc_to_dq, compensate_dead_time, sine_duties, svpwm_duties
from pydae.emac.drive.pwm_plant import PwmPlant

__all__ = ["PIController", "PredictiveController", "PwmPlant", "abc_to_dq", "compensate_dead_time", "build_control_period", "build_machine",
           "build_window", "dq_to_alpha_beta", "leg_voltages_to_alpha_beta", "limit_voltage",
           "pwm_pieces", "simulate", "sine_duties", "svpwm_duties", "time_average"]
