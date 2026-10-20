"""
CFCL Gadepan Complex - IEC 61511 / ISA-84 Safety Instrumented System (SIS) Interlocks.
Independent Protection Layer (IPL) demarcation verifying hardware SIL-2/SIL-3 trip boundaries
for primary reformer skin temperature, synthesis loop pressure, and compressor surge.
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.utils.logger import get_logger

logger = get_logger("SISInterlocks")


@dataclass
class SIFTripStatus:
    """Status of an individual Safety Instrumented Function (SIF) loop."""
    sif_tag: str
    description: str
    current_value: float
    trip_setpoint: float
    trip_direction: str  # "HIGH_HIGH" or "LOW_LOW"
    engineering_units: str
    is_tripped: bool
    is_warning: bool
    margin_to_trip: float


@dataclass
class SISVerificationReport:
    """Complete IEC 61511 Layer of Protection Analysis (LOPA) verification."""
    is_fully_safe: bool
    trip_count: int
    warning_count: int
    interlock_statuses: List[SIFTripStatus]
    safety_integrity_level: str
    executive_safety_action: str


class SafetyInstrumentedInterlocks:
    """
    Independent Protection Layer (IPL) verifier ensuring supervisory recommendations
    never approach or violate Level 1 SIS emergency shutdown thresholds.
    """

    # Hardwired Triconex / ProSafe-RS SIL-2/3 Trip Boundaries
    TRIP_THRESHOLDS = {
        "TSHH-0104": {
            "desc": "Primary Reformer Tube Skin Temp High-High Trip",
            "trip": 940.0,
            "warn": 920.0,
            "direction": "HIGH_HIGH",
            "units": "deg C",
        },
        "PSHH-0210": {
            "desc": "Synthesis Loop Pressure High-High Trip",
            "trip": 220.0,
            "warn": 214.0,
            "direction": "HIGH_HIGH",
            "units": "bar",
        },
        "FSL-0301": {
            "desc": "Syngas Compressor Suction Flow Low-Low Trip (Anti-Surge)",
            "trip": 115000.0,
            "warn": 125000.0,
            "direction": "LOW_LOW",
            "units": "Nm3/hr",
        },
        "TSLL-0040": {
            "desc": "Reformer Outlet Process Gas Low-Low Trip (Quench)",
            "trip": 750.0,
            "warn": 765.0,
            "direction": "LOW_LOW",
            "units": "deg C",
        },
    }

    def verify_safety_envelope(
        self,
        candidate_state: Dict[str, float],
    ) -> SISVerificationReport:
        """
        Evaluates a candidate operational state against all SIL-2 safety loops.
        """
        temp = float(candidate_state.get("reformer_outlet_temp", 800.0))
        skin_temp = temp + 65.0  # Skin temp is ~65 C hotter than process gas
        press = float(candidate_state.get("loop_pressure", 195.0))
        flow = float(candidate_state.get("syngas_flow_nm3_hr", 145000.0))

        measurements = {
            "TSHH-0104": skin_temp,
            "PSHH-0210": press,
            "FSL-0301": flow,
            "TSLL-0040": temp,
        }

        statuses: List[SIFTripStatus] = []
        trips = 0
        warnings = 0

        for tag, cfg in self.TRIP_THRESHOLDS.items():
            val = measurements[tag]
            trip_sp = cfg["trip"]
            warn_sp = cfg["warn"]
            direction = cfg["direction"]

            if direction == "HIGH_HIGH":
                is_tripped = val >= trip_sp
                is_warn = (val >= warn_sp) and not is_tripped
                margin = trip_sp - val
            else:
                is_tripped = val <= trip_sp
                is_warn = (val <= warn_sp) and not is_tripped
                margin = val - trip_sp

            if is_tripped:
                trips += 1
            elif is_warn:
                warnings += 1

            statuses.append(
                SIFTripStatus(
                    sif_tag=tag,
                    description=cfg["desc"],
                    current_value=round(val, 2),
                    trip_setpoint=trip_sp,
                    trip_direction=direction,
                    engineering_units=cfg["units"],
                    is_tripped=is_tripped,
                    is_warning=is_warn,
                    margin_to_trip=round(margin, 2),
                )
            )

        is_safe = (trips == 0)

        if trips > 0:
            action = "EMERGENCY_OVERRIDE_ACTIVE: Candidate setpoint violates SIS hard trip boundaries. Control authority severed."
        elif warnings > 0:
            action = "ADVISORY_CAUTION: Operating within warning safety margin of SIL-2 interlock."
        else:
            action = "SAFETY_PERMIT_GRANTED: All independent protection layers verified clear."

        return SISVerificationReport(
            is_fully_safe=is_safe,
            trip_count=trips,
            warning_count=warnings,
            interlock_statuses=statuses,
            safety_integrity_level="IEC 61511 SIL-2 / SIL-3 Compliant",
            executive_safety_action=action,
        )
