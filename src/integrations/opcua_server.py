"""
CFCL Gadepan Complex - Native OPC-UA Industrial Tag Gateway Server.
Implements IEC 62541 OPC-UA information model mapping DCS Level 2 sensor PVs
and supervisory Level 3 optimization SP targets for Yokogawa CENTUM VP & Honeywell Experion.
Sanitized DCS Telemetry under strict NDA.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.data.schema import CONTROLLABLE_FEATURES, FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("OPCUAGateway")


@dataclass
class OPCUANodeMapping:
    """Mapping between a process variable and an industrial OPC-UA Node ID."""
    variable_name: str
    node_id: str
    data_type: str
    engineering_units: str
    access_level: str
    current_value: float


class OPCUATagGateway:
    """
    Industrial OPC-UA Tag Gateway bridging the Purdue Model Level 2/3 boundary.
    Provides standard IEC 62541 address space mapping for CFCL Gadepan Units.
    """

    TAG_DICTIONARY = {
        "steam_carbon_ratio": {
            "tag": "10FIC004.PV",
            "sp_tag": "10FIC004.SP_OPT",
            "units": "ratio",
            "type": "Float32",
        },
        "reformer_outlet_temp": {
            "tag": "10TIC001.PV",
            "sp_tag": "10TIC001.SP_OPT",
            "units": "deg C",
            "type": "Float32",
        },
        "loop_pressure": {
            "tag": "10PIC012.PV",
            "sp_tag": "10PIC012.SP_OPT",
            "units": "bar",
            "type": "Float32",
        },
        "ambient_temp": {
            "tag": "01TI0101.PV",
            "units": "deg C",
            "type": "Float32",
        },
        "ambient_humidity": {
            "tag": "01HI0102.PV",
            "units": "%",
            "type": "Float32",
        },
        "cw_inlet_temp": {
            "tag": "10TI0205.PV",
            "units": "deg C",
            "type": "Float32",
        },
        "ng_calorific_value": {
            "tag": "10AI0099.PV",
            "units": "kcal/Sm3",
            "type": "Float32",
        },
        "co2_slip_ppm": {
            "tag": "10AI0310.PV",
            "units": "ppm",
            "type": "Float32",
        },
        "catalyst_age_days": {
            "tag": "00XI0001.PV",
            "units": "days",
            "type": "Float32",
        },
        "sec_gcal_per_mt": {
            "tag": "10SEC_CALC.PV",
            "units": "Gcal/MT",
            "type": "Float32",
        },
    }

    def __init__(self, endpoint_url: str = "opc.tcp://0.0.0.0:4840/cfcl/dcs/"):
        self.endpoint_url = endpoint_url
        self.namespace_uri = "http://chambalfert.in/gadepan/dcs/"
        self.plant_id = "gadepan_3"
        self.active_tags: Dict[str, float] = {
            "steam_carbon_ratio": 3.08,
            "reformer_outlet_temp": 801.5,
            "loop_pressure": 194.2,
            "ambient_temp": 33.4,
            "ambient_humidity": 48.0,
            "cw_inlet_temp": 31.8,
            "ng_calorific_value": 8950.0,
            "co2_slip_ppm": 475.0,
            "catalyst_age_days": 320.0,
            "sec_gcal_per_mt": 5.460,
        }
        self.target_setpoints: Dict[str, float] = {
            "steam_carbon_ratio": 3.05,
            "reformer_outlet_temp": 803.0,
            "loop_pressure": 192.5,
        }
        self.watchdog_counter = 0

    def get_node_id(self, plant_id: str, tag_name: str) -> str:
        """Formats the official standard OPC-UA String NodeId."""
        plant_code = "G3" if "3" in plant_id else ("G2" if "2" in plant_id else "G1")
        return f"ns=2;s=CFCL.{plant_code}.{tag_name}"

    def get_address_space_manifest(self, plant_id: Optional[str] = None) -> List[OPCUANodeMapping]:
        """Returns the full address space tag manifest for SCADA / DCS engineers."""
        pid = plant_id or self.plant_id
        manifest = []

        # Process measurement tags (Read-Only)
        for var_name, info in self.TAG_DICTIONARY.items():
            nid = self.get_node_id(pid, info["tag"])
            val = self.active_tags.get(var_name, 0.0)
            manifest.append(
                OPCUANodeMapping(
                    variable_name=var_name,
                    node_id=nid,
                    data_type=info["type"],
                    engineering_units=info["units"],
                    access_level="CurrentRead",
                    current_value=round(val, 3),
                )
            )

        # Setpoint write target tags (Read-Write)
        for c_var in CONTROLLABLE_FEATURES:
            info = self.TAG_DICTIONARY[c_var]
            if "sp_tag" in info:
                nid = self.get_node_id(pid, info["sp_tag"])
                val = self.target_setpoints.get(c_var, 0.0)
                manifest.append(
                    OPCUANodeMapping(
                        variable_name=f"{c_var}_target_sp",
                        node_id=nid,
                        data_type="Float32",
                        engineering_units=info["units"],
                        access_level="CurrentReadOrWrite",
                        current_value=round(val, 3),
                    )
                )

        # Heartbeat watchdog tag
        manifest.append(
            OPCUANodeMapping(
                variable_name="dcs_watchdog_pulse",
                node_id=self.get_node_id(pid, "10OPT_STATUS.WD"),
                data_type="UInt32",
                engineering_units="count",
                access_level="CurrentReadOrWrite",
                current_value=float(self.watchdog_counter),
            )
        )

        return manifest

    def read_pv(self, tag_name: str) -> float:
        """Reads instantaneous process variable value."""
        return self.active_tags.get(tag_name, 0.0)

    def write_sp(self, tag_name: str, value: float) -> bool:
        """Simulates writing a verified setpoint to the DCS tag buffer."""
        self.target_setpoints[tag_name] = value
        self.watchdog_counter += 1
        logger.info(f"OPC-UA Write: Tag {tag_name} set to {value:.3f}")
        return True
