"""
CFCL Gadepan Complex - Modbus TCP Industrial Register Map.
Provides standard IEEE 754 Float32 holding register encoding/decoding
for legacy PLC/SCADA interfaces across auxiliary plant sections.
Sanitized DCS Telemetry under strict NDA.
"""

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from src.utils.logger import get_logger

logger = get_logger("ModbusGateway")


@dataclass
class ModbusRegisterEntry:
    """A documented Modbus holding register definition."""
    address: int
    num_registers: int
    variable_name: str
    data_type: str
    engineering_units: str
    access: str
    current_value: float


class ModbusTCPGateway:
    """
    Modbus TCP Gateway implementing 16-bit word and IEEE 754 Float32 mapping.
    Holding register base 40001 (0x0000 offset).
    """

    REGISTER_MAP = [
        (40001, 2, "sec_gcal_per_mt", "Float32", "Gcal/MT", "RO"),
        (40003, 2, "steam_carbon_ratio", "Float32", "ratio", "RO"),
        (40005, 2, "reformer_outlet_temp", "Float32", "deg C", "RO"),
        (40007, 2, "loop_pressure", "Float32", "bar", "RO"),
        (40009, 2, "ambient_temp", "Float32", "deg C", "RO"),
        (40011, 2, "cw_inlet_temp", "Float32", "deg C", "RO"),
        (40101, 2, "sp_steam_carbon_ratio", "Float32", "ratio", "RW"),
        (40103, 2, "sp_reformer_outlet_temp", "Float32", "deg C", "RW"),
        (40105, 2, "sp_loop_pressure", "Float32", "bar", "RW"),
        (40201, 1, "watchdog_heartbeat", "UInt16", "ticks", "RW"),
        (40202, 1, "sis_interlock_bitmask", "UInt16", "bits", "RO"),
    ]

    def __init__(self):
        # Register storage as 16-bit integers
        self.registers: Dict[int, int] = {}
        # Initialize default register values
        self.write_float(40001, 5.460)
        self.write_float(40003, 3.080)
        self.write_float(40005, 801.5)
        self.write_float(40007, 194.2)
        self.write_float(40009, 33.4)
        self.write_float(40011, 31.8)
        self.write_float(40101, 3.050)
        self.write_float(40103, 803.0)
        self.write_float(40105, 192.5)
        self.registers[40201] = 100
        self.registers[40202] = 0b0000  # Healthy

    @staticmethod
    def float_to_registers(val: float) -> Tuple[int, int]:
        """Encodes IEEE 754 single precision float into two 16-bit registers (Big-Endian)."""
        raw_bytes = struct.pack(">f", float(val))
        reg1, reg2 = struct.unpack(">HH", raw_bytes)
        return reg1, reg2

    @staticmethod
    def registers_to_float(reg1: int, reg2: int) -> float:
        """Decodes two 16-bit registers into an IEEE 754 single precision float."""
        raw_bytes = struct.pack(">HH", reg1, reg2)
        return float(struct.unpack(">f", raw_bytes)[0])

    def write_float(self, start_address: int, val: float) -> None:
        """Writes a 32-bit float across two consecutive 16-bit holding registers."""
        r1, r2 = self.float_to_registers(val)
        self.registers[start_address] = r1
        self.registers[start_address + 1] = r2

    def read_float(self, start_address: int) -> float:
        """Reads a 32-bit float from two consecutive 16-bit holding registers."""
        r1 = self.registers.get(start_address, 0)
        r2 = self.registers.get(start_address + 1, 0)
        return self.registers_to_float(r1, r2)

    def get_register_manifest(self) -> List[ModbusRegisterEntry]:
        """Returns the complete register dictionary for field instrument engineers."""
        manifest = []
        for addr, n_reg, name, dtype, units, acc in self.REGISTER_MAP:
            if dtype == "Float32":
                val = self.read_float(addr)
            else:
                val = float(self.registers.get(addr, 0))

            manifest.append(
                ModbusRegisterEntry(
                    address=addr,
                    num_registers=n_reg,
                    variable_name=name,
                    data_type=dtype,
                    engineering_units=units,
                    access=acc,
                    current_value=round(val, 3),
                )
            )
        return manifest
