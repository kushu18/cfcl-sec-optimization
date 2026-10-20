"""Tests for Production FastAPI Industrial Microservice Endpoints.

Verifies:
1. Root metadata and health check diagnostics.
2. DCS governor safety limits query endpoint.
3. High-throughput single and fleet-wide inference with 95% confidence intervals.
4. SLSQP closed-loop optimization and supervisory setpoint commitment handshakes.
5. Statistical anomaly detection with ISA-18.2 alarm priority and local SHAP.
6. Sensor drift surveillance and DCS fail-safe heartbeat watchdog.
7. Pydantic input rejection of physically unsafe process values (HTTP 422).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient


class TestGeneralEndpoints:
    """Verify landing, health, and governor configuration endpoints."""

    def test_root_endpoint(self, api_client: TestClient) -> None:
        """GET / returns service metadata and online status."""
        r = api_client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "Gadepan" in body.get("facility", "")
        assert body.get("status") == "ONLINE"

    def test_health_endpoint(self, api_client: TestClient) -> None:
        """GET /health verifies surrogate model KPIs and service uptime."""
        r = api_client.get("/health")
        assert r.status_code == 200
        h = r.json()
        assert h["status"] == "HEALTHY"
        assert h["surrogate_model_loaded"] == "Random_Forest"
        assert h["surrogate_model_r2"] >= 0.95
        assert h["surrogate_model_rmse"] <= 0.025
        assert len(h["fleet_units"]) == 3

    def test_governor_limits_endpoint(self, api_client: TestClient) -> None:
        """GET /governor/limits returns DCS safety interlock bounds and slew rates."""
        r = api_client.get("/governor/limits")
        assert r.status_code == 200
        gov = r.json()
        assert "safety_bounds" in gov
        assert gov["safety_bounds"]["steam_carbon_ratio"]["min"] == 2.80
        assert gov["safety_bounds"]["reformer_outlet_temp"]["max"] == 835.0
        assert len(gov["slew_rate_limits"]) == 3
        assert len(gov["plant_units"]) == 3


class TestInferenceEndpoints:
    """Verify single and multi-train real-time inference endpoints."""

    def test_predict_single_telemetry(
        self,
        api_client: TestClient,
        valid_telemetry_payload: dict
    ) -> None:
        """POST /predict produces millisecond SEC prediction with 95% CI."""
        r = api_client.post("/predict", json=valid_telemetry_payload)
        assert r.status_code == 200
        p = r.json()
        assert 4.80 <= p["predicted_sec_gcal_per_mt"] <= 5.15
        assert len(p["confidence_interval_95"]) == 2
        assert p["execution_time_ms"] < 200.0
        assert p["within_dcs_tolerance"] is True

        # First-principles thermodynamic check
        th = p.get("thermodynamics")
        assert th is not None
        assert 0.15 <= th["equilibrium_methane_slip_dry_pct"] <= 1.00
        assert th["radiant_tube_heat_flux_kw_m2"] <= 75.0
        assert th["thermodynamic_sanity_passed"] is True

    def test_predict_fleet_concurrent(
        self,
        api_client: TestClient,
        valid_telemetry_payload: dict
    ) -> None:
        """POST /predict/fleet evaluates all 3 operating units simultaneously."""
        g1 = valid_telemetry_payload.copy()
        g1["plant_id"] = "gadepan_1"
        g2 = valid_telemetry_payload.copy()
        g2["plant_id"] = "gadepan_2"

        fleet_req = {"telemetry_records": [g1, g2, valid_telemetry_payload]}
        r = api_client.post("/predict/fleet", json=fleet_req)
        assert r.status_code == 200
        fp = r.json()
        assert fp["fleet_records_evaluated"] == 3
        assert 4.90 <= fp["fleet_weighted_sec_gcal_per_mt"] <= 5.25
        assert fp["total_fuel_energy_rate_gcal_per_hr"] > 500.0
        assert fp["fleet_status"] == "NOMINAL"


class TestOptimizationEndpoints:
    """Verify supervisory optimization and DCS handshake transfer endpoints."""

    def test_optimize_setpoints(
        self,
        api_client: TestClient,
        valid_telemetry_payload: dict
    ) -> None:
        """POST /optimize calculates recommended setpoints respecting slew limits."""
        opt_req = {
            "telemetry": valid_telemetry_payload,
            "respect_slew_rate": True,
            "max_sc_delta": 0.05,
            "max_temp_delta": 2.0,
            "max_press_delta": 2.0,
        }
        r = api_client.post("/optimize", json=opt_req)
        assert r.status_code == 200
        o = r.json()
        assert o["expected_delta_sec"] >= -1e-5
        assert 2.80 <= o["recommended_setpoints"]["steam_carbon_ratio"] <= 3.40
        assert 780.0 <= o["recommended_setpoints"]["reformer_outlet_temp"] <= 835.0
        assert 145.0 <= o["recommended_setpoints"]["loop_pressure"] <= 185.0
        assert abs(o["setpoint_adjustments"]["steam_carbon_ratio"]) <= 0.051
        assert o["annual_savings_inr_crore"] >= 0.0

    def test_commit_setpoints_to_dcs(self, api_client: TestClient) -> None:
        """POST /optimize/commit executes operator handshake with SIL-2 verification."""
        commit_req = {
            "plant_id": "gadepan_3",
            "target_setpoints": {
                "steam_carbon_ratio": 3.02,
                "reformer_outlet_temp": 811.5,
                "loop_pressure": 158.5,
            },
            "operator_badge_id": "CFCL-OP-4491",
            "shift_incharge_token": "SIC-G3-AUTH-982",
            "control_mode": "SUPERVISORY_AUTO_RAMP",
        }
        r = api_client.post("/optimize/commit", json=commit_req)
        assert r.status_code == 200
        cm = r.json()
        assert cm["status"] == "COMMITTED_TO_DCS"
        assert "SIL2" in cm["safety_sil_verification"]
        assert "CFCL-DCS-TX" in cm["transaction_id"]
        assert "AUDIT-LOG" in cm["dcs_audit_log_id"]

    def test_commit_unsafe_setpoints_rejected(self, api_client: TestClient) -> None:
        """POST /optimize/commit must abort transfer if setpoints violate safety governors."""
        bad_req = {
            "plant_id": "gadepan_3",
            "target_setpoints": {
                "steam_carbon_ratio": 2.20,  # Below coking limit (2.80)
                "reformer_outlet_temp": 811.5,
                "loop_pressure": 158.5,
            },
            "operator_badge_id": "CFCL-OP-4491",
            "shift_incharge_token": "SIC-G3-AUTH-982",
            "control_mode": "SUPERVISORY_AUTO_RAMP",
        }
        r = api_client.post("/optimize/commit", json=bad_req)
        assert r.status_code == 400


class TestDiagnosticsAndSafety:
    """Verify anomaly detection, watchdog heartbeat, drift status, and safety rejection."""

    def test_anomaly_detection_with_isa_priority(
        self,
        api_client: TestClient,
        valid_telemetry_payload: dict
    ) -> None:
        """POST /anomaly flags excursions and classifies ISA-18.2 alarm priority."""
        anom_req = {
            "telemetry": valid_telemetry_payload,
            "actual_sec": 5.35,  # Deliberate excursion
            "z_threshold": 2.0,
        }
        r = api_client.post("/anomaly", json=anom_req)
        assert r.status_code == 200
        a = r.json()
        assert a["is_anomaly"] is True
        assert a["isa18_2_alarm_priority"] in ["PRIORITY_1_CRITICAL_TRIP", "PRIORITY_2_HIGH_EXCURSION"]
        assert a["failure_mode"] is not None
        assert len(a["top_contributing_factors"]) > 0

    def test_dcs_watchdog_heartbeat(self, api_client: TestClient) -> None:
        """GET /dcs/watchdog reports active heartbeat and fail-safe fallback setpoints."""
        r = api_client.get("/dcs/watchdog")
        assert r.status_code == 200
        wd = r.json()
        assert wd["service_status"] == "ONLINE_HEALTHY"
        assert wd["heartbeat_latency_ms"] < 50.0
        assert len(wd["failsafe_fallback_setpoints"]) == 3

    def test_drift_status_endpoint(self, api_client: TestClient) -> None:
        """GET /drift/status returns two-sample KS drift monitoring audit."""
        r = api_client.get("/drift/status")
        assert r.status_code == 200
        d = r.json()
        assert "overall_severity" in d
        assert isinstance(d["drifted_tags"], list)

    def test_pydantic_safety_guard_rejects_unsafe_telemetry(
        self,
        api_client: TestClient,
        unsafe_telemetry_payload: dict
    ) -> None:
        """Pydantic rejects physically unsafe telemetry payloads with HTTP 422."""
        r = api_client.post("/predict", json=unsafe_telemetry_payload)
        assert r.status_code == 422
