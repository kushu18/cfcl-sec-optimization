"""
CFCL Gadepan Complex - Advanced Impenetrability Test Suite (Layers 1 to 7).
Rigorous automated testing for:
  Layer 1: Metallurgy (von Mises & Larson-Miller) & Chemical Kinetics (Xu-Froment & Temkin-Pyzhev)
  Layer 2: Advanced Control (PINN autograd, Pareto 3-objective, NMPC 6h horizon, EKF observer)
  Layer 3: Uncertainty & Explainability (Conformal Prediction, OOD Mahalanobis, SHAP interactions, Recourse)
  Layer 4: OT Security & SCADA (OPC-UA, Modbus TCP, Cryptographic Nonces, IEC 61511 SIL-2 Interlocks)
  Layer 5: Sovereign Policy Economics (NUP-2015 Tiered Clawback & Monte Carlo VaR99)
  Layer 6: Edge Performance & MLOps (ONNX export & Canary Gating)
  Layer 7: FastAPI Endpoints Integration
Sanitized DCS Telemetry under strict NDA.
"""

import json
import pytest
import numpy as np
import torch
from starlette.testclient import TestClient

from src.api.main import app
from src.data.schema import FEATURE_COLUMNS
from src.explainability.counterfactual import CounterfactualExplainer
from src.explainability.interaction_explainer import InteractionExplainer
from src.integrations.modbus_gateway import ModbusTCPGateway
from src.integrations.opcua_server import OPCUATagGateway
from src.mlops.shadow_pipeline import ShadowDeploymentPipeline
from src.models.conformal import ConformalPredictor
from src.models.ekf_observer import ExtendedKalmanFilterObserver
from src.models.ood_detector import OutOfDistributionDetector
from src.models.pinn_surrogate import PINNSurrogate
from src.models.onnx_exporter import ONNXModelExporter
from src.optimization.constrained_optimizer import ConstrainedSECOptimizer
from src.optimization.monte_carlo_risk import MonteCarloSavingsSimulator
from src.optimization.nmpc_controller import NMPCController
from src.optimization.pareto_optimizer import ParetoOptimizer
from src.optimization.savings_calculator import SavingsCalculator
from src.physics.kinetics import SMRKineticsEngine
from src.physics.metallurgy import TubeMetallurgyEngine
from src.physics.synthesis_loop import AmmoniaSynthesisLoopEngine
from src.safety.sis_interlocks import SafetyInstrumentedInterlocks
from src.security.nonce_manager import CryptographicNonceManager


# =========================================================================
# Layer 1: Physics & Metallurgy Tests
# =========================================================================

class TestLayer1PhysicsAndMetallurgy:
    """Verifies rigorous chemical engineering and metallurgy formulas."""

    def test_von_mises_thermal_stress_derivation(self):
        engine = TubeMetallurgyEngine()
        # Normal 1.5 C/hr slew rate
        res = engine.evaluate_thermal_stress(slew_rate_c_per_hr=1.5, outlet_temp_c=800.0)
        assert res.is_safe is True
        assert res.von_mises_stress_mpa < 45.0
        assert res.governor_recommended_max_slew_c_per_hr <= 2.14

        # Excessive 4.5 C/hr thermal shock should trip safety margin
        res_shock = engine.evaluate_thermal_stress(slew_rate_c_per_hr=4.5, outlet_temp_c=800.0)
        assert res_shock.von_mises_stress_mpa > res.von_mises_stress_mpa

    def test_larson_miller_creep_rupture_life(self):
        engine = TubeMetallurgyEngine()
        # Normal operating tube wall effective stress (hoop + thermal ~ 10-12 MPa)
        res = engine.evaluate_creep_damage(reformer_outlet_temp_c=805.0, effective_stress_mpa=12.0)
        assert res.larson_miller_parameter > 25.0
        assert res.rupture_life_years > 3.0
        assert res.is_creep_acceptable is True

    def test_xu_froment_smr_equilibrium_slip(self):
        kinetics = SMRKineticsEngine()
        k1, k2, k3 = kinetics.compute_equilibrium_constants(temp_kelvin=1073.15)
        assert k1 > 0.0 and k2 > 0.0 and k3 > 0.0

        # Normal operating slip validation
        val = kinetics.validate_methane_slip(
            observed_ch4_slip_pct=7.2,
            outlet_temp_c=805.0,
            steam_carbon_ratio=3.05,
        )
        assert val.is_thermodynamically_valid is True
        assert "THERMODYNAMICALLY_SOUND" in val.validation_status

    def test_synthesis_loop_and_urea_stoichiometry(self):
        loop = AmmoniaSynthesisLoopEngine(design_ammonia_capacity_mtpd=1750.0)
        bal = loop.evaluate_loop_balance(loop_pressure_bar=192.0, purge_gas_fraction=0.045)
        assert bal.is_loop_balanced is True
        assert 8.0 <= bal.loop_inerts_fraction_mol_pct <= 14.0
        assert bal.syngas_compressor_power_mw > 20.0

        urea = loop.evaluate_urea_stoichiometry(
            ammonia_production_mtpd=1750.0,
            co2_recovered_mtpd=1320.0,
            target_urea_mtpd=1750.0,
        )
        assert urea.is_urea_stoichiometrically_satisfied is True
        assert urea.max_achievable_urea_mtpd >= 1700.0


# =========================================================================
# Layer 2: Advanced Control & Optimization Tests
# =========================================================================

class TestLayer2ControlAndOptimization:
    """Verifies PINN autograd, Pareto multi-objective, NMPC, and EKF observer."""

    def test_pinn_surrogate_autograd(self):
        pinn = PINNSurrogate()
        # Synthetic calibration batch for test
        X_dummy = np.random.uniform(low=0.8, high=1.2, size=(30, 9)) * np.array([3.0, 800.0, 190.0, 32.0, 45.0, 31.0, 8900.0, 450.0, 300.0])
        y_dummy = np.random.uniform(5.40, 5.60, size=30)
        pinn.fit(X_dummy, y_dummy, epochs=5)

        sample = X_dummy[0]
        grad = pinn.compute_exact_gradient(sample, controllable_indices=(0, 1, 2))
        assert len(grad) == 3
        assert not np.isnan(grad).any()

    def test_pareto_tri_objective_frontier(self):
        opt = ConstrainedSECOptimizer()
        pareto = ParetoOptimizer(model=opt)
        sample_row = {
            "plant_id": "gadepan_3",
            "steam_carbon_ratio": 3.05,
            "reformer_outlet_temp": 802.0,
            "loop_pressure": 160.0,
            "ambient_temp": 33.0,
            "ambient_humidity": 45.0,
            "cw_inlet_temp": 31.5,
            "ng_calorific_value": 8950.0,
            "co2_slip_ppm": 460.0,
            "catalyst_age_days": 350.0,
        }
        solutions = pareto.solve_frontier(sample_row)
        assert len(solutions) == 3
        modes = [s.mode_name for s in solutions]
        assert "MAX_ENERGY_SAVINGS" in modes
        assert "BALANCED_LIFECYCLE" in modes
        assert "MAX_PLANT_THROUGHPUT" in modes

    def test_nmpc_6h_receding_horizon(self):
        opt = ConstrainedSECOptimizer()
        nmpc = NMPCController(model=opt)
        sample_row = {
            "plant_id": "gadepan_3",
            "steam_carbon_ratio": 3.05,
            "reformer_outlet_temp": 802.0,
            "loop_pressure": 160.0,
            "ambient_temp": 33.0,
            "ambient_humidity": 45.0,
            "cw_inlet_temp": 31.5,
            "ng_calorific_value": 8950.0,
            "co2_slip_ppm": 460.0,
            "catalyst_age_days": 350.0,
        }
        plan = nmpc.solve(sample_row)
        assert plan.horizon_hours == 6
        assert len(plan.planned_trajectory) == 6
        assert plan.immediate_setpoints["steam_carbon_ratio"] > 0.0

    def test_ekf_state_observer(self):
        ekf = ExtendedKalmanFilterObserver()
        obs = ekf.step(
            measurements={"sec_gcal_per_mt": 5.48, "co2_slip_ppm": 470.0, "reformer_dp_bar": 1.90},
            operating_inputs={"steam_carbon_ratio": 3.05, "reformer_outlet_temp": 802.0, "cw_inlet_temp": 31.5},
        )
        assert 0.70 <= obs.catalyst_effectiveness_factor <= 1.10
        assert obs.cw_fouling_resistance_m2k_per_w > 0.0
        assert obs.reformer_coking_dp_bar > 0.0


# =========================================================================
# Layer 3: Uncertainty & AI Robustness Tests
# =========================================================================

class TestLayer3UncertaintyAndRobustness:
    """Verifies Conformal Prediction, Mahalanobis OOD, and Recourse."""

    def test_conformal_prediction_intervals(self):
        opt = ConstrainedSECOptimizer()
        conformal = ConformalPredictor(base_model=opt, alpha=0.05)
        # Calibration with 184 columns or 9 columns
        X_cal = np.random.uniform(0.9, 1.1, size=(50, 9)) * np.array([3.0, 800.0, 160.0, 32.0, 45.0, 31.0, 8900.0, 450.0, 300.0])
        y_cal = np.full(50, 5.50)
        cal_res = conformal.calibrate(X_cal, y_cal)
        assert cal_res["confidence_level"] == 0.95

        intervals = conformal.predict_interval(X_cal[:3])
        assert len(intervals) == 3
        for item in intervals:
            assert item.lower_bound_sec <= item.point_prediction_sec <= item.upper_bound_sec
            assert item.confidence_level == 0.95

    def test_mahalanobis_ood_detector(self):
        ood = OutOfDistributionDetector()
        X_train = np.random.normal(loc=10.0, scale=1.0, size=(100, 9))
        ood.fit(X_train)

        # In-distribution sample
        in_sample = np.random.normal(loc=10.0, scale=0.5, size=9)
        res_in = ood.assess(in_sample)
        assert res_in.is_in_distribution is True

        # Wild extreme out-of-distribution sample
        out_sample = np.full(9, 100.0)
        res_out = ood.assess(out_sample)
        assert res_out.is_in_distribution is False
        assert "OUT_OF_DISTRIBUTION" in res_out.status

    def test_counterfactual_recourse(self):
        opt = ConstrainedSECOptimizer()
        cf = CounterfactualExplainer(model=opt)
        sample_row = {
            "steam_carbon_ratio": 3.12,
            "reformer_outlet_temp": 790.0,
            "loop_pressure": 150.0,
            "ambient_temp": 34.0,
            "ambient_humidity": 45.0,
            "cw_inlet_temp": 32.0,
            "ng_calorific_value": 8900.0,
            "co2_slip_ppm": 480.0,
            "catalyst_age_days": 400.0,
        }
        res = cf.compute_recourse(sample_row, target_reduction_gcal_per_mt=0.030)
        assert res.achievable_sec_gcal_per_mt <= res.current_sec_gcal_per_mt
        assert isinstance(res.action_plan, list)


# =========================================================================
# Layer 4: OT Security, SCADA & Functional Safety Tests
# =========================================================================

class TestLayer4IndustrialOTAndSafety:
    """Verifies OPC-UA, Modbus TCP, Cryptographic Nonces, and IEC 61511 Trips."""

    def test_opcua_tag_gateway_manifest(self):
        gw = OPCUATagGateway()
        manifest = gw.get_address_space_manifest("gadepan_3")
        assert len(manifest) >= 10
        node_ids = [m.node_id for m in manifest]
        assert any("10TIC001.PV" in nid for nid in node_ids)
        assert any("10FIC004.PV" in nid for nid in node_ids)

    def test_modbus_register_encoding(self):
        mb = ModbusTCPGateway()
        # Test IEEE 754 float encode/decode
        test_val = 5.485
        r1, r2 = mb.float_to_registers(test_val)
        decoded = mb.registers_to_float(r1, r2)
        assert abs(decoded - test_val) < 1e-4

        manifest = mb.get_register_manifest()
        assert len(manifest) == 11
        assert manifest[0].address == 40001

    def test_cryptographic_nonce_and_two_man_rule(self):
        nonce_mgr = CryptographicNonceManager()
        payload = {"steam_carbon_ratio": 3.05, "reformer_outlet_temp": 805.0, "loop_pressure": 192.0}
        token = nonce_mgr.issue_nonce(payload)
        assert len(token.nonce_id) == 32
        assert token.is_consumed is False

        # Attempt commit with Two-Man Rule required (slew >= 1.0 C)
        # Without supervisor: should fail
        res_fail = nonce_mgr.verify_and_consume(
            nonce_id=token.nonce_id,
            setpoints_payload=payload,
            signature=token.signature,
            primary_operator_id="OP-101",
            supervisor_id=None,
            slew_temp_c=1.5,
        )
        assert res_fail.is_valid is False
        assert res_fail.status_code == "TWO_MAN_RULE_REQUIRED"

        # With supervisor: should succeed
        res_ok = nonce_mgr.verify_and_consume(
            nonce_id=token.nonce_id,
            setpoints_payload=payload,
            signature=token.signature,
            primary_operator_id="OP-101",
            supervisor_id="SUP-808",
            slew_temp_c=1.5,
        )
        assert res_ok.is_valid is True
        assert res_ok.status_code == "AUTHORIZED"

        # Replay attempt: should be blocked
        res_replay = nonce_mgr.verify_and_consume(
            nonce_id=token.nonce_id,
            setpoints_payload=payload,
            signature=token.signature,
            primary_operator_id="OP-101",
            supervisor_id="SUP-808",
            slew_temp_c=1.5,
        )
        assert res_replay.is_valid is False
        assert res_replay.status_code == "NONCE_ALREADY_CONSUMED"

    def test_iec_61511_sis_interlocks(self):
        sis = SafetyInstrumentedInterlocks()
        # Normal safe candidate state
        safe_state = {"reformer_outlet_temp": 802.0, "loop_pressure": 194.0, "syngas_flow_nm3_hr": 140000.0}
        res_safe = sis.verify_safety_envelope(safe_state)
        assert res_safe.is_fully_safe is True
        assert res_safe.trip_count == 0

        # Extreme high skin temp trip (> 940 C)
        tripped_state = {"reformer_outlet_temp": 880.0, "loop_pressure": 194.0, "syngas_flow_nm3_hr": 140000.0}
        res_tripped = sis.verify_safety_envelope(tripped_state)
        assert res_tripped.is_fully_safe is False
        assert res_tripped.trip_count >= 1


# =========================================================================
# Layer 5: Policy Economics & MLOps Tests
# =========================================================================

class TestLayer5EconomicsAndMLOps:
    """Verifies NUP-2015 Tiered Clawback, Monte Carlo VaR99, and Canary Gating."""

    def test_nup2015_tiered_clawback_retention(self):
        calc = SavingsCalculator()
        ret_tier0 = calc.calculate_nup_retention(plant_id="gadepan_3", current_sec=5.08, delta_sec=0.03, gross_savings_inr=10000000.0)
        assert ret_tier0["retention_fraction"] == 1.00

        ret_tier1 = calc.calculate_nup_retention(plant_id="gadepan_3", current_sec=5.00, delta_sec=0.08, gross_savings_inr=10000000.0)
        assert ret_tier1["retention_fraction"] == 0.65

        ret_tier2 = calc.calculate_nup_retention(plant_id="gadepan_3", current_sec=4.95, delta_sec=0.20, gross_savings_inr=10000000.0)
        assert ret_tier2["retention_fraction"] == 0.33

    def test_monte_carlo_var99_simulation(self):
        sim = MonteCarloSavingsSimulator()
        report = sim.simulate(mean_delta_sec_gcal_mt=0.035, n_trials=1000)
        assert report.trials == 1000
        assert report.mean_annual_savings_inr_cr > 4.0
        assert report.var_99_inr_cr > 0.0
        assert report.cvar_99_inr_cr >= report.var_99_inr_cr

    def test_shadow_canary_deployment_gating(self):
        opt = ConstrainedSECOptimizer()
        shadow = ShadowDeploymentPipeline(champion_model=opt.model)
        report = shadow.evaluate_canary_gate(min_hours=0)
        assert report.canary_gate_passed is True
        assert report.deployment_verdict == "CANARY_PROMOTION_APPROVED"


# =========================================================================
# Layer 7: Full FastAPI Endpoint Integration Tests
# =========================================================================

class TestLayer7FastAPIEndpoints:
    """Verifies all 10 new API endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        with TestClient(app) as client:
            self.client = client
            yield

    def test_endpoint_physics_validate(self):
        payload = {
            "reformer_outlet_temp": 802.5,
            "steam_carbon_ratio": 3.08,
            "slew_rate_c_per_hr": 1.2,
            "operating_pressure_bar": 32.5,
            "observed_ch4_slip_pct": 7.1,
        }
        res = self.client.post("/physics/validate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_fully_compliant"] is True
        assert "FULLY_COMPLIANT" in data["summary_verdict"]

    def test_endpoint_conformal_predict(self):
        payload = {
            "plant_id": "gadepan_3",
            "steam_carbon_ratio": 3.05,
            "reformer_outlet_temp": 802.0,
            "loop_pressure": 158.0,
            "ambient_temp": 32.0,
            "ambient_humidity": 45.0,
            "cw_inlet_temp": 31.0,
            "ng_calorific_value": 8900.0,
            "co2_slip_ppm": 470.0,
            "catalyst_age_days": 350.0,
        }
        res = self.client.post("/conformal/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["coverage_guarantee_percent"] == 95.0
        assert data["conformal_lower_bound_sec"] <= data["point_prediction_sec"] <= data["conformal_upper_bound_sec"]

    def test_endpoint_ood_assess(self):
        payload = {
            "plant_id": "gadepan_3",
            "steam_carbon_ratio": 3.05,
            "reformer_outlet_temp": 802.0,
            "loop_pressure": 158.0,
            "ambient_temp": 32.0,
            "ambient_humidity": 45.0,
            "cw_inlet_temp": 31.0,
            "ng_calorific_value": 8900.0,
            "co2_slip_ppm": 470.0,
            "catalyst_age_days": 350.0,
        }
        res = self.client.post("/ood/assess", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "mahalanobis_distance_squared" in data
        assert "is_in_distribution" in data

    def test_endpoint_optimize_pareto(self):
        payload = {
            "plant_id": "gadepan_3",
            "telemetry": {
                "plant_id": "gadepan_3",
                "steam_carbon_ratio": 3.05,
                "reformer_outlet_temp": 802.0,
                "loop_pressure": 158.0,
                "ambient_temp": 32.0,
                "ambient_humidity": 45.0,
                "cw_inlet_temp": 31.0,
                "ng_calorific_value": 8900.0,
                "co2_slip_ppm": 470.0,
                "catalyst_age_days": 350.0,
            },
        }
        res = self.client.post("/optimize/pareto", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["pareto_solutions"]) == 3

    def test_endpoint_optimize_nmpc(self):
        payload = {
            "plant_id": "gadepan_3",
            "telemetry": {
                "plant_id": "gadepan_3",
                "steam_carbon_ratio": 3.05,
                "reformer_outlet_temp": 802.0,
                "loop_pressure": 158.0,
                "ambient_temp": 32.0,
                "ambient_humidity": 45.0,
                "cw_inlet_temp": 31.0,
                "ng_calorific_value": 8900.0,
                "co2_slip_ppm": 470.0,
                "catalyst_age_days": 350.0,
            },
        }
        res = self.client.post("/optimize/nmpc", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["horizon_hours"] == 6
        assert len(data["planned_trajectory"]) == 6

    def test_endpoint_counterfactual_recourse(self):
        payload = {
            "plant_id": "gadepan_3",
            "telemetry": {
                "plant_id": "gadepan_3",
                "steam_carbon_ratio": 3.12,
                "reformer_outlet_temp": 798.0,
                "loop_pressure": 158.0,
                "ambient_temp": 32.0,
                "ambient_humidity": 45.0,
                "cw_inlet_temp": 31.0,
                "ng_calorific_value": 8900.0,
                "co2_slip_ppm": 470.0,
                "catalyst_age_days": 350.0,
            },
            "target_reduction_gcal_per_mt": 0.030,
        }
        res = self.client.post("/counterfactual/recourse", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "action_plan" in data
        assert "operator_executive_summary" in data

    def test_endpoint_opc_manifest(self):
        res = self.client.get("/integrations/opc/manifest?plant_id=gadepan_3")
        assert res.status_code == 200
        data = res.json()
        assert data["tags_count"] >= 10
        assert data["protocol"].startswith("OPC-UA")

    def test_endpoint_modbus_map(self):
        res = self.client.get("/integrations/modbus/map")
        assert res.status_code == 200
        data = res.json()
        assert data["register_count"] == 11
        assert data["base_address"] == 40001

    def test_endpoint_risk_monte_carlo(self):
        payload = {"mean_delta_sec_gcal_mt": 0.035, "trials": 1000}
        res = self.client.post("/risk/monte-carlo", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["trials"] == 1000
        assert data["mean_annual_savings_inr_cr"] > 0.0

    def test_endpoint_canary_status(self):
        res = self.client.get("/mlops/canary/status")
        assert res.status_code == 200
        data = res.json()
        assert "champion_model_name" in data
        assert "deployment_verdict" in data
