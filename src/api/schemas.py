"""Pydantic V2 Request & Response Schemas for CFCL SEC Optimization API.

Enforces strict DCS physical range bounds, telemetry data validation,
and structured API payloads for real-time supervisory integration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class PlantTelemetryInput(BaseModel):
    """Real-time DCS telemetry snapshot input for an operational plant unit."""
    
    plant_id: Literal["gadepan_1", "gadepan_2", "gadepan_3"] = Field(
        ...,
        description="Operating unit identifier: gadepan_1 (MEA, 1993), gadepan_2 (Benfield, 1999), gadepan_3 (aMDEA, 2019)",
        example="gadepan_3"
    )
    timestamp: Optional[str] = Field(
        None,
        description="DCS observation timestamp in ISO 8601 format (defaults to current server time)",
        example="2026-06-30T14:00:00"
    )
    steam_carbon_ratio: float = Field(
        ...,
        ge=2.5,
        le=4.0,
        description="Primary reformer steam injection ratio (mol/mol). Normal: 2.80 - 3.40",
        example=3.05
    )
    reformer_outlet_temp: float = Field(
        ...,
        ge=700.0,
        le=900.0,
        description="Primary reformer exit gas temperature (°C). Normal: 780.0 - 835.0",
        example=812.5
    )
    loop_pressure: float = Field(
        ...,
        ge=130.0,
        le=200.0,
        description="Ammonia synthesis loop pressure (kg/cm²). Normal: 145.0 - 185.0",
        example=158.0
    )
    ambient_temp: float = Field(
        ...,
        ge=0.0,
        le=55.0,
        description="Kota ambient dry-bulb temperature (°C). Range: 10.0 - 50.0",
        example=42.0
    )
    ambient_humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative ambient humidity (%). Range: 10.0 - 100.0",
        example=38.0
    )
    cw_inlet_temp: float = Field(
        ...,
        ge=15.0,
        le=50.0,
        description="Cooling water supply temperature from Kalisindh river / cooling tower (°C)",
        example=33.5
    )
    ng_calorific_value: float = Field(
        ...,
        ge=7500.0,
        le=10000.0,
        description="Natural gas lower heating value (LHV) from HBJ pipeline (kcal/SCM)",
        example=8620.0
    )
    co2_slip_ppm: float = Field(
        ...,
        ge=10.0,
        le=3500.0,
        description="CO₂ slip breakthrough from absorption column (ppm)",
        example=450.0
    )
    catalyst_age_days: float = Field(
        ...,
        ge=0.0,
        le=3000.0,
        description="Operating days since primary reformer nickel catalyst installation",
        example=180.0
    )
    shift: Optional[Literal["morning", "afternoon", "night"]] = Field(
        "afternoon",
        description="Operating shift: morning (06-14h), afternoon (14-22h), night (22-06h)"
    )

    @field_validator("steam_carbon_ratio")
    @classmethod
    def validate_sc_coking(cls, v: float) -> float:
        if v < 2.80:
            # Advisory warning: lower limit risks carbon coking
            pass
        return v


class OptimizationRequest(BaseModel):
    """Closed-loop supervisory setpoint optimization request payload."""
    telemetry: PlantTelemetryInput
    respect_slew_rate: bool = Field(
        True,
        description="Enforce 1-hour DCS slew-rate ramp limits (|ΔS/C| ≤ 0.05, |ΔT| ≤ 2.0°C, |ΔP| ≤ 2.0 kg/cm²)"
    )
    max_sc_delta: float = Field(0.05, ge=0.01, le=0.20, description="Max allowable S/C ratio adjustment per hour")
    max_temp_delta: float = Field(2.0, ge=0.5, le=10.0, description="Max allowable Reformer Temp adjustment (°C/hr)")
    max_press_delta: float = Field(2.0, ge=0.5, le=10.0, description="Max allowable Loop Pressure adjustment (kg/cm²/hr)")


class AnomalyDetectionRequest(BaseModel):
    """Operational excursion detection and root-cause analysis request payload."""
    telemetry: PlantTelemetryInput
    actual_sec: float = Field(
        ...,
        ge=4.0,
        le=7.0,
        description="Current measured Specific Energy Consumption from DCS (Gcal/MT urea)",
        example=5.32
    )
    z_threshold: float = Field(2.0, ge=1.5, le=4.0, description="Excursion detection threshold (standard deviations)")


class FleetPredictionRequest(BaseModel):
    """Multi-train simultaneous DCS telemetry input for the entire Gadepan complex."""
    telemetry_records: List[PlantTelemetryInput] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of plant telemetry snapshots for concurrent fleet evaluation"
    )


# =============================================================================
# Response Schemas
# =============================================================================

class ThermodynamicSanityMetrics(BaseModel):
    """First-principles thermodynamic equilibrium and equipment safety envelope metrics."""
    equilibrium_methane_slip_dry_pct: float = Field(
        ...,
        description="Thermodynamic equilibrium methane slip from primary reformer exit gas (dry mol%)"
    )
    radiant_tube_heat_flux_kw_m2: float = Field(
        ...,
        description="Average heat flux across Incoloy 800H reformer catalyst tubes (kW/m², limit <= 75.0)"
    )
    compressor_anti_surge_margin_pct: float = Field(
        ...,
        description="Syngas centrifugal synthesis compressor operating margin from surge limit (%)"
    )
    thermodynamic_sanity_passed: bool = Field(
        ...,
        description="Whether all first-principles thermodynamic and mechanical criteria are satisfied"
    )
    safety_governor_status: str = Field(
        ...,
        description="Safety status: NOMINAL_ENVELOPE, ADVISORY_APPROACHING_LIMIT, or HARD_INTERLOCK_EXCEEDED"
    )


class PredictionResponse(BaseModel):
    """Surrogate model SEC prediction output with statistical confidence intervals."""
    plant_id: str
    predicted_sec_gcal_per_mt: float
    confidence_interval_95: Tuple[float, float]
    uncertainty_margin: float
    surrogate_model: str
    within_dcs_tolerance: bool
    thermodynamics: Optional[ThermodynamicSanityMetrics] = None
    execution_time_ms: float
    timestamp: str


class OptimizationResponse(BaseModel):
    """Closed-loop recommended setpoint adjustments and energy reduction metrics."""
    plant_id: str
    baseline_sec_gcal_per_mt: float
    optimized_sec_gcal_per_mt: float
    expected_delta_sec: float
    percent_reduction: float
    current_setpoints: Dict[str, float]
    recommended_setpoints: Dict[str, float]
    setpoint_adjustments: Dict[str, float]
    active_constraints: List[str]
    annual_savings_inr_crore: float
    annual_co2_abated_tonnes: float
    thermodynamics: Optional[ThermodynamicSanityMetrics] = None
    execution_time_ms: float
    timestamp: str


class AnomalyResponse(BaseModel):
    """Incident diagnostic response with failure-mode classification and SHAP breakdown."""
    plant_id: str
    actual_sec: float
    expected_sec: float
    z_score: float
    is_anomaly: bool
    severity: Literal["NORMAL", "MILD_EXCURSION_2SIGMA", "SEVERE_TRIP_3SIGMA"]
    isa18_2_alarm_priority: Literal["PRIORITY_1_CRITICAL_TRIP", "PRIORITY_2_HIGH_EXCURSION", "PRIORITY_3_OPERATOR_ADVISORY", "NORMAL"]
    failure_mode: Optional[str]
    top_contributing_factors: List[Dict[str, Any]]
    operator_action_advisory: str
    execution_time_ms: float
    timestamp: str


class DriftStatusResponse(BaseModel):
    """Live telemetry concept and covariate drift diagnostic status."""
    monitoring_timestamp: str
    overall_severity: str
    drifted_tags_count: int
    monitored_tags_count: int
    drifted_tags: List[str]
    recommended_dcs_action: str
    significance_level: float


class HealthResponse(BaseModel):
    """Service health and industrial telemetry status."""
    status: Literal["HEALTHY", "DEGRADED", "OFFLINE"]
    service_name: str
    version: str
    facility: str
    surrogate_model_loaded: str
    surrogate_model_r2: float
    surrogate_model_rmse: float
    fleet_units: List[str]
    uptime_seconds: float
    total_inferences_served: int


class FleetPredictionResponse(BaseModel):
    """Fleet-wide Specific Energy Consumption evaluation summary."""
    timestamp: str
    fleet_records_evaluated: int
    predictions: List[PredictionResponse]
    fleet_weighted_sec_gcal_per_mt: float
    total_fuel_energy_rate_gcal_per_hr: float
    fleet_status: Literal["NOMINAL", "ADVISORY", "EXCURSION"]
    execution_time_ms: float


class GovernorLimitsResponse(BaseModel):
    """DCS safety envelopes, slew-rate ramp governors, and design baselines."""
    facility: str
    safety_bounds: Dict[str, Dict[str, Any]]
    slew_rate_limits: Dict[str, Dict[str, Any]]
    plant_units: Dict[str, Dict[str, Any]]


class SetpointCommitRequest(BaseModel):
    """DCS Supervisory Control handshake request committing recommended setpoints to DCS."""
    plant_id: Literal["gadepan_1", "gadepan_2", "gadepan_3"]
    target_setpoints: Dict[str, float] = Field(
        ...,
        description="Target setpoints to transfer to DCS PID controllers"
    )
    operator_badge_id: str = Field(
        ...,
        min_length=3,
        description="CFCL Operator badge identification string",
        example="CFCL-OP-4491"
    )
    shift_incharge_token: str = Field(
        ...,
        min_length=4,
        description="Shift In-Charge authorization approval token",
        example="SIC-G3-AUTH-982"
    )
    control_mode: Literal["ADVISORY_ACCEPTED", "SUPERVISORY_AUTO_RAMP"] = Field(
        "SUPERVISORY_AUTO_RAMP",
        description="Supervisory execution mode"
    )


class SetpointCommitResponse(BaseModel):
    """DCS Supervisory Control confirmation handshake response with full audit trail."""
    transaction_id: str
    plant_id: str
    status: Literal["COMMITTED_TO_DCS", "INTERLOCK_ABORTED"]
    safety_sil_verification: str
    setpoints_transferred: Dict[str, float]
    slew_ramp_duration_minutes: int
    timestamp: str
    dcs_audit_log_id: str


class DCSWatchdogResponse(BaseModel):
    """DCS Supervisory communication watchdog and fail-safe fallback readiness."""
    service_status: Literal["ONLINE_HEALTHY", "WATCHDOG_TIMEOUT", "FAILSAFE_BYPASS"]
    watchdog_interval_seconds: float
    heartbeat_latency_ms: float
    failsafe_fallback_setpoints: Dict[str, Dict[str, float]]
    timestamp: str


# =========================================================================
# Advanced First-Principles & Functional Safety Schemas (Layers 1 - 7)
# =========================================================================

class PhysicsValidationRequest(BaseModel):
    """Request to validate thermodynamic equilibrium and metallurgical tube stress."""
    reformer_outlet_temp: float = Field(..., ge=700.0, le=850.0)
    steam_carbon_ratio: float = Field(..., ge=2.5, le=3.6)
    slew_rate_c_per_hr: float = Field(1.5, ge=0.0, le=5.0)
    operating_pressure_bar: float = Field(32.5, ge=20.0, le=45.0)
    observed_ch4_slip_pct: Optional[float] = Field(7.2, ge=1.0, le=15.0)


class PhysicsValidationResponse(BaseModel):
    """Detailed chemical kinetics and tube metallurgical creep stress evaluation."""
    timestamp: str
    reformer_kinetics: Dict[str, Any]
    tube_metallurgy: Dict[str, Any]
    larson_miller_creep: Dict[str, Any]
    is_fully_compliant: bool
    summary_verdict: str


class ConformalPredictionResponse(BaseModel):
    """Finite-sample distribution-free coverage guarantee intervals."""
    plant_id: str
    point_prediction_sec: float
    conformal_lower_bound_sec: float
    conformal_upper_bound_sec: float
    conformal_interval_half_width: float
    coverage_guarantee_percent: float
    is_distribution_free: bool
    is_tight_interval: bool
    epistemic_uncertainty_flag: bool
    timestamp: str


class OODAssessmentResponse(BaseModel):
    """Multivariate Mahalanobis distance & Free Energy OOD assessment."""
    plant_id: str
    mahalanobis_distance_squared: float
    chi2_critical_threshold: float
    ood_probability: float
    free_energy_score: float
    is_in_distribution: bool
    status: str
    dominant_deviating_features: List[str]
    timestamp: str


class ParetoOptimizationResponse(BaseModel):
    """Multi-objective Pareto frontier options (Energy vs Tube Life vs Capacity)."""
    plant_id: str
    current_sec_gcal_per_mt: float
    pareto_solutions: List[Dict[str, Any]]
    recommended_mode: str
    timestamp: str


class NMPCPlanResponse(BaseModel):
    """6-hour dynamic receding horizon trajectory incorporating ambient weather."""
    plant_id: str
    current_sec_gcal_per_mt: float
    horizon_hours: int
    immediate_setpoints: Dict[str, float]
    planned_trajectory: List[Dict[str, Any]]
    average_projected_sec: float
    projected_6h_energy_reduction_gcal_per_mt: float
    solver_converged: bool
    timestamp: str


class CounterfactualRequest(BaseModel):
    """Request for minimum-effort actionable operator adjustments."""
    plant_id: Literal["gadepan_1", "gadepan_2", "gadepan_3"]
    telemetry: PlantTelemetryInput
    target_reduction_gcal_per_mt: float = Field(0.030, ge=0.005, le=0.150)


class CounterfactualResponse(BaseModel):
    """Actionable minimum-effort operator recourse recommendations."""
    current_sec_gcal_per_mt: float
    target_sec_gcal_per_mt: float
    achievable_sec_gcal_per_mt: float
    energy_reduction_gcal_per_mt: float
    estimated_daily_savings_inr: float
    total_recourse_effort_score: float
    action_plan: List[Dict[str, Any]]
    operator_executive_summary: str
    timestamp: str


class OPCUAManifestResponse(BaseModel):
    """IEC 62541 industrial OPC-UA address space tag dictionary."""
    protocol: str
    endpoint_url: str
    namespace: str
    tags_count: int
    tags: List[Dict[str, Any]]


class ModbusMapResponse(BaseModel):
    """Modbus TCP 40000-series holding register table."""
    protocol: str
    port: int
    base_address: int
    register_count: int
    registers: List[Dict[str, Any]]


class MonteCarloRiskRequest(BaseModel):
    """Request for 10,000-trial stochastic risk simulation."""
    mean_delta_sec_gcal_mt: float = Field(0.038, ge=0.005, le=0.100)
    trials: int = Field(10000, ge=1000, le=50000)


class MonteCarloRiskResponse(BaseModel):
    """Stochastic financial Value-at-Risk (VaR99 / CVaR99) report."""
    trials: int
    mean_annual_savings_inr_cr: float
    median_annual_savings_inr_cr: float
    std_annual_savings_inr_cr: float
    var_95_inr_cr: float
    var_99_inr_cr: float
    cvar_99_inr_cr: float
    percentile_5th_inr_cr: float
    percentile_95th_inr_cr: float
    gas_tariff_simulated_mean_usd: float
    exchange_rate_simulated_mean_inr: float
    downside_tail_risk_probability: float
    financial_resilience_rating: str
    timestamp: str


class CanaryStatusResponse(BaseModel):
    """Continuous shadow evaluation and Canary gating review."""
    champion_model_name: str
    challenger_model_name: str
    shadow_evaluation_hours: int
    champion_rmse_gcal_mt: float
    challenger_rmse_gcal_mt: float
    rmse_improvement_percent: float
    champion_thermodynamic_compliance_pct: float
    challenger_thermodynamic_compliance_pct: float
    canary_gate_passed: bool
    deployment_verdict: str
    action_executed: str
    timestamp: str

