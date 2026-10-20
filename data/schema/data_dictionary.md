# Data Dictionary — CFCL SEC Optimization Engine

> All variables use generic industry-standard terminology for gas-based urea plants.
> Ranges are calibrated against publicly available CFCL Annual Report, BRSR disclosures,
> and standard Indian fertilizer industry operating parameters.

## Target Variable

| Variable | Column Name | Type | Unit | Range | Description |
|----------|-------------|------|------|-------|-------------|
| Specific Energy Consumption | `sec_gcal_per_mt` | float64 | Gcal/MT urea | 4.5 – 6.0 | Total energy consumed per metric ton of urea produced. Primary KPI for plant energy efficiency. National average ~5.5; CFCL benchmark ~5.0. |

## Process Variables (Model Inputs)

| # | Variable | Column Name | Type | Unit | Safety Range | Optimal Range | Controllable | Source |
|---|----------|-------------|------|------|-------------|---------------|-------------|--------|
| 1 | Steam-to-Carbon Ratio | `steam_carbon_ratio` | float64 | mol/mol | 2.8 – 3.4 | 3.0 – 3.2 | ✅ Yes | Operator setpoint |
| 2 | Synthesis Loop Pressure | `loop_pressure` | float64 | kg/cm² | 145 – 185 | 155 – 170 | ✅ Yes | Operator setpoint |
| 3 | Reformer Outlet Temperature | `reformer_outlet_temp` | float64 | °C | 780 – 830 | 800 – 815 | ✅ Yes | Operator setpoint |
| 4 | Cooling Water Inlet Temperature | `cw_inlet_temp` | float64 | °C | 24 – 40 | 28 – 32 | ❌ No | Cooling tower / ambient |
| 5 | Ambient Temperature | `ambient_temp` | float64 | °C | 12 – 48 | — | ❌ No | Weather (Kota IMD) |
| 6 | Ambient Relative Humidity | `ambient_humidity` | float64 | % | 10 – 100 | — | ❌ No | Weather (monsoon) |
| 7 | CO₂ Slip | `co2_slip_ppm` | float64 | ppm | 50 – 2,500 | Technology-dependent | ❌ No | CO₂ removal system |
| 8 | Natural Gas Calorific Value | `ng_calorific_value` | float64 | kcal/SCM | 8,400 – 9,200 | 8,600 – 9,000 | ❌ No | HBJ pipeline |
| 9 | Catalyst Age | `catalyst_age_days` | int64 | days | 0 – 2,500 | 0 – 90 (fresh) | ❌ No | Turnaround schedule |

## Categorical Variables

| Variable | Column Name | Type | Valid Values | Description |
|----------|-------------|------|-------------|-------------|
| Plant ID | `plant_id` | category | `gadepan_1`, `gadepan_2`, `gadepan_3` | CFCL Gadepan complex plant identifier |
| Operator Shift | `shift` | category | `morning`, `afternoon`, `night` | 8-hour shift classification |

## Metadata Columns

| Variable | Column Name | Type | Description |
|----------|-------------|------|-------------|
| Timestamp | `timestamp` | datetime64 | Hourly DCS historian snapshot timestamp |

## Shift Schedule

| Shift | Hours | Notes |
|-------|-------|-------|
| Morning (A-shift) | 06:00 – 14:00 | Standard daytime operations |
| Afternoon (B-shift) | 14:00 – 22:00 | Peak heat hours (14:00–16:00) drive SEC up |
| Night (C-shift) | 22:00 – 06:00 | Conservative operator behavior → 2–3% higher SEC |

## Plant-Specific Parameters

| Parameter | Gadepan-I | Gadepan-II | Gadepan-III |
|-----------|-----------|------------|-------------|
| Commissioned | 1993 | 1999 | 2019 |
| Capacity (MTPD) | 1,520 | 1,520 | 2,350 |
| CO₂ Removal | MEA | Benfield | aMDEA |
| SEC Range (Gcal/MT) | 5.08 – 5.25 | 5.05 – 5.22 | 4.88 – 4.97 |
| CO₂ Slip Range (ppm) | 800 – 2,000 | 600 – 1,500 | 200 – 600 |
| Typical Turnaround | Apr – May | May – Jun | Mar – Apr |

## Engineered Features (Created in feature_engine.py)

| Feature | Description | Window |
|---------|-------------|--------|
| `sec_rolling_mean_24h` | 24-hour rolling mean SEC | 24 hours |
| `sec_rolling_std_24h` | 24-hour rolling std SEC | 24 hours |
| `temp_humidity_interaction` | `ambient_temp × ambient_humidity / 100` | — |
| `sc_pressure_interaction` | `steam_carbon_ratio × loop_pressure` | — |
| `is_monsoon` | Binary: 1 if July/August, 0 otherwise | — |
| `is_turnaround_recovery` | Binary: 1 if within 30 days post-turnaround | — |
| `catalyst_decay_factor` | Exponential decay: `1 - exp(-catalyst_age_days / 180)` | — |
| `hour_of_day` | Hour extracted from timestamp (0–23) | — |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | — |
| `is_weekend` | Binary: 1 if Saturday/Sunday | — |

## Data Quality Notes

- **Missing Data**: Turnaround windows (20–30 days per plant) have NaN gaps — this is realistic
- **Sensor Accuracy**: ±0.5% instrument accuracy noise is baked into synthetic data
- **Autocorrelation**: Plant thermal inertia means changes propagate over 15–30 minutes
- **Shift Handovers**: Small SEC discontinuities at 06:00/14:00/22:00 are expected
