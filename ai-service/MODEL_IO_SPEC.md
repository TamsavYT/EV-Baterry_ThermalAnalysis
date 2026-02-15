# EV Battery Predictive Maintenance Model - Inputs & Outputs

## Model Overview
Multi-Task LSTM for predicting battery health, remaining useful life, and failure probability.

## INPUTS (5 Features)

The model takes a **sequence of 50 timesteps** (15-minute intervals = 12.5 hours of data):

1. **Battery_Voltage** (V)
   - Range: ~200-400V
   - Indicates charge state and cell degradation

2. **Battery_Current** (A)
   - Range: -200 to +50A (negative = charging, positive = discharging)
   - Shows load patterns and stress

3. **Battery_Temperature** (°C)
   - Range: 25-60°C
   - Critical for degradation modeling

4. **SoC** (State of Charge)
   - Range: 0.0-1.0 (0-100%)
   - Current charge level

5. **Charge_Cycles** (count)
   - Range: 100-700 cycles
   - Cumulative usage indicator

**Input Shape:** `(batch_size, 50, 5)`
- 50 timesteps (sliding window)
- 5 features per timestep

## OUTPUTS (3 Predictions)

For each input sequence, the model predicts:

### 1. RUL (Remaining Useful Life)
- **Type:** Regression
- **Range:** 0-300 cycles
- **Meaning:** How many more charge cycles before battery needs replacement
- **Example:** 216 cycles ≈ 7 months of daily charging

### 2. Failure Probability
- **Type:** Binary Classification (0-1 probability)
- **Range:** 0.0-1.0 (0-100%)
- **Meaning:** Likelihood of imminent failure/safety event
- **Threshold:** >0.5 = High risk, <0.2 = Safe
- **Example:** 0.09 = 9% failure risk (safe)

### 3. Component_Health_Score
- **Type:** Regression
- **Range:** 0-100
- **Meaning:** Overall battery health percentage
- **Interpretation:**
  - 90-100: Excellent
  - 70-90: Good
  - 50-70: Fair (consider monitoring)
  - <50: Poor (replacement recommended)
- **Example:** 85/100 = Good health

## Usage Example

```python
# Input: Last 50 readings (12.5 hours)
input_sequence = [
    [350.2, -25.3, 32.1, 0.85, 245],  # t-49
    [348.9, -24.8, 32.3, 0.84, 245],  # t-48
    ...
    [355.1, -28.1, 33.2, 0.92, 245]   # t-0 (now)
]

# Output
{
    "rul": 216.38,              # cycles remaining
    "failure_prob": 0.0979,     # 9.79% risk
    "health_score": 85.2        # out of 100
}
```

## Model Improvements (Latest Version)

✅ Increased capacity: 256 hidden units, 3 LSTM layers
✅ Proper Failure Probability handling (0/1, not scaled)
✅ Weighted loss: RUL×2 + Failure×5 + Health×1
✅ Learning rate scheduler (ReduceLROnPlateau)
✅ Synthetic battery IDs (17 batteries from timeline chunks)

**Expected Performance:**
- RUL RMSE: 40-60 cycles
- Failure ROC-AUC: 0.75-0.85
- Health MAE: 5-10 points
