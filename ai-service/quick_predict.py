# Quick Prediction Example
# Run this after training to test your model

from predict_battery import load_model, prepare_input, predict
import pandas as pd

# Load model
model, device = load_model()

# Load recent data from CSV
df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv')
recent_data = df.tail(30)  # Last 30 readings

# Prepare and predict
input_tensor, _ = prepare_input(recent_data)
result = predict(model, device, input_tensor)

# Show results
print("\n🔋 Battery Health Prediction:")
print(f"   RUL: {result['RUL']:.1f} cycles")
print(f"   Failure Risk: {result['Failure_Probability']:.1%}")
print(f"   Health Score: {result['Health_Score']:.1f}/100")
print(f"   Risk Level: {result['Risk_Level']}")
