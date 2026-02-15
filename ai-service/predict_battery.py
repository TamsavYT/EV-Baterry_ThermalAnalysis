"""
Battery Predictive Maintenance - Inference Script
Load the trained model and make predictions on new battery data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# ==========================================
# MODEL DEFINITION (Must match training)
# ==========================================
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiTaskLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Output heads
        self.head_rul = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_fail = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_health = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        
        rul_out = self.head_rul(last_step)
        fail_out = self.head_fail(last_step)
        health_out = self.head_health(last_step)
        
        return rul_out, fail_out, health_out

# ==========================================
# LOAD MODEL & SCALERS
# ==========================================
def load_model():
    """Load the trained model"""
    device = torch.device("cpu")
    
    # Model config (must match training)
    model = MultiTaskLSTM(
        input_dim=5,      # 5 features
        hidden_dim=128,   # Match your training config
        num_layers=2,     # Match your training config
        dropout=0.3
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('battery_model.pth', map_location=device))
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, device

# ==========================================
# PREPARE INPUT DATA
# ==========================================
def prepare_input(battery_data, scaler_X=None):
    """
    Prepare battery sensor data for prediction
    
    Args:
        battery_data: List of dicts or DataFrame with columns:
                     ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC', 'Charge_Cycles']
                     Must have at least 30 rows (window_size)
        scaler_X: Optional pre-fitted scaler (if None, will fit on input data)
    
    Returns:
        Scaled tensor ready for model input
    """
    # Convert to DataFrame if needed
    if isinstance(battery_data, list):
        df = pd.DataFrame(battery_data)
    else:
        df = battery_data
    
    # Extract features in correct order
    features = ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC', 'Charge_Cycles']
    X = df[features].values
    
    # Scale features
    if scaler_X is None:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = scaler_X.transform(X)
    
    # Take last 30 timesteps (window_size)
    window_size = 30
    if len(X_scaled) < window_size:
        raise ValueError(f"Need at least {window_size} timesteps, got {len(X_scaled)}")
    
    X_window = X_scaled[-window_size:]
    
    # Convert to tensor: (1, 30, 5)
    X_tensor = torch.FloatTensor(X_window).unsqueeze(0)
    
    return X_tensor, scaler_X

# ==========================================
# MAKE PREDICTION
# ==========================================
def predict(model, device, input_tensor, rul_mean=None, rul_std=None, health_mean=None, health_std=None):
    """
    Make prediction using the trained model
    
    Args:
        model: Trained PyTorch model
        device: torch device
        input_tensor: Prepared input (1, 30, 5)
        rul_mean, rul_std: For unscaling RUL (if None, returns scaled)
        health_mean, health_std: For unscaling Health (if None, returns scaled)
    
    Returns:
        Dictionary with predictions
    """
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        rul_pred, fail_pred, health_pred = model(input_tensor)
    
    # Extract values
    rul_scaled = rul_pred.item()
    fail_prob = torch.sigmoid(fail_pred).item()  # Convert logits to probability
    health_scaled = health_pred.item()
    
    # Unscale if parameters provided
    if rul_mean is not None and rul_std is not None:
        rul = rul_scaled * rul_std + rul_mean
    else:
        rul = rul_scaled
    
    if health_mean is not None and health_std is not None:
        health = health_scaled * health_std + health_mean
    else:
        health = health_scaled
    
    return {
        'RUL': rul,
        'Failure_Probability': fail_prob,
        'Health_Score': health,
        'Risk_Level': 'HIGH' if fail_prob > 0.5 else 'MEDIUM' if fail_prob > 0.2 else 'LOW'
    }

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("EV Battery Predictive Maintenance - Inference")
    print("=" * 60)
    
    # 1. Load model
    model, device = load_model()
    
    # 2. Example: Create sample input data (30 timesteps)
    # In real use, this would come from your sensor readings
    sample_data = []
    for i in range(30):
        sample_data.append({
            'Battery_Voltage': 350 + np.random.randn() * 5,
            'Battery_Current': -25 + np.random.randn() * 3,
            'Battery_Temperature': 32 + np.random.randn() * 2,
            'SoC': 0.85 + np.random.randn() * 0.05,
            'Charge_Cycles': 245
        })
    
    print(f"\n📊 Input: {len(sample_data)} timesteps of battery sensor data")
    
    # 3. Prepare input
    input_tensor, scaler = prepare_input(sample_data)
    print(f"✅ Input prepared: shape {input_tensor.shape}")
    
    # 4. Make prediction
    # NOTE: For proper unscaling, you need the mean/std from training
    # For now, we'll use scaled outputs
    result = predict(model, device, input_tensor)
    
    # 5. Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"🔋 Remaining Useful Life (RUL): {result['RUL']:.2f} cycles")
    print(f"⚠️  Failure Probability: {result['Failure_Probability']:.2%}")
    print(f"💚 Health Score: {result['Health_Score']:.2f}/100")
    print(f"🚨 Risk Level: {result['Risk_Level']}")
    print("=" * 60)
    
    # ==========================================
    # EXAMPLE 2: Using Real CSV Data
    # ==========================================
    print("\n\n📁 Example 2: Loading from CSV file...")
    
    try:
        # Load last 30 rows from your dataset
        df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv')
        recent_data = df.tail(30)
        
        input_tensor, scaler = prepare_input(recent_data)
        result = predict(model, device, input_tensor)
        
        print("\n" + "=" * 60)
        print("PREDICTION FROM CSV DATA")
        print("=" * 60)
        print(f"🔋 RUL: {result['RUL']:.2f} cycles")
        print(f"⚠️  Failure Prob: {result['Failure_Probability']:.2%}")
        print(f"💚 Health: {result['Health_Score']:.2f}/100")
        print(f"🚨 Risk: {result['Risk_Level']}")
        print("=" * 60)
        
    except FileNotFoundError:
        print("⚠️  CSV file not found. Using sample data only.")

    # ==========================================
    # EXAMPLE 3: Real-time Prediction Function
    # ==========================================
    print("\n\n" + "=" * 60)
    print("USAGE IN YOUR APPLICATION")
    print("=" * 60)
    print("""
# Load model once at startup
model, device = load_model()

# For each prediction request:
battery_readings = [
    {'Battery_Voltage': 355.2, 'Battery_Current': -28.1, ...},  # t-29
    {'Battery_Voltage': 354.8, 'Battery_Current': -27.9, ...},  # t-28
    ...
    {'Battery_Voltage': 356.1, 'Battery_Current': -29.2, ...}   # t-0 (now)
]

input_tensor, _ = prepare_input(battery_readings)
prediction = predict(model, device, input_tensor)

# Use prediction
if prediction['Failure_Probability'] > 0.5:
    send_alert("High failure risk detected!")
    """)
