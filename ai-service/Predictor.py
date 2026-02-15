import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('nev_battery_charging.csv')

# Feature Engineering (We only use the "Raw 3" as starting points)
df['temp_rate'] = df['battery_temp'].diff().fillna(0)
df['volt_rate'] = df['terminal_voltage'].diff().fillna(0)
df['power'] = df['terminal_voltage'] * df['battery_current']

# INPUTS: Only the 3 physical sensors + their rates
features = ['battery_temp', 'terminal_voltage', 'battery_current', 'temp_rate', 'volt_rate', 'power']

# TARGETS: The AI now "senses" SOC, SOH, and Risk
targets = ['SOC', 'SOH', 'thermal_stress_index']

X_raw = df[features].values
y_raw = df[targets].values

# 2. Scaling
scaler_X = StandardScaler()
scaler_y = MinMaxScaler() # Keeps SOC (0-100), SOH (0-1), and Risk (0-1) normalized

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 3. Create Sequences (15-second window to understand the "trend")
def create_sequences(X, y, time_steps=15):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps=15)
X_train, X_test, y_train, y_test = train_test_split(
    torch.tensor(X_seq, dtype=torch.float32), 
    torch.tensor(y_seq, dtype=torch.float32), 
    test_size=0.2, random_state=42
)

# 4. The "Method 3" AI Architecture
class BatteryDeepSensingLSTM(nn.Module):
    def __init__(self):
        super(BatteryDeepSensingLSTM, self).__init__()
        # Shared LSTM layers to understand battery chemistry patterns
        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.batch_norm = nn.BatchNorm1d(128)
        
        # Branch 1: SOC Estimator (State of Charge)
        self.soc_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        # Branch 2: SOH Estimator (State of Health)
        self.soh_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        # Branch 3: Risk Predictor (Thermal Stress)
        self.risk_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x_latent = self.batch_norm(h_n[-1])
        
        soc = self.soc_head(x_latent)
        soh = self.soh_head(x_latent)
        risk = self.risk_head(x_latent)
        
        return torch.cat((soc, soh, risk), dim=1)

model = BatteryDeepSensingLSTM()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 5. Training
print("AI is learning to sense SOC, SOH, and Risk...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} Loss: {loss.item():.6f}")

# 6. The Method 3 "Analyzer" Function
def method_3_analyzer(raw_sensor_window):
    """
    Input: Array of 15 samples with 3 columns [Temp, Voltage, Current]
    Output: Estimated SOC, SOH, RUL, and Risk
    """
    temp, volt, curr = raw_sensor_window[:,0], raw_sensor_window[:,1], raw_sensor_window[:,2]
    t_rate, v_rate, pwr = np.diff(temp, prepend=temp[0]), np.diff(volt, prepend=volt[0]), volt * curr
    
    # Pack for AI
    full_features = np.column_stack((temp, volt, curr, t_rate, v_rate, pwr))
    
    model.eval()
    with torch.no_grad():
        scaled = scaler_X.transform(full_features)
        input_t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(input_t).numpy()
        
        # Decode the AI output
        final_out = scaler_y.inverse_transform(pred_scaled)[0]
        est_soc, est_soh, est_risk = final_out[0], final_out[1], final_out[2]
        
        # RUL Logic
        rul_val = max(0, (est_soh - 0.80) / (1.0 - 0.80)) * 100
        
        return {
            "Sensed SOC": f"{est_soc:.1f}%",
            "Sensed SOH": f"{est_soh:.2%}",
            "RUL": f"{rul_val:.1f}% Life left",
            "Thermal Risk": f"{est_risk:.2%}",
            "Safety Status": "CRITICAL" if est_risk > 0.7 else "SAFE"
        }

# --- TEST ---
# Simulating a battery at 3.7V, 25 degC, with low current
test_input = np.array([[33.0, 3.7, 25] for _ in range(15)])
results = method_3_analyzer(test_input)

print("\n--- AI (METHOD 3) SENSING RESULTS ---")
for k, v in results.items():
    print(f"{k}: {v}")