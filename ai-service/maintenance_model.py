import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Force CPU usage as requested
device = torch.device("cpu")
print(f"Using device: {device}")

CONFIG = {
    'window_size': 50,
    'hidden_size': 256,  # Increased from 128
    'num_layers': 3,     # Increased from 2
    'dropout': 0.4,      # Increased from 0.3
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'patience': 10,
    'features': ['Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 'SoC', 'Charge_Cycles'],
    'targets': ['RUL', 'Failure_Probability', 'Component_Health_Score'],
    'model_path': 'battery_model.pth'
}

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None, None, None, None

    df = pd.read_csv(filepath)
    
    # Handle missing battery_id
    if 'battery_id' not in df.columns:
        print("  'battery_id' column not found. Assuming single continuous battery timeline.")
        print("  Generating synthetic battery_id based on timestamp gaps (if any) or treating as ID=1.")
        # Check for timestamp if needed, but for now assign all to 1
        df['battery_id'] = 1
        # Optional: Split by large gaps if timestamp exists
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp')
    
    # Fill missing values if any
    df = df.ffill().bfill()

    # Features and Targets
    X = df[CONFIG['features']].values
    y = df[CONFIG['targets']].values

    # Scale Features
    print("  Scaling features...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # We don't scale targets for the final output metrics, but for training stability it helps.
    # However, for this specific request, we'll train directly on targets but use appropriate loss weights/scales.
    # Actually, RUL can be large (0-100+), Health (0-100). 
    # To ensure convergence, we will scale targets as well and inverse_transform for evaluation.
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    return df, X_scaled, y_scaled, scaler_X, scaler_y

class BatteryDataset(Dataset):
    def __init__(self, X, y, battery_ids, window_size):
        self.X_seq = []
        self.y_seq = []
        
        # Generate sliding windows strictly within each battery_id group
        unique_ids = np.unique(battery_ids)
        for bid in unique_ids:
            # Get indices for this battery
            indices = np.where(battery_ids == bid)[0]
            if len(indices) <= window_size:
                continue
                
            # Extract data for this battery
            X_bat = X[indices]
            y_bat = y[indices]
            
            # Create sequences
            # We predict the target at the END of the window
            for i in range(len(X_bat) - window_size):
                self.X_seq.append(X_bat[i : i + window_size])
                self.y_seq.append(y_bat[i + window_size])
                
        self.X_seq = torch.FloatTensor(np.array(self.X_seq))
        self.y_seq = torch.FloatTensor(np.array(self.y_seq))
        
        print(f"  Generated {len(self.X_seq)} sequences from {len(unique_ids)} batteries.")

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiTaskLSTM, self).__init__()
        
        # Shared Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Heads
        # 1. RUL (Regression) - Remaining Useful Life
        self.head_rul = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 2. Failure Probability (Binary Classification / Probability)
        # Using logits for BCEWithLogitsLoss
        self.head_fail = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
        # 3. Component Health Score (Regression 0-100)
        self.head_health = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # LSTM output: (batch, seq, hidden)
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output for prediction
        last_step = lstm_out[:, -1, :]
        
        rul_out = self.head_rul(last_step)
        fail_out = self.head_fail(last_step)
        health_out = self.head_health(last_step)
        
        return rul_out, fail_out, health_out

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def train_model(model, train_loader, val_loader, scaler_y):
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    early_stop_count = 0
    
    # Targets indices in y array: 0:RUL, 1:Failure_Probability, 2:Health_Score
    # NOTE: Since we scaled targets, Failure_Prob is also scaled. 
    # For BCE loss, we need binary/probability targets (0-1). 
    # We CANNOT use StandardScaler on binary/prob targets easily if we want to use BCE.
    # CORRECTION: We should NOT scale Failure_Probability with StandardScaler if we treat it as binary/prob.
    # However, the code in main block scaled ALL targets. 
    # Fix: We will unscale the Failure target inside the loss calculation or handle it separately.
    # A better approach is to NOT scale Failure Probability column in preprocessing.
    # Let's adjust preprocessing logic conceptually here: 
    # We will assume scaler_y handles the scaling, and we accept it might shift 0/1 to floats.
    # Actually, BCEWithLogitsLoss expects targets in [0, 1]. StandardScaler shifts them.
    # This is a critical nuance. I will disable scaling for Failure Probability in the main flow logic below.
    
    print("\nStarting training...")
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        total_loss = 0
        loss_rul_sum = 0
        loss_fail_sum = 0
        loss_health_sum = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            pred_rul, pred_fail, pred_health = model(X_batch)
            
            # Loss Calculation with Weighted Sum
            # 1. RUL Loss (MSE)
            loss_rul = criterion_mse(pred_rul, y_batch[:, 0:1])
            
            # 2. Failure Loss (BCE) - Targets are 0/1 (not scaled)
            loss_fail = criterion_bce(pred_fail, y_batch[:, 1:2])
            
            # 3. Health Loss (MSE)
            loss_health = criterion_mse(pred_health, y_batch[:, 2:3])
            
            # Weighted Sum: Prioritize failure prediction (5x), RUL (2x), Health (1x)
            loss = 2.0 * loss_rul + 5.0 * loss_fail + 1.0 * loss_health
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loss_rul_sum += loss_rul.item()
            loss_fail_sum += loss_fail.item()
            loss_health_sum += loss_health.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                p_rul, p_fail, p_health = model(X_val)
                l_rul = criterion_mse(p_rul, y_val[:, 0:1])
                l_fail = criterion_bce(p_fail, y_val[:, 1:2])
                l_health = criterion_mse(p_health, y_val[:, 2:3])
                val_loss += (2.0 * l_rul + 5.0 * l_fail + 1.0 * l_health).item()
        
        val_loss /= len(val_loader)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Train Loss: {avg_loss:.4f} (RUL:{loss_rul_sum/len(train_loader):.2f}, F:{loss_fail_sum/len(train_loader):.2f}, H:{loss_health_sum/len(train_loader):.2f}) | Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), CONFIG['model_path'])
        else:
            early_stop_count += 1
            if early_stop_count >= CONFIG['patience']:
                print("Early stopping triggered.")
                break
                
    print("Training complete.")

def evaluate_model(model, test_loader, scaler_y):
    print("\nEvaluating model...")
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()
    
    all_preds_rul = []
    all_preds_fail = []
    all_preds_health = []
    all_targets_rul = []
    all_targets_fail = []
    all_targets_health = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            # Predictions
            p_rul, p_fail, p_health = model(X_batch)
            
            all_preds_rul.extend(p_rul.cpu().numpy())
            # For failure, apply sigmoid to get probability
            all_preds_fail.extend(torch.sigmoid(p_fail).cpu().numpy())
            all_preds_health.extend(p_health.cpu().numpy())
            
            all_targets_rul.extend(y_batch[:, 0:1].numpy())
            all_targets_fail.extend(y_batch[:, 1:2].numpy())
            all_targets_health.extend(y_batch[:, 2:3].numpy())
            
    # Inverse Transform
    # We need to construct a matrix to use inverse_transform
    # Construct combined arrays
    # Note: We handled 'Failure' separately (not scaled), so we only unscale RUL and Health.
    # To make it simple: We just unscale RUL and Health manually using the scaler parameters if possible, 
    # OR we follow the specialized scaler approach.
    
    # Simplified approach for reporting:
    preds_rul = np.array(all_preds_rul)
    preds_health = np.array(all_preds_health)
    targets_rul = np.array(all_targets_rul)
    targets_health = np.array(all_targets_health)
    
    # We need to undo scaling for RUL (index 0) and Health (index 2)
    # The scaler was fitted on 3 columns. We can verify column indices: 0, 1, 2.
    # But we skipped scaling for col 1 (Failure) in the modified logic.
    # Let's assume manual unscaling for correctness or use a dummy array.
    
    # Create Full dummy array for inverse transform
    def unscale(col_idx, data):
        # Create a shape (N, 3) with zeros
        dummy = np.zeros((len(data), 3))
        dummy[:, col_idx] = data.flatten()
        return scaler_y.inverse_transform(dummy)[:, col_idx]

    # Metrics
    # RUL: RMSE
    # Important: The targets in the dataset class were SCALED. We compare Scaled Preds vs Scaled Targets? 
    # No, usually we want reporting in real units.
    # Let's unscale.
    
    try:
        # Scale back to original units using the scaler provided
        # NOTE: This assumes the scaler used ALL 3 columns including Failure. 
        # If we skip Failure scaling, we must adjust scaler usage.
        # Decision: I will modify the scaler logic in `load_and_preprocess_data` to only scale RUL and Health.
        
        # Calculate Metrics (on Scaled values first, or Real if possible)
        # For simplicity in this script, let's report scaled metrics or try to recover real.
        
        # recovering real values roughly:
        real_preds_rul = unscale(0, preds_rul)
        real_targets_rul = unscale(0, targets_rul)
        
        real_preds_health = unscale(2, preds_health)
        real_targets_health = unscale(2, targets_health)
        
        rmse_rul = np.sqrt(mean_squared_error(real_targets_rul, real_preds_rul))
        mae_health = mean_absolute_error(real_targets_health, real_preds_health)
        
        # Failure Probability (is already 0-1, and unscaled targets are 0-1)
        # Targets for AUC must be binary. If Failure_Probability is float (0.0 - 1.0), we can threshold or use as is?
        # Typically AUC uses probabilities.
        targets_fail = np.array(all_targets_fail)
        preds_fail = np.array(all_preds_fail)
        
        # Binarize targets for AUC if they are probabilities? Or assume dataset has 0/1?
        # csv says "Failure_Probability", likely 0 or 1 or float. 
        # If it's probability, we treat it as regression or classification? 
        # User requested BCEWithLogitsLoss, implying binary Classification or Probabilities.
        # AUC works on scores.
        # Check if targets are binary
        unique_targets = np.unique(targets_fail)
        if len(unique_targets) > 2:
             # It's a regression of probability? Then AUC might not be appropriate directly without thresholding.
             # But usually Failure Prob is the LABEL (0 or 1).
             pass
        
        try:
            roc_auc = roc_auc_score(targets_fail > 0.5, preds_fail)
        except:
            roc_auc = 0.0 # Handle edge case if only 1 class
            
        print(f"\nFinal Evaluation Results:")
        print(f"  RUL RMSE: {rmse_rul:.4f} (Original Units)")
        print(f"  Failure Probability ROC-AUC: {roc_auc:.4f}")
        print(f"  Health Score MAE: {mae_health:.4f} (Original Units)")
        
    except Exception as e:
        print(f"Metric calculation warning: {e}")

# ==========================================
# 5. INFERENCE
# ==========================================
def predict_battery_status(voltage, current, temp, soc, cycles):
    # Load model and scalers
    # Note: Requires saved scalers. For this script we assume running in same session or we'd load pickles.
    # Here we just show the structure using global scaler objects if available.
    pass # Implementation inside main for continuity

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    data_path = 'EV_Predictive_Maintenance_Dataset_15min.csv'
    
    # 1. Load Data
    # CUSTOM SCALING LOGIC INSERTION
    # We will redefine load_and_preprocess slightly to handle the Failure column selectively
    
    # Reload logic wrapper
    print("--- 1. Loading & Preprocessing ---")
    df = pd.read_csv(data_path)
    
    # Create synthetic battery IDs by splitting timeline into chunks
    # This simulates multiple batteries and provides pattern diversity
    chunk_size = 10000  # ~1 week of 15-min intervals
    if 'battery_id' not in df.columns:
        df['battery_id'] = df.index // chunk_size
        print(f"  Created {df['battery_id'].nunique()} synthetic battery IDs from timeline chunks")
    
    # Fill NA
    df = df.ffill().bfill()
    
    X_raw = df[CONFIG['features']].values
    y_raw = df[CONFIG['targets']].values # RUL, Failure, Health
    
    # Scale Features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    # Scale Targets - CRITICAL FIX: Don't scale Failure Probability (binary 0/1)
    y_scaled = np.copy(y_raw).astype(float)
    
    # RUL (col 0) - Scale
    rul_mean = np.mean(y_raw[:, 0])
    rul_std = np.std(y_raw[:, 0])
    y_scaled[:, 0] = (y_raw[:, 0] - rul_mean) / (rul_std + 1e-6)
    
    # Failure Probability (col 1) - DO NOT SCALE, keep as 0/1 for BCE loss
    # y_scaled[:, 1] remains as y_raw[:, 1]
    
    # Health (col 2) - Scale
    health_mean = np.mean(y_raw[:, 2])
    health_std = np.std(y_raw[:, 2])
    y_scaled[:, 2] = (y_raw[:, 2] - health_mean) / (health_std + 1e-6)
    
    # Mock scaler for consistent interface in inverse transform later
    # We'll attach means/stds to a dummy object or just use variables
    class CustomScaler:
        def __init__(self, means, stds):
            self.means = means
            self.stds = stds
        def inverse_transform(self, data):
            # data is (N, 3)
            res = np.copy(data)
            res[:, 0] = data[:, 0] * self.stds[0] + self.means[0]
            res[:, 2] = data[:, 2] * self.stds[2] + self.means[2]
            return res
            
    scaler_y_custom = CustomScaler([rul_mean, 0, health_mean], [rul_std, 1, health_std])
    
    # 2. Split Data (By Battery ID, or Time Split if ID=1)
    # Since we likely defined ID=1 for all, we split by time (80/20)
    split_idx = int(len(df) * 0.8)
    train_ids = df['battery_id'].values[:split_idx] # Actually this logic fails if ID is same.
    
    # Correct Group Split Logic:
    pool_ids = df['battery_id'].unique()
    if len(pool_ids) > 1:
        # Proper battery split
        train_id_list, test_id_list = pool_ids[:int(len(pool_ids)*0.8)], pool_ids[int(len(pool_ids)*0.8):]
        # Mask
        train_mask = df['battery_id'].isin(train_id_list)
        test_mask = df['battery_id'].isin(test_id_list)
        
        X_train, y_train = X_scaled[train_mask], y_scaled[train_mask]
        X_test, y_test = X_scaled[test_mask], y_scaled[test_mask]
        
        train_ids_seq = df['battery_id'][train_mask].values
        test_ids_seq = df['battery_id'][test_mask].values
    else:
        # Time split
        print("  Single battery detected. Splitting by time (First 80% Train, Last 20% Test).")
        X_train, y_train = X_scaled[:split_idx], y_scaled[:split_idx]
        X_test, y_test = X_scaled[split_idx:], y_scaled[split_idx:]
        train_ids_seq = df['battery_id'].values[:split_idx]
        test_ids_seq = df['battery_id'].values[split_idx:]

    # 3. Create Datasets
    train_dataset = BatteryDataset(X_train, y_train, train_ids_seq, CONFIG['window_size'])
    test_dataset = BatteryDataset(X_test, y_test, test_ids_seq, CONFIG['window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 4. Initialize Model
    print("--- 2. Initializing Model ---")
    model = MultiTaskLSTM(input_dim=len(CONFIG['features']), 
                          hidden_dim=CONFIG['hidden_size'], 
                          num_layers=CONFIG['num_layers'], 
                          dropout=CONFIG['dropout']).to(device)
    
    # 5. Train
    train_model(model, train_loader, test_loader, scaler_y_custom)
    
    # 6. Evaluate
    evaluate_model(model, test_loader, scaler_y_custom)
    
    # 7. Inference Demo
    print("\n--- 3. Inference Demo ---")
    def run_inference():
        # Fake input sequence (random)
        sample_seq = X_test[0:CONFIG['window_size']] # Take first test sequence
        sample_tensor = torch.FloatTensor(sample_seq).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            r, f, h = model(sample_tensor)
            
        # Inverse transform
        # r, h are scaled
        real_r = r.item() * rul_std + rul_mean
        real_h = h.item() * health_std + health_mean
        prob_f = torch.sigmoid(f).item()
        
        print(f"Input Sequence Shape: {sample_tensor.shape}")
        print(f"Predicted RUL: {real_r:.2f} cycles")
        print(f"Predicted Failure Prob: {prob_f:.2%}")
        print(f"Predicted Health Score: {real_h:.2f}/100")
        
    run_inference()

# ==========================================
# EXPLANATION
# ==========================================
"""
WHY LSTM?
---------
LSTM (Long Short-Term Memory) is ideal for battery degradation modeling because battery health 
is a path-dependent process. The current state (SoH, RUL) depends not just on current sensor 
readings (Voltage, Temp) but on the history of usage (Charge Cycles, thermal stress over time).
LSTMs capture these long-term temporal dependencies and trends (like gradual capacity fade) 
that standard regression models (Random Forest) might miss.

WHY MULTI-TASK LEARNING?
------------------------
Predicting RUL, Failure Probability, and Health Score simultaneously improves generalization because:
1. **Shared Representations:** The model learns a robust common feature extractor (Encoder) that 
   understands the underlying physics of degradation, rather than overfitting to one noisy target.
2. **Regularization:** Learning multiple related tasks acts as a regularizer, preventing the model 
   from latching onto spurious correlations for a single task.
3. **Efficiency:** A single forward pass provides comprehensive diagnostics.
"""
