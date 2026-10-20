"""PyTorch LSTM Recurrent Neural Network for Temporal SEC Modeling.

Captures sequential process dynamics, thermal inertia, and multi-hour lag effects
using a 2-layer stacked LSTM architecture with normalized multi-plant sequence modeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.config_loader import get_model_params, get_paths, get_project_root, load_config
from src.utils.logger import get_logger


class TimeseriesDataset(Dataset):
    """Sliding-window timeseries dataset for PyTorch LSTM training."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class StackedLSTM(nn.Module):
    """Stacked LSTM with Dropout and Linear Readout Head."""

    def __init__(self, input_size: int, hidden_units: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Select last timestep representation
        last_step = out[:, -1, :]
        return self.fc(last_step)


class LSTMSEC:
    """LSTM sequence model for specific energy consumption prediction."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("lstm_sec")
        self.model_cfg = load_config("model")
        self.params = get_model_params("lstm")
        
        self.sequence_length = self.params.get("sequence_length", 24)
        self.hidden_units = self.params.get("hidden_units", 64)
        self.num_layers = self.params.get("num_layers", 2)
        self.dropout = self.params.get("dropout", 0.2)
        self.batch_size = self.params.get("batch_size", 64)
        self.epochs = self.params.get("epochs", 50)
        self.learning_rate = self.params.get("learning_rate", 0.003)
        self.patience = self.params.get("patience", 15)
        
        # Core process features + plant identity + diurnal cycles
        self.feature_columns = list(FEATURE_COLUMNS) + [
            "plant_gadepan_1", "plant_gadepan_2", "plant_gadepan_3",
            "hour_sin", "hour_cos"
        ]
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.model = StackedLSTM(
            input_size=len(self.feature_columns),
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        self.metrics: Dict[str, float] = {}

    def prepare_dataset_splits(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate contiguous sequence windows split chronologically per plant."""
        # Fit scalers
        clean_df = df.dropna(subset=[TARGET_COLUMN]).copy()
        for col in self.feature_columns:
            if col not in clean_df.columns:
                clean_df[col] = 0.0
                
        self.feature_scaler.fit(clean_df[self.feature_columns].fillna(clean_df[self.feature_columns].mean()))
        self.target_scaler.fit(clean_df[[TARGET_COLUMN]])
        
        tr_sequences, tr_targets = [], []
        va_sequences, va_targets = [], []
        
        for plant_id, grp in clean_df.groupby("plant_id", sort=False):
            grp_sorted = grp.sort_values("timestamp")
            feat_vals = self.feature_scaler.transform(grp_sorted[self.feature_columns])
            tgt_vals = self.target_scaler.transform(grp_sorted[[TARGET_COLUMN]]).flatten()
            
            n = int(len(grp_sorted) * train_ratio)
            # Train window
            for i in range(n - self.sequence_length):
                tr_sequences.append(feat_vals[i : i + self.sequence_length])
                tr_targets.append(tgt_vals[i + self.sequence_length])
                
            # Validation window
            for i in range(n - self.sequence_length, len(grp_sorted) - self.sequence_length):
                va_sequences.append(feat_vals[i : i + self.sequence_length])
                va_targets.append(tgt_vals[i + self.sequence_length])
                
        return np.array(tr_sequences), np.array(tr_targets), np.array(va_sequences), np.array(va_targets)

    def fit(self, df: pd.DataFrame) -> LSTMSEC:
        """Train the LSTM model with early stopping on temporal validation split."""
        self.logger.info("Preparing sequence dataset for LSTM (window=%d hours, features=%d)",
                         self.sequence_length, len(self.feature_columns))
        
        X_tr, y_tr, X_va, y_va = self.prepare_dataset_splits(df, train_ratio=0.8)
        
        train_dataset = TimeseriesDataset(X_tr, y_tr)
        val_dataset = TimeseriesDataset(X_va, y_va)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.logger.info("Training LSTM for max %d epochs (patience=%d)...", self.epochs, self.patience)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
                
            train_loss /= len(train_dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item() * len(batch_x)
            val_loss /= len(val_dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 5 == 0 or epoch == 1:
                self.logger.info("Epoch %02d/%d | Train Loss: %.5f | Val Loss: %.5f", epoch, self.epochs, train_loss, val_loss)
                
            if patience_counter >= self.patience:
                self.logger.info("Early stopping triggered at epoch %d", epoch)
                break
                
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        # Final validation evaluation with inverse scaling
        self.model.eval()
        with torch.no_grad():
            val_preds_list = []
            for batch_x, _ in val_loader:
                preds = self.model(batch_x)
                val_preds_list.extend(preds.squeeze(-1).numpy())
                
        val_preds_scaled = np.array(val_preds_list).reshape(-1, 1)
        val_preds = self.target_scaler.inverse_transform(val_preds_scaled).flatten()
        y_true = self.target_scaler.inverse_transform(y_va.reshape(-1, 1)).flatten()
        
        self.metrics = {
            "val_r2": round(float(r2_score(y_true, val_preds)), 4),
            "val_mae": round(float(mean_absolute_error(y_true, val_preds)), 4),
            "val_rmse": round(float(np.sqrt(mean_squared_error(y_true, val_preds))), 4),
            "val_mape": round(float(mean_absolute_percentage_error(y_true, val_preds) * 100), 4),
        }
        self.logger.info("LSTM Final Validation: R²=%.4f | RMSE=%.4f | MAE=%.4f",
                         self.metrics["val_r2"], self.metrics["val_rmse"], self.metrics["val_mae"])
        return self

    def predict_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Generate predictions given 3D array of sequences [N, window, features]."""
        self.model.eval()
        with torch.no_grad():
            tensor_seq = torch.tensor(sequences, dtype=torch.float32)
            preds_scaled = self.model(tensor_seq).numpy()
            return self.target_scaler.inverse_transform(preds_scaled).flatten()

    def save(self, filepath: Optional[Path | str] = None) -> Path:
        """Persist model weights, scalers, and metadata."""
        if filepath is None:
            paths = get_paths()
            filepath = paths["models"] / "lstm_sec.pt"
        else:
            filepath = Path(filepath)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "params": self.params,
        }, filepath)
        self.logger.info("Saved PyTorch LSTM model artifact to %s", filepath)
        
        metrics_path = filepath.parent / "lstm_sec_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": self.metrics, "params": self.params}, f, indent=2)
            
        return filepath

    def load(self, filepath: Path | str) -> LSTMSEC:
        """Load model weights and feature configuration."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_scaler = checkpoint["feature_scaler"]
        self.target_scaler = checkpoint.get("target_scaler", StandardScaler())
        self.feature_columns = checkpoint["feature_columns"]
        self.metrics = checkpoint.get("metrics", {})
        return self


if __name__ == "__main__":
    from src.utils.config_loader import get_project_root
    
    processed_csv = get_project_root() / "data" / "processed" / "plant_data_processed.csv"
    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_csv}")
        
    df = pd.read_csv(processed_csv, parse_dates=["timestamp"])
    lstm = LSTMSEC()
    lstm.fit(df)
    lstm.save()
    print("\nLSTM Model Training Complete. Artifact saved successfully.")
    print("Metrics:", lstm.metrics)
