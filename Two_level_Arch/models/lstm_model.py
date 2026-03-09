import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Tuple


class RoutinePredictor(nn.Module):
   

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_devices: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
       
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_devices = num_devices
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected output
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_devices),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            last_hidden = torch.cat(
                (h_n[-2, :, :], h_n[-1, :, :]), dim=1
            )
        else:
            last_hidden = h_n[-1, :, :]

        # Apply dropout and FC
        out = self.dropout(last_hidden)
        logits = self.fc(out)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def predict_numpy(
        self,
        X: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Predict from numpy arrays.

        Args:
            X: Input array [N, seq_len, input_size].
            device: torch device.

        Returns:
            Probability array [N, num_devices].
        """
        if device is None:
            device = next(self.parameters()).device

        x_tensor = torch.FloatTensor(X).to(device)
        probs = self.predict(x_tensor)
        return probs.cpu().numpy()

    def save_model(self, path: Optional[str] = None) -> str:
        
        if path is None:
            from Two_level_Arch.src.config import MODEL_SAVE_DIR
            path = os.path.join(MODEL_SAVE_DIR, "lstm_routine_predictor.pt")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_devices": self.num_devices,
            "bidirectional": self.bidirectional,
        }, path)
        print(f"[INFO] LSTM model saved to {path}")
        return path

    @classmethod
    def load_model(cls, path: Optional[str] = None, device: Optional[torch.device] = None) -> "RoutinePredictor":
        """Load model from checkpoint.

        Args:
            path: File path.
            device: torch device.

        Returns:
            Loaded RoutinePredictor instance.
        """
        if path is None:
            from Two_level_Arch.src.config import MODEL_SAVE_DIR
            path = os.path.join(MODEL_SAVE_DIR, "lstm_routine_predictor.pt")

        checkpoint = torch.load(path, map_location=device or "cpu")
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            num_devices=checkpoint["num_devices"],
            bidirectional=checkpoint["bidirectional"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if device:
            model.to(device)
        print(f"[INFO] LSTM model loaded from {path}")
        return model


def train_model(
    model: RoutinePredictor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    patience: int = 7,
) -> Dict[str, list]:
    """Train the LSTM Routine Predictor.

    Args:
        model: The LSTM model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        epochs: Max number of epochs.
        device: Compute device.
        patience: Early stopping patience.

    Returns:
        Dictionary with training history (train_loss, val_loss, val_acc).
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"[INFO] Training LSTM Routine Predictor for {epochs} epochs on {device}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("=" * 70)

    for epoch in range(epochs):
        # ── Training Phase ──
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        ):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)
            train_samples += X_batch.size(0)

        avg_train_loss = train_loss_sum / train_samples

        # ── Validation Phase ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False
            ):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss_sum += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_samples += X_batch.size(0)

        avg_val_loss = val_loss_sum / val_samples
        val_acc = val_correct / val_samples

        # ── Record History ──
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        # ── Print Epoch Summary ──
        print(
            f"  Epoch {epoch+1:>3}/{epochs} │ "
            f"Train Loss: {avg_train_loss:.4f} │ "
            f"Val Loss: {avg_val_loss:.4f} │ "
            f"Val Acc: {val_acc:.4f}"
        )

        # ── Early Stopping Check ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] Restored best model (val_loss={best_val_loss:.4f})")

    print("=" * 70)
    print("[INFO] LSTM training complete.")
    return history

