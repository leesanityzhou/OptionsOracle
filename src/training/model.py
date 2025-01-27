import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple

class OptionsGPTConfig:
    """
    Simple configuration class for OptionsGPT model.
    Adjust fields as necessary for your use case.
    """
    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        n_layer: int = 3,
        n_positions: int = 256,
        n_features: int = 15,
        dropout: float = 0.25
    ):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.n_features = n_features
        self.dropout = dropout


class OptionsGPT(nn.Module):
    """
    Improved OptionsGPT model for time-series (e.g. stock/options) prediction.
    """

    def __init__(self, config: OptionsGPTConfig) -> None:
        super().__init__()
        self.config = config
        
        # ------------------------------------------------------------------
        # 1) Feature Interaction Layers (process features independently at each timestep)
        # ------------------------------------------------------------------
        self.feature_layers = nn.Sequential(
            nn.Linear(config.n_features, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout + 0.05),
            
            nn.Linear(config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout + 0.1)
        )
        
        # ------------------------------------------------------------------
        # 2) Positional Encoding
        #    Create a parameter for maximum sequence length. We'll slice it at runtime.
        # ------------------------------------------------------------------
        self.pos_encoder = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        
        # ------------------------------------------------------------------
        # 3) Temporal Transformer Blocks
        #    Each block has MultiheadAttention + a feed-forward MLP
        # ------------------------------------------------------------------
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True),
                'attn_norm': nn.LayerNorm(config.n_embd),
                'ffn': nn.Sequential(
                    nn.Linear(config.n_embd, config.n_embd * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.n_embd * 4, config.n_embd)
                ),
                'ffn_norm': nn.LayerNorm(config.n_embd)
            })
            for _ in range(config.n_layer)
        ])
        
        # ------------------------------------------------------------------
        # 4) Combine feature and temporal info
        # ------------------------------------------------------------------
        self.combine_layer = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # ------------------------------------------------------------------
        # 5) Final Prediction (price or next-day target)
        # ------------------------------------------------------------------
        self.price_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, 1)
        )

    def forward(self, 
                features: torch.Tensor, 
                time_weights: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch_size, seq_len, n_features]
            time_weights: [seq_len] or [batch_size, seq_len] (optional)
                          If provided, applies weighting before pooling.
        
        Returns:
            Dictionary with 'price_pred': [batch_size]
        """
        # ------------------------------------
        # 1) Feature interaction at each timestep
        # ------------------------------------
        batch_size, seq_len, n_features = features.shape
        assert n_features == self.config.n_features, \
            f"Expected input feature size {self.config.n_features}, got {n_features}"

        # Flatten => transform => unflatten
        x = features.reshape(batch_size * seq_len, n_features)
        x = self.feature_layers(x)
        x = x.reshape(batch_size, seq_len, -1)  # => [B, T, n_embd]
        feature_info = x  # Store feature information
        
        # ------------------------------------
        # 2) Add positional encodings
        # ------------------------------------
        if seq_len > self.config.n_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum position encoding {self.config.n_positions}."
            )
        pos_enc = self.pos_encoder[:, :seq_len, :]  # => [1, seq_len, n_embd]
        x = x + pos_enc
        
        # ------------------------------------
        # 3) Temporal attention blocks
        # ------------------------------------
        for block in self.temporal_layers:
            # Multihead attention
            attn_out, _ = block['attention'](x, x, x, need_weights=False)  # [B, T, n_embd]
            x = block['attn_norm'](x + attn_out)
            
            # Feed-forward
            ffn_out = block['ffn'](x)   # [B, T, n_embd]
            x = block['ffn_norm'](x + ffn_out)
        
        temporal_info = x  # Store temporal information
        
        # ------------------------------------
        # 4) Pooling with optional time weights
        # ------------------------------------
        if time_weights is not None:
            # If time_weights is 1D [seq_len], expand to [batch_size, seq_len]
            if time_weights.dim() == 1:
                time_weights = time_weights.unsqueeze(0).expand(batch_size, seq_len)
            elif time_weights.dim() == 2:
                # Should match (batch_size, seq_len)
                assert time_weights.shape == (batch_size, seq_len), \
                    "time_weights must match (batch_size, seq_len) if 2D."
            else:
                raise ValueError("time_weights must be 1D [seq_len] or 2D [batch_size, seq_len].")
            
            # Weighted sum
            time_weights = time_weights.unsqueeze(-1)  # => [B, T, 1]
            temporal_info = torch.sum(temporal_info * time_weights, dim=1)  # => [B, n_embd]
            feature_info = torch.sum(feature_info * time_weights, dim=1)  # => [B, n_embd]
        else:
            # Simple mean pooling
            temporal_info = temporal_info.mean(dim=1)  # => [B, n_embd]
            feature_info = feature_info.mean(dim=1)  # => [B, n_embd]
        
        # ------------------------------------
        # 5) Combine and predict
        # ------------------------------------
        combined = torch.cat([feature_info, temporal_info], dim=-1)  # => [B, 2*n_embd]
        combined = self.combine_layer(combined)                      # => [B, n_embd]
        price_pred = self.price_head(combined).squeeze(-1)          # => [B]

        return {
            'price_pred': price_pred
        }

    @torch.no_grad()
    def predict(self, features: torch.Tensor, time_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method for inference. 
        Args:
            features: [batch_size, seq_len, n_features]
            time_weights: optional, same shape considerations as forward

        Returns:
            price_pred: [batch_size]
        """
        self.eval()
        outputs = self(features, time_weights)
        return outputs['price_pred']

    def save_pretrained(self, save_directory: str):
        """
        Save model weights and config.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(save_directory, "config.pt")
        torch.save(self.config, config_path)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        Load model from directory.
        """
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        config_path = os.path.join(load_directory, "config.pt")
        
        # Load config
        config = torch.load(config_path)
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        
        return model


class OptionsDataset(Dataset):
    """
    Simple dataset to produce sequences for next-day price prediction.
    """
    def __init__(self,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 seq_length: int = 10) -> None:
        """
        Args:
            features: shape [num_samples, n_features]
            labels:   shape [num_samples]
            seq_length: number of timesteps per sample
        """
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
        # Example of exponential time weights:
        self.time_weights = torch.exp(torch.linspace(0, 1, seq_length))
        self.time_weights /= self.time_weights.sum()  # Normalize

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            feature_seq: [seq_length, n_features]
            label:       float (the next day's price or same-day price)
            time_weights: [seq_length] (just an example usage)
        """
        feature_seq = self.features[idx:idx + self.seq_length]  # shape [seq_length, n_features]
        
        # Next-day label: you might want to shift by +1 to truly be "next day"
        # but for demonstration, let's just take the last day in the window.
        label = self.labels[idx + self.seq_length - 1]
        
        return feature_seq, label, self.time_weights


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Hypothetical dataset: 100 samples, each with 15 features
    num_samples = 120
    n_features = 15
    seq_length = 10
    
    # Mock data
    dummy_features = torch.randn(num_samples, n_features)
    dummy_labels   = torch.randn(num_samples)  # e.g. next-day price or any numeric target
    
    # Dataset & dataloader
    dataset = OptionsDataset(dummy_features, dummy_labels, seq_length=seq_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Model config & initialization
    config = OptionsGPTConfig(
        n_embd=64,
        n_head=4,
        n_layer=2,
        n_positions=seq_length,
        n_features=n_features,
        dropout=0.25
    )
    model = OptionsGPT(config)
    
    # Simple training loop (illustrative only)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(2):
        model.train()
        total_loss = 0.0
        
        for batch_features, batch_labels, time_weights in dataloader:
            # Forward
            out = model(batch_features, time_weights)
            price_pred = out['price_pred']
            
            # Compute loss
            loss = criterion(price_pred, batch_labels)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss = {avg_loss:.4f}")
    
    # Example of saving & loading
    save_dir = "my_options_gpt_model"
    model.save_pretrained(save_dir)
    
    loaded_model = OptionsGPT.from_pretrained(save_dir)
    loaded_model.eval()
    
    # Single inference example
    test_seq = dummy_features[0:seq_length].unsqueeze(0) # => [1, seq_length, n_features]
    test_weights = dataset.time_weights.unsqueeze(0)     # => [1, seq_length]
    
    with torch.no_grad():
        prediction = loaded_model.predict(test_seq, test_weights)
    print("Prediction for first sample:", prediction.item())
