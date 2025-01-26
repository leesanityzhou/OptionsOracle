"""
Custom model architecture for options trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2Model, PretrainedConfig, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from typing import Optional
import math

class OptionsGPTConfig(PretrainedConfig):
    """Configuration class for OptionsGPT model."""
    model_type = "options_gpt"

    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        vocab_size=50257,  # Default GPT2 vocab size
        n_positions=1024,  # Default GPT2 max sequence length
        n_ctx=1024,  # Default GPT2 context size
        n_embd=None,  # Will be set to hidden_size
        n_layer=None,  # Will be set to num_layers
        n_head=None,  # Will be set to num_heads
        n_inner=None,  # Will be set to hidden_size * 4
        activation_function="gelu_new",  # Default GPT2 activation
        resid_pdrop=None,  # Will be set to dropout
        embd_pdrop=None,  # Will be set to dropout
        attn_pdrop=None,  # Will be set to dropout
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Set main parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Set required GPT2 parameters
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.max_position_embeddings = n_positions
        
        # Set aliases for GPT2 compatibility
        self.n_embd = hidden_size if n_embd is None else n_embd
        self.n_layer = num_layers if n_layer is None else n_layer
        self.n_head = num_heads if n_head is None else n_head
        self.n_inner = hidden_size * 4 if n_inner is None else n_inner
        self.num_hidden_layers = self.n_layer  # Required by GPT2
        self.num_attention_heads = self.n_head  # Required by GPT2
        
        # Set dropout aliases
        self.resid_pdrop = dropout if resid_pdrop is None else resid_pdrop
        self.embd_pdrop = dropout if embd_pdrop is None else embd_pdrop
        self.attn_pdrop = dropout if attn_pdrop is None else attn_pdrop
        
        # Set additional required GPT2 parameters
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.activation_function = activation_function
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

class FeatureProcessor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.feature_projection = nn.Linear(hidden_size, hidden_size)  # Project from hidden_size to hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Handle both 3D and 4D inputs
        if len(x.shape) == 3:
            # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = x.shape
        else:
            # [batch_size, 1, seq_len, hidden_size]
            batch_size, _, seq_len, hidden_size = x.shape
            x = x.squeeze(1)  # Remove the extra dimension
        
        # Project features
        x = self.feature_projection(x)  # [batch_size, seq_len, hidden_size]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x  # Return in 3D format [batch_size, seq_len, hidden_size]

class TemporalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.resid_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
    def forward(self, x, mask=None, output_attentions=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Linear projections
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        
        if mask is not None:
            att = att.masked_fill(mask[:, None, :, :] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        
        if output_attentions:
            return y, att
        return y

class FinancialFeatureProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature encoders
        self.price_encoder = nn.Sequential(
            nn.Linear(1, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, config.n_embd // 2)
        )
        
        self.volume_encoder = nn.Sequential(
            nn.Linear(1, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, config.n_embd // 2)
        )
        
        self.options_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, config.n_embd)
        )
        
        self.technical_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, config.n_embd)
        )

        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.ReLU()
        )

    def forward(self, price, volume, options, technical):
        # Process each feature type
        price_encoded = self.price_encoder(price)  # [batch, seq_len, n_embd//2]
        volume_encoded = self.volume_encoder(volume)  # [batch, seq_len, n_embd//2]
        options_encoded = self.options_encoder(options)  # [batch, seq_len, n_embd]
        technical_encoded = self.technical_encoder(technical)  # [batch, seq_len, n_embd]
        
        # Concatenate encoded features
        combined = torch.cat([price_encoded, volume_encoded], dim=-1)
        
        # Combine features
        return self.feature_combiner(combined)  # [batch, seq_len, n_embd]

    def combine_features(self, features):
        """Combine pre-encoded features."""
        return self.feature_combiner(features)

class OptionsGPT(nn.Module):
    """Custom GPT-2 model for options trading."""
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # vocab_size=1 since we're not using tokens
        gpt2_config = GPT2Config(
            vocab_size=1,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=4 * 768,  # 4x embedding size
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_by_inverse_layer_idx=True,
            reorder_and_upcast_attn=True,
            use_cache=False
        )
        
        self.transformer = GPT2Model(gpt2_config)
        self.feature_processor = FinancialFeatureProcessor(config)
        self.temporal_attention = TemporalAttention(config)
        
        # Prediction heads with layer norm
        self.direction_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 2)
        )
        
        self.return_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 1)
        )
        
        # Loss functions
        self.direction_loss_fct = nn.CrossEntropyLoss()
        self.return_loss_fct = nn.MSELoss()
        self.volatility_loss_fct = nn.MSELoss()
        
    def save_pretrained(self, save_directory):
        """Save model to the specified directory."""
        import os
        import torch
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
    @classmethod
    def from_pretrained(cls, load_directory):
        """Load model from the specified directory."""
        import os
        import torch
        
        # Create new model instance
        config = GPT2Config(n_embd=768)  # Default config
        model = cls(config)
        
        # Load state dict
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path))
        
        return model
        
    def forward(self, inputs_embeds=None, labels=None, output_attentions=False):
        """
        Forward pass of the model.
        """
        # Process features through temporal attention
        if output_attentions:
            hidden_states, attentions = self.temporal_attention(inputs_embeds, output_attentions=True)
        else:
            hidden_states = self.temporal_attention(inputs_embeds)
        
        # Project to prediction heads
        direction_logits = self.direction_head(hidden_states[:, -1, :])  # Use last timestep
        return_preds = self.return_head(hidden_states[:, -1, :]).view(-1, 1)  # [batch_size, 1]
        volatility_preds = self.volatility_head(hidden_states[:, -1, :]).view(-1, 1)  # [batch_size, 1]
        
        # Get direction predictions
        direction_preds = F.softmax(direction_logits, dim=-1)
        
        outputs = {
            'direction_logits': direction_logits,
            'direction_preds': direction_preds,
            'return_pred': return_preds,  # [batch_size, 1]
            'volatility_pred': volatility_preds  # [batch_size, 1]
        }
        
        if output_attentions:
            outputs['attentions'] = attentions
        
        if labels is not None:
            direction_labels = labels[:, 0]
            return_labels = labels[:, 1].view(-1, 1)  # Match shape with predictions
            volatility_labels = labels[:, 2].view(-1, 1)  # Match shape with predictions
            
            # Calculate losses
            direction_loss = self.direction_loss_fct(direction_logits, direction_labels.long())
            return_loss = self.return_loss_fct(return_preds, return_labels)
            volatility_loss = self.volatility_loss_fct(volatility_preds, volatility_labels)
            
            # Combined loss with higher weight on direction prediction
            loss = 2.0 * direction_loss + 0.5 * return_loss + 0.5 * volatility_loss
            outputs["loss"] = loss
            
        return outputs
    
    def predict(self, features):
        """
        Make predictions on input features.
        """
        outputs = self(inputs_embeds=features)
        
        direction_preds = outputs['direction_preds']
        direction = torch.argmax(direction_preds, dim=-1)
        
        return {
            'direction': direction,
            'direction_prob': direction_preds,
            'return': outputs['return_pred'],  # Already [batch_size, 1]
            'volatility': outputs['volatility_pred']  # Already [batch_size, 1]
        } 