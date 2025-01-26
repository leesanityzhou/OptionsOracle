"""
Custom model architecture for options trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2Model
from typing import Optional

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for financial time series."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        """
        Initialize temporal attention.
        
        Args:
            hidden_size: Size of hidden states
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scaling = self.head_size ** -0.5
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of temporal attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Attended tensor of same shape as input
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Project inputs to Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(context)
        
        return output

class FinancialFeatureProcessor(nn.Module):
    """Process financial features with domain-specific layers."""
    
    def __init__(self, config):
        """
        Initialize feature processor.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
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
        
        self.technical_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        
        self.options_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.ReLU(),
            nn.Dropout(config.resid_pdrop)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature processor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Processed features of shape (batch_size, seq_len, hidden_size)
        """
        # For now, treat input as combined features
        technical_features = self.technical_encoder(x)
        options_features = self.options_encoder(x)
        
        # Combine features
        combined = torch.cat([technical_features, options_features], dim=-1)
        output = self.feature_combiner(combined)
        
        return output

class OptionsGPT(GPT2PreTrainedModel):
    """Custom GPT-2 model for options trading."""
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Load pretrained GPT-2 model
        self.transformer = GPT2Model(config)
        
        # Add temporal attention
        self.temporal_attention = TemporalAttention(config.n_embd)
        
        # Add financial feature processor
        self.feature_processor = FinancialFeatureProcessor(config)
        
        # Add custom layers for options trading
        self.feature_projection = nn.Linear(config.n_embd, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Output layers for different prediction tasks
        self.direction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, 2)  # Binary classification (up/down)
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, 1)  # Return prediction
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, 1)  # Volatility prediction
        )
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for training
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs including predictions and loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = None
        
        # Process input features
        if inputs_embeds is not None:
            processed_features = self.feature_processor(inputs_embeds)
            if output_attentions:
                # Get transformer outputs for attention visualization
                transformer_outputs = self.transformer(
                    inputs_embeds=processed_features,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            # Get transformer outputs
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            processed_features = transformer_outputs[0]
        
        # Apply temporal attention
        attended_features = self.temporal_attention(processed_features)
        
        # Project features
        projected_features = self.feature_projection(attended_features)
        projected_features = self.layer_norm(projected_features)
        projected_features = self.dropout(projected_features)
        
        # Get predictions from different heads (use last sequence element)
        direction_logits = self.direction_head(projected_features[:, -1, :])
        return_pred = self.return_head(projected_features[:, -1, :])
        volatility_pred = self.volatility_head(projected_features[:, -1, :])
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            direction_labels = labels[:, 0].long()  # First column is direction (convert to long)
            return_labels = labels[:, 1]  # Second column is return
            volatility_labels = labels[:, 2]  # Third column is volatility
            
            # Loss functions
            direction_loss_fct = nn.CrossEntropyLoss()
            regression_loss_fct = nn.MSELoss()
            
            # Calculate losses
            direction_loss = direction_loss_fct(direction_logits, direction_labels)
            return_loss = regression_loss_fct(return_pred.squeeze(), return_labels)
            volatility_loss = regression_loss_fct(volatility_pred.squeeze(), volatility_labels)
            
            # Combine losses with weights
            loss = direction_loss + 0.5 * return_loss + 0.5 * volatility_loss
        
        if not return_dict:
            output = (direction_logits, return_pred, volatility_pred)
            if transformer_outputs is not None:
                output = output + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'direction_logits': direction_logits,
            'return_pred': return_pred,
            'volatility_pred': volatility_pred,
            'hidden_states': transformer_outputs.hidden_states if transformer_outputs is not None and output_hidden_states else None,
            'attentions': transformer_outputs.attentions if transformer_outputs is not None and output_attentions else None
        }
    
    def predict(self, features: torch.Tensor) -> dict:
        """
        Make predictions on new data.
        
        Args:
            features: Input features tensor
            
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs_embeds=features)
            
            direction_probs = torch.softmax(outputs['direction_logits'], dim=-1)
            direction_pred = direction_probs.argmax(dim=-1)
            
            return {
                'direction': direction_pred,
                'direction_prob': direction_probs,
                'return': outputs['return_pred'],
                'volatility': outputs['volatility_pred']
            } 