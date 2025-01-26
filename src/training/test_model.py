"""
Tests for the options trading model.
"""

import unittest
import torch
import numpy as np
from transformers import GPT2Config

from src.training.model import OptionsGPT, TemporalAttention, FinancialFeatureProcessor

class TestTemporalAttention(unittest.TestCase):
    """Test cases for temporal attention mechanism."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = GPT2Config(n_embd=768)
        self.batch_size = 4
        self.seq_length = 30
        self.attention = TemporalAttention(self.config)
        
    def test_attention_shape(self):
        """Test output shape of attention mechanism."""
        x = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        output = self.attention(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_attention_mask(self):
        """Test attention masking."""
        x = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)
        mask[:, :, self.seq_length//2:] = 0  # Mask out second half
        
        output_masked = self.attention(x, mask)
        output_unmasked = self.attention(x)
        
        # Outputs should be different with mask
        self.assertFalse(torch.allclose(output_masked, output_unmasked))
        
    def test_attention_gradients(self):
        """Test gradient flow through attention."""
        x = torch.randn(self.batch_size, self.seq_length, self.config.n_embd, requires_grad=True)
        output = self.attention(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

class TestFinancialFeatureProcessor(unittest.TestCase):
    """Test cases for financial feature processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = GPT2Config(n_embd=768)
        self.processor = FinancialFeatureProcessor(self.config)
        self.batch_size = 4
        self.seq_length = 30
        
    def test_price_encoder(self):
        """Test price encoding."""
        price = torch.randn(self.batch_size, self.seq_length, 1)
        output = self.processor.price_encoder(price)
        
        expected_shape = (self.batch_size, self.seq_length, self.config.n_embd // 2)
        self.assertEqual(output.shape, expected_shape)
        
    def test_volume_encoder(self):
        """Test volume encoding."""
        volume = torch.randn(self.batch_size, self.seq_length, 1)
        output = self.processor.volume_encoder(volume)
        
        expected_shape = (self.batch_size, self.seq_length, self.config.n_embd // 2)
        self.assertEqual(output.shape, expected_shape)
        
    def test_technical_encoder(self):
        """Test technical feature encoding."""
        features = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        output = self.processor.technical_encoder(features)
        
        expected_shape = (self.batch_size, self.seq_length, self.config.n_embd)
        self.assertEqual(output.shape, expected_shape)
        
    def test_options_encoder(self):
        """Test options feature encoding."""
        features = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        output = self.processor.options_encoder(features)
        
        expected_shape = (self.batch_size, self.seq_length, self.config.n_embd)
        self.assertEqual(output.shape, expected_shape)
        
    def test_feature_combiner(self):
        """Test feature combination."""
        features = torch.randn(self.batch_size, self.seq_length, self.config.n_embd * 2)
        output = self.processor.feature_combiner(features)
        
        expected_shape = (self.batch_size, self.seq_length, self.config.n_embd)
        self.assertEqual(output.shape, expected_shape)

class TestOptionsGPT(unittest.TestCase):
    """Test cases for the main model."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = GPT2Config(
            n_embd=768,
            n_layer=6,
            n_head=8,
            resid_pdrop=0.1
        )
        self.model = OptionsGPT(self.config)
        self.batch_size = 4
        self.seq_length = 30
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.transformer)
        self.assertIsNotNone(self.model.temporal_attention)
        self.assertIsNotNone(self.model.feature_processor)
        
    def test_forward_pass_with_embeddings(self):
        """Test forward pass with input embeddings."""
        inputs_embeds = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        outputs = self.model(inputs_embeds=inputs_embeds)
        
        self.assertIn('direction_logits', outputs)
        self.assertIn('return_pred', outputs)
        self.assertIn('volatility_pred', outputs)
        
        # Check shapes
        self.assertEqual(outputs['direction_logits'].shape, (self.batch_size, 2))
        self.assertEqual(outputs['return_pred'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['volatility_pred'].shape, (self.batch_size, 1))
        
    def test_forward_pass_with_labels(self):
        """Test forward pass with labels."""
        inputs_embeds = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        labels = torch.zeros(self.batch_size, 3)  # [direction, return, volatility]
        labels[:, 0] = torch.randint(0, 2, (self.batch_size,))  # Random binary labels for direction
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        
        self.assertIn('loss', outputs)
        self.assertIsNotNone(outputs['loss'])
        self.assertFalse(torch.isnan(outputs['loss']).any())
        
    def test_predict_method(self):
        """Test prediction method."""
        features = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        predictions = self.model.predict(features)
        
        self.assertIn('direction', predictions)
        self.assertIn('direction_prob', predictions)
        self.assertIn('return', predictions)
        self.assertIn('volatility', predictions)
        
        # Check shapes
        self.assertEqual(predictions['direction'].shape, (self.batch_size,))
        self.assertEqual(predictions['direction_prob'].shape, (self.batch_size, 2))
        self.assertEqual(predictions['return'].shape, (self.batch_size, 1))
        self.assertEqual(predictions['volatility'].shape, (self.batch_size, 1))
        
    def test_model_training(self):
        """Test model training step."""
        inputs_embeds = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        labels = torch.zeros(self.batch_size, 3)  # [direction, return, volatility]
        labels[:, 0] = torch.randint(0, 2, (self.batch_size,))  # Random binary labels for direction
        
        # Set model to training mode
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs['loss']
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Check if at least some parameters have gradients
        has_grad = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        self.assertTrue(has_grad, "No parameter gradients found after backward pass")
        
    def test_attention_visualization(self):
        """Test attention weight visualization."""
        inputs_embeds = torch.randn(self.batch_size, self.seq_length, self.config.n_embd)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )
        
        self.assertIn('attentions', outputs)
        self.assertIsNotNone(outputs['attentions'])
        
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            self.model.save_pretrained(temp_dir)
            
            # Load model
            loaded_model = OptionsGPT.from_pretrained(temp_dir)
            
            # Check if models have the same parameters
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

if __name__ == '__main__':
    unittest.main() 