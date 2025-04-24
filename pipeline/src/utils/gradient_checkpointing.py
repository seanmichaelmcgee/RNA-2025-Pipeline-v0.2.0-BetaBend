"""
Gradient Checkpointing Utilities for RNA Structure Prediction

This module provides tools for applying gradient checkpointing to reduce
memory usage in transformer and IPA-based RNA structure prediction models.
"""

import logging
import torch
import torch.utils.checkpoint as checkpoint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Convenient alias functions for backwards compatibility
def enable_gradient_checkpointing(model: torch.nn.Module) -> bool:
    """Convenience function to enable gradient checkpointing"""
    return apply_gradient_checkpointing(model, enable=True)

def apply_gradient_checkpointing_to_transformer(model: torch.nn.Module) -> bool:
    """Applies gradient checkpointing to transformer blocks in the model"""
    return apply_to_transformers(model, enable=True)

def apply_checkpointing_to_ipa(model: torch.nn.Module) -> bool:
    """Applies gradient checkpointing to IPA modules in the model"""
    return apply_to_ipa(model, enable=True)

def apply_gradient_checkpointing(model: torch.nn.Module, enable: bool = True) -> bool:
    """
    Apply gradient checkpointing to supported model components.
    
    Args:
        model: PyTorch model
        enable: Whether to enable or disable checkpointing
        
    Returns:
        True if checkpointing was applied to any component
    """
    applied = False
    
    # Try to apply to transformer blocks
    if apply_to_transformers(model, enable):
        applied = True
        
    # Try to apply to IPA module
    if apply_to_ipa(model, enable):
        applied = True
        
    # Try to apply to attention layers
    if apply_to_attention(model, enable):
        applied = True
    
    if applied:
        logger.info(f"{'Enabled' if enable else 'Disabled'} gradient checkpointing")
    else:
        logger.warning("No compatible components found for gradient checkpointing")
    
    return applied


def apply_to_transformers(model: torch.nn.Module, enable: bool = True) -> bool:
    """
    Apply gradient checkpointing to transformer blocks in a model.
    
    Args:
        model: PyTorch model containing transformer blocks
        enable: Whether to enable or disable checkpointing
        
    Returns:
        True if checkpointing was applied to any transformer blocks
    """
    applied = False
    
    # Look for transformer blocks with activation checkpointing attribute
    if hasattr(model, 'transformer_blocks'):
        for i, block in enumerate(model.transformer_blocks):
            if hasattr(block, 'use_checkpointing'):
                block.use_checkpointing = enable
                applied = True
            elif hasattr(block, 'self_attention') and hasattr(block.self_attention, 'use_checkpoint'):
                # Alternative attribute name
                block.self_attention.use_checkpoint = enable
                applied = True
            
            # Try direct method patching
            _patch_forward_method(block, enable)
    
    # Look for encoder/decoder with transformer blocks
    for component_name in ['encoder', 'decoder']:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if hasattr(component, 'layers'):
                for layer in component.layers:
                    # Try direct method patching
                    _patch_forward_method(layer, enable)
                    
                    # Try attention components
                    if hasattr(layer, 'self_attn'):
                        _patch_forward_method(layer.self_attn, enable)
                    
                    # Also try to apply to each layer
                    if hasattr(layer, 'use_checkpointing'):
                        layer.use_checkpointing = enable
                        applied = True
    
    if applied:
        logger.info(f"{'Enabled' if enable else 'Disabled'} checkpointing for transformer blocks")
    
    return applied


def apply_to_ipa(model: torch.nn.Module, enable: bool = True) -> bool:
    """
    Apply gradient checkpointing to IPA (Invariant Point Attention) modules.
    
    Args:
        model: PyTorch model containing IPA modules
        enable: Whether to enable or disable checkpointing
        
    Returns:
        True if checkpointing was applied to any IPA modules
    """
    applied = False
    
    # Check if model has IPA module
    if hasattr(model, 'ipa_module'):
        # Check for use_checkpointing attribute
        if hasattr(model.ipa_module, 'use_checkpointing'):
            model.ipa_module.use_checkpointing = enable
            applied = True
        
        # Apply to IPA module itself
        _patch_forward_method(model.ipa_module, enable)
        
        # Try to apply to individual layers if present
        if hasattr(model.ipa_module, 'layers'):
            for layer in model.ipa_module.layers:
                if hasattr(layer, 'use_checkpointing'):
                    layer.use_checkpointing = enable
                    applied = True
                
                # Apply to each layer
                _patch_forward_method(layer, enable)
    
    if applied:
        logger.info(f"{'Enabled' if enable else 'Disabled'} checkpointing for IPA module")
    
    return applied


def apply_to_attention(model: torch.nn.Module, enable: bool = True) -> bool:
    """
    Apply gradient checkpointing to attention mechanisms.
    
    Args:
        model: PyTorch model with attention mechanisms
        enable: Whether to enable or disable checkpointing
        
    Returns:
        True if checkpointing was applied to any attention mechanism
    """
    applied = False
    
    # Common attention module patterns
    attention_names = [
        'self_attention', 'attention', 'self_attn', 'mha', 'multihead_attn',
        'cross_attention', 'cross_attn'
    ]
    
    # Check direct attributes
    for name in attention_names:
        if hasattr(model, name):
            attn_module = getattr(model, name)
            _patch_forward_method(attn_module, enable)
            
            if hasattr(attn_module, 'use_checkpointing'):
                attn_module.use_checkpointing = enable
                applied = True
    
    # Recursive application to all modules with matching names
    for name, module in model.named_children():
        if any(attn_name in name.lower() for attn_name in attention_names):
            _patch_forward_method(module, enable)
            
            if hasattr(module, 'use_checkpointing'):
                module.use_checkpointing = enable
                applied = True
        
        # Recurse into child modules
        if len(list(module.children())) > 0:
            sub_applied = apply_to_attention(module, enable)
            applied = applied or sub_applied
    
    return applied


def _patch_forward_method(module: torch.nn.Module, enable: bool = True) -> bool:
    """
    Patch the forward method of a module with gradient checkpointing.
    
    Args:
        module: Module to patch
        enable: Whether to enable or disable checkpointing
        
    Returns:
        True if patching was successful
    """
    if not hasattr(module, 'forward'):
        return False
    
    # Store original forward if not already stored
    if not hasattr(module, '_original_forward'):
        module._original_forward = module.forward
    
    # Define checkpointed forward
    def checkpointed_forward(*args, **kwargs):
        def custom_forward(*inputs):
            if kwargs:
                return module._original_forward(*inputs, **kwargs)
            else:
                return module._original_forward(*inputs)
        
        return checkpoint.checkpoint(custom_forward, *args)
    
    # Apply or remove patching
    if enable:
        module.forward = checkpointed_forward
        return True
    else:
        # Restore original forward if it exists
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            return True
    
    return False
