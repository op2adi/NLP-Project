import torch

def debug_dimensions(model, verbose=False):
    """
    Analyze a model to report dimensions of parameters and buffers
    
    Args:
        model: PyTorch model to analyze
        verbose: Whether to print all parameter dimensions (True) or just summary (False)
    
    Returns:
        Dictionary with dimension information
    """
    results = {
        'total_params': 0,
        'trainable_params': 0,
        'param_dimensions': {},
        'potential_mismatches': []
    }
    
    # Track dimensions of linear layers for mismatch detection
    linear_layers = {}
    
    # Iterate through all named parameters and modules
    for name, param in model.named_parameters():
        results['total_params'] += param.numel()
        if param.requires_grad:
            results['trainable_params'] += param.numel()
        
        results['param_dimensions'][name] = list(param.shape)
        
        if verbose:
            print(f"{name}: {param.shape}")
    
    # Check modules for potential dimension mismatches
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            linear_layers[name] = (in_features, out_features)
    
    # Find potential dimension mismatches between connected linear layers
    for name1, (_, out_features1) in linear_layers.items():
        for name2, (in_features2, _) in linear_layers.items():
            # Check if layers might be connected based on naming patterns
            # This is a heuristic and might need adjustment for specific models
            if name1 != name2 and name1.split('.')[:-1] == name2.split('.')[:-1]:
                if out_features1 != in_features2:
                    results['potential_mismatches'].append(
                        f"Potential mismatch: {name1} (out: {out_features1}) -> {name2} (in: {in_features2})"
                    )
    
    return results

def print_model_structure(model):
    """Print the structure of a model, highlighting its hierarchy"""
    print("Model Structure:")
    print("---------------")
    
    def print_module_structure(module, prefix=''):
        for name, child in module.named_children():
            class_name = child.__class__.__name__
            params = sum(p.numel() for p in child.parameters())
            print(f"{prefix}├─ {name} ({class_name}), Parameters: {params:,}")
            print_module_structure(child, prefix + '│  ')
    
    print_module_structure(model)
    print("---------------")

def fix_dimension_mismatch(model_config, lstm_dim, mlp_dim):
    """
    Print instructions for fixing dimension mismatches
    
    Args:
        model_config: Current model configuration
        lstm_dim: LSTM output dimension
        mlp_dim: MLP dimension
    """
    print(f"Dimension Mismatch Detected: LSTM dim ({lstm_dim}) ≠ MLP expected input ({mlp_dim})")
    print("\nFix options:")
    print(f"1. Update config: Set dim_mlp={lstm_dim} to match LSTM output")
    print(f"2. Update model.py: Modify BiaffineScorer to handle dimension {lstm_dim}")
    print("\nExample config fix:")
    print("-------------------")
    print(f"dim_mlp = {lstm_dim}  # Changed from {mlp_dim}")
