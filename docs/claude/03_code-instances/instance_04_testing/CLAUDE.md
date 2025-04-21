# CLAUDE.md for Testing Instance (04_testing)

This file provides guidance to Claude Code when working with the testing instance of the RNA 3D folding pipeline. It includes test patterns, utilities, and commands for effective verification and validation.

## Test Configuration Commands

### Pytest Commands
- Run all tests: `python -m pytest tests/`
- Run specific test file: `python -m pytest tests/test_data_loading.py`
- Run specific test class: `python -m pytest tests/test_data_loading.py::TestRNADataset`
- Run specific test: `python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization`
- Run with verbosity: `python -m pytest tests/ -v`
- Run with coverage: `python -m pytest tests/ --cov=src`
- Run with coverage report: `python -m pytest tests/ --cov=src --cov-report=term-missing`
- Run with xvs flag for detailed tensor assertions: `python -m pytest tests/ -v -xvs`
- Run tests that match pattern: `python -m pytest tests/ -k "embedding"`
- Run tests with print output: `python -m pytest tests/ -s`

### Coverage Commands
- Generate coverage report: `coverage report -m`
- Generate HTML coverage report: `coverage html`
- View coverage report: `firefox htmlcov/index.html`

### Memory and Performance Profiling
- Memory profiling: `python -m memory_profiler tests/profile_memory.py`
- Line-by-line memory profile: `python -m memory_profiler tests/test_file.py::test_function`
- GPU profiling: `nvprof python tests/profile_gpu.py`
- CPU profiling: `python -m cProfile -o profile.stats tests/profile_cpu.py`
- View profile results: `python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(30)"`

## Test Pattern Utilities

### Tensor Assertion Helpers
```python
def assert_tensor_equal(actual, expected, rtol=1e-5, atol=1e-5, msg=None):
    """Assert that two tensors are equal within tolerance."""
    # Convert to tensors if needed
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected)
    
    # Check shape
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    
    # Check device
    assert actual.device == expected.device, f"Device mismatch: {actual.device} vs {expected.device}"
    
    # Check dtype
    assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
    
    # Check values
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Values not equal: max diff {torch.max(torch.abs(actual - expected))}"

def assert_shape(tensor, expected_shape, msg=None):
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, \
        f"{msg or 'Shape mismatch:'} {tensor.shape} vs {expected_shape}"

def assert_mask_consistency(mask, data, msg=None):
    """Assert that a mask is consistent with the data it masks."""
    assert mask.shape == data.shape[:len(mask.shape)], \
        f"{msg or 'Mask shape inconsistent with data:'} {mask.shape} vs {data.shape[:len(mask.shape)]}"
    assert mask.dtype == torch.bool, f"Mask must be boolean, got {mask.dtype}"

def assert_gradient_flow(model, inputs, loss_fn):
    """Assert that gradients flow through the model."""
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad_tensor = param.grad.abs().sum() > 0
            if has_grad_tensor:
                has_grad = True
                break
    
    assert has_grad, "No gradients are flowing through the model"
```

### Mock Data Generation
```python
def generate_mock_sequence(length, batch_size=1):
    """Generate mock RNA sequence data."""
    # Generate one-hot encoded sequences (A, C, G, U)
    sequences = torch.zeros((batch_size, length, 4), dtype=torch.float32)
    for b in range(batch_size):
        for i in range(length):
            base_idx = torch.randint(0, 4, (1,))
            sequences[b, i, base_idx] = 1.0
    return sequences

def generate_mock_coordinates(length, batch_size=1):
    """Generate mock 3D coordinates."""
    # Generate coordinates for N, C, O atoms (3 atoms per residue)
    coords = torch.randn((batch_size, length, 3, 3), dtype=torch.float32)
    return coords

def generate_mock_features(length, feature_dim, batch_size=1):
    """Generate mock feature tensors."""
    features = torch.randn((batch_size, length, feature_dim), dtype=torch.float32)
    return features

def generate_mock_pair_features(length, feature_dim, batch_size=1):
    """Generate mock pairwise feature tensors."""
    pair_features = torch.randn((batch_size, length, length, feature_dim), dtype=torch.float32)
    return pair_features

def generate_mock_masks(length, batch_size=1, valid_prob=0.9):
    """Generate mock masks with some positions masked."""
    masks = torch.rand((batch_size, length)) < valid_prob
    return masks.to(torch.bool)

def generate_mock_batch(batch_size, min_length=50, max_length=100, feature_dim=128, pair_dim=64):
    """Generate a mock batch with variable sequence lengths."""
    lengths = torch.randint(min_length, max_length+1, (batch_size,))
    max_len = lengths.max().item()
    
    # Initialize tensors with padding
    sequences = torch.zeros((batch_size, max_len, 4), dtype=torch.float32)
    coordinates = torch.zeros((batch_size, max_len, 3, 3), dtype=torch.float32)
    features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    pair_features = torch.zeros((batch_size, max_len, max_len, pair_dim), dtype=torch.float32)
    masks = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill with data up to each sequence's length
    for i in range(batch_size):
        length = lengths[i].item()
        sequences[i, :length] = generate_mock_sequence(length, 1)[0]
        coordinates[i, :length] = generate_mock_coordinates(length, 1)[0]
        features[i, :length] = generate_mock_features(length, feature_dim, 1)[0]
        pair_features[i, :length, :length] = generate_mock_pair_features(length, pair_dim, 1)[0]
        masks[i, :length] = True
    
    return {
        'sequences': sequences,
        'coordinates': coordinates,
        'features': features,
        'pair_features': pair_features,
        'masks': masks,
        'lengths': lengths
    }
```

### Memory Profiling Utilities
```python
def profile_memory_usage(func):
    """Decorator to profile memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            start_memory = psutil.Process().memory_info().rss / 1024**2
        
        # Run function
        result = func(*args, **kwargs)
        
        # Record memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            end_memory = psutil.Process().memory_info().rss / 1024**2
        
        print(f"Memory usage for {func.__name__}: {end_memory - start_memory:.2f} MB")
        return result
    return wrapper

def measure_peak_memory(func):
    """Measure peak memory usage during function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            result = func(*args, **kwargs)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU memory for {func.__name__}: {peak_memory:.2f} MB")
        else:
            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak CPU memory for {func.__name__}: {peak / 1024**2:.2f} MB")
            tracemalloc.stop()
        return result
    return wrapper
```

### Reproducibility Utilities
```python
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_reproducible_dataloader(dataset, batch_size, seed=42):
    """Create a reproducible dataloader with fixed seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
```

## Custom Assertion Helpers for Testing

### Tensor Shape Verification
```python
# Test helper for verifying tensor shapes throughout the model
def verify_tensor_shapes(model, input_shape, expected_shapes_dict):
    """
    Verify tensor shapes at various points in a model.
    
    Args:
        model: The model to test
        input_shape: The input shape to the model
        expected_shapes_dict: Dict mapping layer names to expected output shapes
    """
    # Create a dummy input
    x = torch.randn(input_shape)
    
    # Store activations
    activations = {}
    hooks = []
    
    # Define hook function
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    # Register hooks for each layer
    for name, layer in model.named_modules():
        if name in expected_shapes_dict:
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model(x)
    
    # Check shapes
    for name, expected_shape in expected_shapes_dict.items():
        assert name in activations, f"No activation recorded for {name}"
        if isinstance(expected_shape, tuple):
            # Single output
            assert activations[name].shape == expected_shape, \
                f"Shape mismatch for {name}: expected {expected_shape}, got {activations[name].shape}"
        else:
            # Multiple outputs
            for i, shape in enumerate(expected_shape):
                assert activations[name][i].shape == shape, \
                    f"Shape mismatch for {name}[{i}]: expected {shape}, got {activations[name][i].shape}"
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
```

### Gradient Flow Testing
```python
def test_gradient_flow(model, input_tensors, target=None, loss_fn=None):
    """
    Test that gradients flow through the model.
    
    Args:
        model: The model to test
        input_tensors: Input to the model
        target: Target values if needed
        loss_fn: Loss function to use, if None uses MSE on model output
    """
    # Set gradients to zero
    model.zero_grad()
    
    # Forward pass
    output = model(input_tensors)
    
    # Compute loss
    if loss_fn is None:
        if target is None:
            # Default: Use mean of output as loss
            loss = output.mean()
        else:
            # Default: MSE loss
            loss = torch.nn.functional.mse_loss(output, target)
    else:
        loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are not zero
    grads_exist = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    grads_exist = True
                    break
    
    assert grads_exist, "No gradients are flowing through the model"
    
    # Return gradients for additional checks
    return {name: param.grad.clone() for name, param in model.named_parameters() 
            if param.requires_grad and param.grad is not None}
```

### Performance Benchmarking
```python
def benchmark_function(func, *args, n_runs=10, **kwargs):
    """
    Benchmark a function's execution time.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to the function
        n_runs: Number of runs to average
        kwargs: Keyword arguments to pass to the function
        
    Returns:
        Average execution time in milliseconds
    """
    times = []
    for _ in range(n_runs):
        # Ensure CUDA operations are completed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time the function
        start = time.time()
        result = func(*args, **kwargs)
        
        # Ensure CUDA operations are completed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        
    avg_time = sum(times) / len(times)
    print(f"Function '{func.__name__}' average execution time: {avg_time:.2f} ms over {n_runs} runs")
    return avg_time, result
```

## Verification Protocol Utilities

### Interface Contract Validator
```python
def validate_interface_contract(component, contract_dict):
    """
    Validate that a component adheres to its interface contract.
    
    Args:
        component: The component to validate
        contract_dict: Dictionary defining the expected interface
            {
                'methods': {
                    'method_name': {
                        'params': [{'name': 'x', 'type': torch.Tensor}, ...],
                        'return': {'type': torch.Tensor, 'shape': (None, 10)}
                    },
                    ...
                },
                'attributes': {
                    'attr_name': {'type': int},
                    ...
                }
            }
    
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check methods
    for method_name, method_spec in contract_dict.get('methods', {}).items():
        # Check method exists
        if not hasattr(component, method_name):
            errors.append(f"Method '{method_name}' not found")
            continue
            
        method = getattr(component, method_name)
        if not callable(method):
            errors.append(f"'{method_name}' is not callable")
            continue
            
        # Check method signature
        sig = inspect.signature(method)
        expected_params = method_spec.get('params', [])
        
        # Check parameter count (excluding self)
        if len(sig.parameters) - 1 != len(expected_params):
            errors.append(f"Method '{method_name}' has {len(sig.parameters) - 1} parameters, expected {len(expected_params)}")
        
        # Check parameter names and annotations
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
        for i, expected in enumerate(expected_params):
            if i >= len(param_names):
                errors.append(f"Method '{method_name}' missing parameter '{expected['name']}'")
                continue
                
            actual_name = param_names[i]
            if actual_name != expected['name']:
                errors.append(f"Method '{method_name}' parameter {i} has name '{actual_name}', expected '{expected['name']}'")
    
    # Check attributes
    for attr_name, attr_spec in contract_dict.get('attributes', {}).items():
        # Check attribute exists
        if not hasattr(component, attr_name):
            errors.append(f"Attribute '{attr_name}' not found")
            continue
            
        attr = getattr(component, attr_name)
        expected_type = attr_spec.get('type')
        
        # Check attribute type
        if expected_type and not isinstance(attr, expected_type):
            errors.append(f"Attribute '{attr_name}' has type '{type(attr)}', expected '{expected_type}'")
    
    return errors
```

## Test Organization Standards

### Test Class Structure
```python
class TestComponentName(unittest.TestCase):
    """Tests for ComponentName."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Set random seeds for reproducibility
        set_seeds(42)
        
        # Create shared resources
        cls.mock_data = ...

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a new instance for each test
        self.component = ComponentName(...)
    
    def tearDown(self):
        """Clean up after each test."""
        # Release resources
        pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Release shared resources
        pass
    
    # Test groups
    
    # 1. Initialization tests
    def test_initialization(self):
        """Test that the component initializes correctly."""
        ...
    
    # 2. Interface tests
    def test_interface_parameters(self):
        """Test that the interface accepts the correct parameters."""
        ...
    
    def test_interface_return_values(self):
        """Test that the interface returns the correct values."""
        ...
    
    # 3. Functionality tests
    def test_core_functionality(self):
        """Test the core functionality of the component."""
        ...
    
    # 4. Edge case tests
    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        ...
    
    def test_edge_case_max_values(self):
        """Test behavior with maximum values."""
        ...
    
    # 5. Error handling tests
    def test_error_invalid_input(self):
        """Test that appropriate errors are raised for invalid input."""
        ...
    
    # 6. Performance tests
    def test_performance(self):
        """Test performance characteristics."""
        ...
```

## Verification Templates

### Component Verification Plan Template
```
# Verification Plan: [Component Name]

## 1. Verification Scope
- [Brief description of what will be verified]
- Component source: [path to component implementation]
- Interface contract: [path to interface contract]

## 2. Verification Team
- Lead Verifier: [name/id]
- Supporting Verifiers: [names/ids]

## 3. Verification Schedule
- Start Date: [date]
- Target Completion: [date]
- Verification Review: [date]

## 4. Verification Environment
- Hardware Configuration: [CPU/GPU specs]
- Software Dependencies: [versions]
- Test Framework: [pytest version]

## 5. Verification Approach
- Unit Testing: [approach]
- Integration Testing: [approach]
- Performance Testing: [approach]
- Edge Case Testing: [approach]

## 6. Verification Test Cases
[List of specific test cases to be implemented]

## 7. Acceptance Criteria
- Interface Compliance: [criteria]
- Functional Correctness: [criteria]
- Performance Requirements: [criteria]
- Integration Compatibility: [criteria]

## 8. Risks and Mitigations
[List of verification risks and planned mitigations]

## 9. Verification Deliverables
- Test Reports: [format/location]
- Performance Analysis: [format/location]
- Issue Reports: [format/location]
```

## Verification Checklist
1. **Verification Preparation**
   - [ ] Review component interface documentation
   - [ ] Review component implementation
   - [ ] Identify integration points with other components
   - [ ] Create verification plan
   - [ ] Prepare test environment
   - [ ] Develop mock data generators

2. **Interface Verification**  
   - [ ] Verify public interface matches documentation
   - [ ] Verify parameter types and shapes
   - [ ] Verify return types and shapes
   - [ ] Verify error handling
   - [ ] Verify device compatibility

3. **Functional Verification**
   - [ ] Test core functionality with valid inputs
   - [ ] Test error handling with invalid inputs
   - [ ] Test edge cases
   - [ ] Test numerical stability
   - [ ] Test mask handling

4. **Integration Verification**
   - [ ] Test with adjacent components
   - [ ] Verify data flow
   - [ ] Test end-to-end processing
   - [ ] Verify memory management

5. **Performance Benchmarking**
   - [ ] Measure execution time
   - [ ] Analyze memory usage
   - [ ] Test scaling behavior
   - [ ] Profile GPU utilization
   - [ ] Compare against performance requirements