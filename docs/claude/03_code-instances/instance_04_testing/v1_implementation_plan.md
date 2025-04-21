# V1 Implementation-Focused Plan for Testing Instance

## 1. Overview and Objectives

This plan refocuses the testing instance (04_testing) on implementation-oriented goals to accelerate the development of a functional V1 model. Following the validation strategy document, we prioritize technical validation over extensive documentation and shift resources toward functional testing.

**Primary Objectives:**
1. Implement Tier 1 (Fast Technical Validation) framework
2. Create end-to-end pipeline test with mock data
3. Fix critical numerical issues in loss functions
4. Set up validation notebooks following the standardized structure
5. Implement core metrics (TM-score, RMSD) for structure evaluation

## 2. Implementation Priorities

### 2.1 Phase 1: End-to-End Pipeline Verification (1-2 days)
- [x] Implement test fixtures for mock data generation
- [x] Create end-to-end integration test (test_integration.py)
- [ ] Verify data → model → loss flow with mock data
- [ ] Fix critical numerical stability issues in loss functions
- [ ] Implement checkpointing and loading test

### 2.2 Phase 2: Technical Validation Framework (2-3 days)
- [ ] Create validation notebook template following section 4.1 structure
- [ ] Implement Tier 1 validation script for 3-5 sequences
- [ ] Establish TM-score and RMSD calculation utilities
- [ ] Set up basic visualization components for structure comparison
- [ ] Create mock data representing different RNA types (short, medium, long)

### 2.3 Phase 3: Mock-to-Real Data Transition (2-3 days)
- [ ] Create small validation subset from real data
- [ ] Update data loading test to support validation subset
- [ ] Implement temporal cutoff verification
- [ ] Create validation results tracking system
- [ ] Set up version numbering scheme per section 6.2

## 3. Implementation Tasks

### 3.1 End-to-End Pipeline

#### 3.1.1 Pipeline Integration Test
```python
# Task: Enhance test_integration.py to validate full pipeline execution
# Priority: High
# Dependencies: None
# Expected completion: 1 day

def test_end_to_end_pipeline():
    """Test complete data->model->loss->backward pipeline."""
    # Create mock dataset with varied sequence lengths
    # Create model with test configuration
    # Run forward pass through model
    # Calculate losses
    # Run backward pass
    # Verify gradients flow through all components
    # Check model can be saved and loaded
```

#### 3.1.2 Loss Function Stability Fixes
```python
# Task: Fix critical numerical issues in loss functions
# Priority: High
# Dependencies: None
# Expected completion: 1 day

def test_stable_kabsch_align():
    """Test improved Kabsch alignment with edge cases."""
    # Test with rotated points
    # Test with collinear points
    # Test with reflection cases
    # Verify numerical stability
```

```python
# Task: Fix robust distance calculation
# Priority: High
# Dependencies: None
# Expected completion: 0.5 days

def test_robust_distance():
    """Test robust distance calculation with edge cases."""
    # Test with identical points
    # Test with very small differences
    # Test with epsilon handling
    # Verify numerical stability
```

### 3.2 Validation Framework

#### 3.2.1 Validation Notebook Template
```python
# Task: Create validation notebook template
# Priority: Medium
# Dependencies: End-to-end pipeline test
# Expected completion: 1 day

"""
Structure:
1. Setup & Data Loading
2. Model Loading
3. Inference & Prediction
4. Evaluation & Metrics

Implement as per section 4.1 of validation strategy document
"""
```

#### 3.2.2 TM-Score Implementation
```python
# Task: Implement TM-score calculation
# Priority: Medium
# Dependencies: None
# Expected completion: 1.5 days

def calculate_tm_score(pred_coords, true_coords, mask=None):
    """Calculate TM-score between predicted and true coordinates."""
    # Implement TM-score algorithm
    # Support masking for variable length sequences
    # Match Kaggle evaluation formula
    # Optimize for performance
```

#### 3.2.3 RMSD Implementation
```python
# Task: Implement RMSD calculation
# Priority: Medium
# Dependencies: None
# Expected completion: 0.5 days

def calculate_rmsd(pred_coords, true_coords, mask=None):
    """Calculate RMSD after optimal superposition."""
    # Center coordinates
    # Apply Kabsch alignment
    # Calculate RMSD
    # Handle masks for partial structures
```

### 3.3 Validation Data Management

#### 3.3.1 Mock Validation Data
```python
# Task: Create mock validation data for different RNA types
# Priority: Medium
# Dependencies: None
# Expected completion: 1 day

def generate_mock_validation_data(num_short=2, num_medium=3, num_long=2):
    """Generate mock validation data for different RNA lengths."""
    # Create short sequences (<50nt)
    # Create medium sequences (50-150nt)
    # Create long sequences (>150nt)
    # Add varied secondary structure patterns
    # Assign mock coordinates and features
```

#### 3.3.2 Real Validation Subset
```python
# Task: Extract small validation subset from real data
# Priority: Low (dependent on data availability)
# Dependencies: None
# Expected completion: 1 day

def create_validation_subsets(sequences_df, labels_df, temporal_cutoff="2022-05-27"):
    """Create stratified validation subsets while respecting temporal cutoff."""
    # Implement as in section 2.3 of validation strategy document
    # Filter by temporal cutoff
    # Group by sequence length
    # Select diverse set
```

## 4. Testing Approach

### 4.1 Tiered Validation Implementation

We will implement the tiered validation approach from section 5.1:

#### Tier 1: Fast Technical Validation
- Use 3-5 sequences (short, medium, long)
- Focus on shape checks, loss values, basic RMSD
- Target runtime: <5 minutes
- Run after significant code changes

```python
def run_tier1_validation(model_path=None):
    """Run fast technical validation to verify code functionality."""
    # Load or create model
    # Load 3-5 test sequences
    # Run inference
    # Calculate basic metrics
    # Return validation summary
```

#### Tier 2: Intermediate Scientific Validation
- Implement only after Tier 1 is stable
- Use 10-15 sequences (balanced distribution)
- Focus on TM-score, RMSD, confidence correlation
- Target runtime: 15-30 minutes

### 4.2 Continuous Testing Strategy

- Run unit tests for individual components
- Run integration tests after component changes
- Run Tier 1 validation after pipeline changes
- Automate tracking of results in simple CSV or JSON file

## 5. Resource Management

### 5.1 Validation Efficiency

- Implement validation data caching
- Use appropriate batch sizes based on sequence length
- Support CPU fallback for environments without GPU
- Optimize metric calculations for performance

### 5.2 Experiment Tracking

Implement simple experiment tracking:

```python
def log_validation_results(version, metrics, config, notes=""):
    """Log validation results to a central file."""
    # Create metadata dictionary
    metadata = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_config": config,
        "results": metrics,
        "notes": notes
    }
    
    # Save to file
    with open(f"validation_results_{version}.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

## 6. Timeline and Deliverables

### 6.1 Week 1: Pipeline Verification
- End-to-end pipeline test with mock data
- Loss function stability fixes
- Basic validation notebook template

### 6.2 Week 2: Validation Framework
- TM-score and RMSD implementation
- Tier 1 validation script
- Mock validation data generation
- Version tracking system

### 6.3 Week 3: Real Data Validation
- Small real data validation subset
- Updated validation notebook with real data
- Performance baseline establishment
- Initial V1 model validation

## 7. Success Criteria for V1

Based on section 6.1 of the validation strategy:

1. **Technical Requirements**
   - All V1 components implemented and tested
   - End-to-end pipeline functioning with mock and real data
   - No memory issues or numerical instabilities
   - Core functionality verified on different sequence lengths

2. **Performance Targets** (Long-term)
   - Mean TM-score >0.4 on scientific validation set
   - Per-residue confidence correlation >0.5
   - Or: Plateau after multiple iterations

## 8. Conclusion

This implementation-focused plan shifts resources from extensive documentation to functional validation and testing. By prioritizing the end-to-end pipeline verification and implementing the tiered validation approach, we can accelerate the development of a working V1 model while maintaining sufficient quality control and experiment tracking.

The testing instance will focus on implementing the technical validation framework, fixing critical issues, and establishing the core metrics for structure evaluation. This approach aligns with the project's immediate goal of developing a functional V1 model while laying groundwork for more sophisticated validation in future versions.