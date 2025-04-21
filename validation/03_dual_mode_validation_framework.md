# Dual-Mode RNA Structure Validation Framework

## 1. Overview: Addressing the Validation Challenge

The RNA 3D structure prediction model faces a critical methodological challenge: **feature availability mismatch between training and testing**. During training, we have three feature types (thermodynamic, MI matrices, pseudo-dihedral angles), but at test time, we only have two (thermodynamic and MI matrices).

This document provides a comprehensive implementation strategy for a **dual-mode validation framework** that resolves this challenge while maintaining scientific rigor and development efficiency.

## 2. Dual-Mode Validation Framework

### 2.1 Core Concept

We implement two validation modes that run in parallel:

1. **Test-Equivalent Mode**: Uses only features available during Kaggle testing (thermodynamic + MI matrices)
2. **Training-Equivalent Mode**: Uses all features available during training (including pseudo-dihedral angles)

This approach provides both realistic leaderboard performance estimation and validation of our training methodology.

### 2.2 Implementation

```python
# validation/framework.py

class ValidationRunner:
    """
    Runs validation in two modes: test-equivalent and training-equivalent.
    
    Test-equivalent mode excludes pseudo-dihedral angles to match Kaggle test conditions.
    Training-equivalent mode includes all features to validate auxiliary learning.
    """
    def __init__(self, model, data_dir, config):
        self.model = model
        self.data_dir = data_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def run_validation(self, subset_name='technical', run_both_modes=True):
        """
        Run validation in one or both modes.
        
        Args:
            subset_name: Which validation subset to use ('technical', 'scientific', or 'comprehensive')
            run_both_modes: Whether to run both test and training mode validation
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Always run test-equivalent mode (most important for Kaggle estimation)
        test_mode_results = self.run_test_equivalent_mode(subset_name)
        results['test_mode'] = test_mode_results
        
        # Optionally run training-equivalent mode
        if run_both_modes:
            train_mode_results = self.run_training_equivalent_mode(subset_name)
            results['train_mode'] = train_mode_results
            
            # Calculate performance difference
            results['analysis'] = self.analyze_mode_differences(
                test_mode_results, 
                train_mode_results
            )
            
        return results
        
    def run_test_equivalent_mode(self, subset_name='technical'):
        """Run validation using only test-available features (no pseudo-dihedrals)."""
        dataset = self._create_dataset(subset_name, test_mode=True)
        dataloader = self._create_dataloader(dataset)
        return self._evaluate_model(dataloader, 'test_equivalent')
        
    def run_training_equivalent_mode(self, subset_name='technical'):
        """Run validation using all training features (including pseudo-dihedrals)."""
        dataset = self._create_dataset(subset_name, test_mode=False)
        dataloader = self._create_dataloader(dataset)
        return self._evaluate_model(dataloader, 'training_equivalent')
    
    def _create_dataset(self, subset_name, test_mode=True):
        """Create validation dataset with appropriate feature filtering."""
        return ValidationDataset(
            data_dir=self.data_dir,
            subset_name=subset_name,
            test_mode=test_mode
        )
    
    def _create_dataloader(self, dataset):
        """Create DataLoader with appropriate batch size based on sequence lengths."""
        # Dynamically adjust batch size based on sequence length
        avg_seq_len = dataset.get_average_sequence_length()
        batch_size = self._calculate_optimal_batch_size(avg_seq_len)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
    
    def _calculate_optimal_batch_size(self, avg_seq_len):
        """Calculate optimal batch size based on sequence length to avoid OOM."""
        if avg_seq_len <= 50:
            return 16
        elif avg_seq_len <= 150:
            return 8
        else:
            return 4
    
    def _evaluate_model(self, dataloader, mode_name):
        """Evaluate model on the provided dataloader."""
        self.model.eval()
        metrics = {
            'tm_score': 0.0,
            'rmsd': 0.0,
            'confidence_correlation': 0.0,
            'per_target_metrics': {}
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(outputs, batch)
                
                # Update metrics
                for key in ['tm_score', 'rmsd', 'confidence_correlation']:
                    metrics[key] += batch_metrics[key] * len(batch['target_id'])
                
                # Store per-target metrics
                for i, target_id in enumerate(batch['target_id']):
                    metrics['per_target_metrics'][target_id] = {
                        key: batch_metrics[key][i] if isinstance(batch_metrics[key], list) else batch_metrics[key]
                        for key in batch_metrics
                    }
        
        # Average metrics
        num_samples = len(dataloader.dataset)
        for key in ['tm_score', 'rmsd', 'confidence_correlation']:
            metrics[key] /= num_samples
            
        # Add mode information
        metrics['mode'] = mode_name
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def _calculate_metrics(self, outputs, batch):
        """Calculate all validation metrics."""
        # Get predictions and ground truth
        pred_coords = outputs['pred_coords']
        true_coords = batch['atom_positions']
        pred_confidence = outputs.get('pred_confidence', None)
        
        # Calculate TM-score
        tm_score = calculate_tm_score(pred_coords, true_coords, batch['mask'])
        
        # Calculate RMSD
        rmsd = calculate_rmsd(pred_coords, true_coords, batch['mask'])
        
        # Calculate confidence correlation if available
        conf_corr = 0.0
        if pred_confidence is not None:
            conf_corr = calculate_confidence_correlation(
                pred_confidence, 
                pred_coords, 
                true_coords, 
                batch['mask']
            )
        
        return {
            'tm_score': tm_score,
            'rmsd': rmsd,
            'confidence_correlation': conf_corr
        }
    
    def analyze_mode_differences(self, test_results, train_results):
        """Analyze differences between test and training modes."""
        analysis = {}
        
        # Calculate TM-score improvement
        tm_score_delta = train_results['tm_score'] - test_results['tm_score']
        relative_improvement = (tm_score_delta / test_results['tm_score']) * 100
        
        analysis['tm_score_absolute_improvement'] = tm_score_delta
        analysis['tm_score_relative_improvement'] = f"{relative_improvement:.1f}%"
        
        # Calculate RMSD improvement
        rmsd_delta = test_results['rmsd'] - train_results['rmsd']  # Lower is better
        relative_rmsd_improvement = (rmsd_delta / test_results['rmsd']) * 100
        
        analysis['rmsd_absolute_improvement'] = rmsd_delta
        analysis['rmsd_relative_improvement'] = f"{relative_rmsd_improvement:.1f}%"
        
        # Flag significant differences (potential overreliance on dihedrals)
        analysis['significant_dihedral_dependence'] = relative_improvement > 20.0
        
        return analysis
```

## 3. Feature Loading Modifications

The core implementation of feature filtering happens in the `ValidationDataset` and `NPZFeatureLoader` classes:

```python
# validation/data_loading.py

class ValidationDataset(Dataset):
    """Dataset for validation with test/train mode switching."""
    
    def __init__(self, data_dir, subset_name='technical', test_mode=True, seed=42):
        """
        Initialize validation dataset.
        
        Args:
            data_dir: Base directory for data
            subset_name: Which validation subset to use ('technical', 'scientific', 'comprehensive')
            test_mode: If True, excludes pseudo-dihedral features to match test conditions
            seed: Random seed for subset selection
        """
        self.data_dir = data_dir
        self.subset_name = subset_name
        self.test_mode = test_mode
        
        # Setup coordinate and feature loaders
        self.coord_loader = CSVCoordinateLoader(data_dir)
        self.feature_loader = NPZFeatureLoader(data_dir, test_mode=test_mode)
        
        # Load appropriate validation subset
        self.target_ids = self._load_validation_subset(subset_name, seed)
        
    def __len__(self):
        return len(self.target_ids)
    
    def __getitem__(self, idx):
        target_id = self.target_ids[idx]
        
        # Load coordinates and features
        coordinates = self.coord_loader.get_coordinates(target_id)
        features = self.feature_loader.get_features(target_id)
        
        # Combine into a sample
        sample = {
            'target_id': target_id,
            'sequence': features['sequence'],
            'atom_positions': coordinates['atom_positions'],
            'mask': coordinates['mask']
        }
        
        # Add all available features
        for key, value in features.items():
            if key not in sample:
                sample[key] = value
        
        # Convert to tensors
        sample = self._convert_to_tensors(sample)
        
        return sample
    
    def _load_validation_subset(self, subset_name, seed):
        """Load the appropriate validation subset."""
        # Define subset sizes
        subset_sizes = {
            'technical': 5,      # Small, fast subset
            'scientific': 15,    # Medium-sized diverse subset
            'comprehensive': 30  # Large, thorough subset
        }
        
        size = subset_sizes.get(subset_name, 5)
        
        # Load all potential validation targets
        all_validation_targets = self._load_all_validation_targets()
        
        # Select a diverse subset
        subset = self._select_diverse_subset(all_validation_targets, size, seed)
        
        return subset
    
    def _select_diverse_subset(self, targets, size, seed):
        """Select a diverse subset of targets by length and structure type."""
        # Group targets by length category
        short = [t for t in targets if self._get_seq_length(t) < 50]
        medium = [t for t in targets if 50 <= self._get_seq_length(t) < 150]
        long = [t for t in targets if self._get_seq_length(t) >= 150]
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Select approximately equal numbers from each length category
        num_each = max(1, size // 3)
        
        subset = (
            random.sample(short, min(num_each, len(short))) +
            random.sample(medium, min(num_each, len(medium))) +
            random.sample(long, min(num_each, len(long)))
        )
        
        # If we need more to reach desired size, sample from all
        if len(subset) < size:
            remaining = [t for t in targets if t not in subset]
            subset += random.sample(remaining, min(size - len(subset), len(remaining)))
        
        return subset[:size]  # Ensure we don't exceed the requested size
    
    def get_average_sequence_length(self):
        """Get average sequence length in this dataset."""
        return sum(self._get_seq_length(t) for t in self.target_ids) / len(self.target_ids)


class NPZFeatureLoader:
    """Loads NPZ features with test-mode filtering."""
    
    def __init__(self, data_dir, test_mode=True):
        """
        Initialize feature loader.
        
        Args:
            data_dir: Base directory for data
            test_mode: If True, excludes pseudo-dihedral features to match test conditions
        """
        self.data_dir = data_dir
        self.test_mode = test_mode
    
    def get_features(self, target_id):
        """Get features for a target, with test mode filtering."""
        # Load all available features
        features = {}
        
        # Try to load thermodynamic features (required)
        thermo_features = self._load_thermo_features(target_id)
        if thermo_features is None:
            raise ValueError(f"Thermodynamic features missing for {target_id}")
        features.update(thermo_features)
        
        # Try to load MI features (required)
        mi_features = self._load_evolutionary_features(target_id)
        if mi_features is None:
            raise ValueError(f"Mutual Information features missing for {target_id}")
        features.update(mi_features)
        
        # Try to load dihedral features (only used in train mode)
        if not self.test_mode:
            dihedral_features = self._load_dihedral_features(target_id)
            if dihedral_features is not None:
                features.update(dihedral_features)
        
        return features
    
    def _load_thermo_features(self, target_id):
        """Load thermodynamic features from NPZ file."""
        # Implementation details omitted for brevity
        # This would load pairing_probs, positional_entropy, etc.
        pass
    
    def _load_evolutionary_features(self, target_id):
        """Load evolutionary (MI) features from NPZ file."""
        # Implementation details omitted for brevity
        # This would load coupling_matrix, etc.
        pass
    
    def _load_dihedral_features(self, target_id):
        """Load pseudo-dihedral features from NPZ file."""
        # Implementation details omitted for brevity
        # This would load the features array with sin/cos encodings
        pass
```

## 4. Validation Subset Creation

Create three distinct validation subsets for different purposes:

```python
# validation/subset_selection.py

def create_validation_subsets(data_dir, output_dir, temporal_cutoff="2022-05-27"):
    """
    Create validation subsets for different validation tiers.
    
    Creates three subsets:
    1. Technical (5 sequences): For quick technical validation
    2. Scientific (15 sequences): For scientific method validation
    3. Comprehensive (30 sequences): For thorough model evaluation
    
    Args:
        data_dir: Directory containing sequence and label data
        output_dir: Directory to save subset files
        temporal_cutoff: Only use sequences before this date
    """
    # Load sequences and filter by temporal cutoff
    sequences_df = pd.read_csv(os.path.join(data_dir, "train_sequences.csv"))
    sequences_df['temporal_cutoff'] = pd.to_datetime(sequences_df['temporal_cutoff'])
    cutoff_date = pd.to_datetime(temporal_cutoff)
    
    # Filter by cutoff date
    valid_sequences = sequences_df[sequences_df['temporal_cutoff'] < cutoff_date]
    
    # Group by sequence length
    short = valid_sequences[valid_sequences['sequence'].str.len() < 50]
    medium = valid_sequences[(valid_sequences['sequence'].str.len() >= 50) & 
                             (valid_sequences['sequence'].str.len() < 150)]
    long = valid_sequences[valid_sequences['sequence'].str.len() >= 150]
    
    # Create subsets
    technical_subset = select_diverse_subset(short, medium, long, 5)
    scientific_subset = select_diverse_subset(short, medium, long, 15)
    comprehensive_subset = select_diverse_subset(short, medium, long, 30)
    
    # Save subsets
    os.makedirs(output_dir, exist_ok=True)
    technical_subset.to_csv(os.path.join(output_dir, "technical_subset.csv"), index=False)
    scientific_subset.to_csv(os.path.join(output_dir, "scientific_subset.csv"), index=False)
    comprehensive_subset.to_csv(os.path.join(output_dir, "comprehensive_subset.csv"), index=False)
    
    print(f"Created validation subsets in {output_dir}")
    print(f"Technical: {len(technical_subset)} sequences")
    print(f"Scientific: {len(scientific_subset)} sequences")
    print(f"Comprehensive: {len(comprehensive_subset)} sequences")
    
def select_diverse_subset(short_df, medium_df, long_df, total_size):
    """Select a diverse subset with balanced length distribution."""
    num_each = max(1, total_size // 3)
    
    # Sample from each length category
    short_sample = short_df.sample(min(num_each, len(short_df)))
    medium_sample = medium_df.sample(min(num_each, len(medium_df)))
    long_sample = long_df.sample(min(num_each, len(long_df)))
    
    # Combine samples
    combined = pd.concat([short_sample, medium_sample, long_sample])
    
    # If we need more, sample from all
    if len(combined) < total_size:
        remaining = pd.concat([short_df, medium_df, long_df]).drop(combined.index)
        additional = remaining.sample(min(total_size - len(combined), len(remaining)))
        combined = pd.concat([combined, additional])
    
    return combined.iloc[:total_size]  # Ensure we don't exceed the requested size
```

## 5. Performance Comparison Implementation

Create a module to track and visualize the differences between test-mode and train-mode performance:

```python
# validation/performance_analysis.py

def analyze_performance(results_file):
    """
    Analyze performance differences between test and train modes.
    
    Args:
        results_file: Path to validation results JSON file
        
    Returns:
        None (displays analysis and creates plots)
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    test_metrics = results['test_mode']
    train_metrics = results.get('train_mode')
    
    # Display test mode metrics (Kaggle estimation)
    print(f"Test-Mode Metrics (Kaggle Estimation):")
    print(f"TM-Score: {test_metrics['tm_score']:.4f}")
    print(f"RMSD: {test_metrics['rmsd']:.4f}")
    print(f"Confidence Correlation: {test_metrics['confidence_correlation']:.4f}")
    
    # If we have train mode metrics, compare them
    if train_metrics:
        print("\nTrain-Mode Metrics (With Pseudo-Dihedrals):")
        print(f"TM-Score: {train_metrics['tm_score']:.4f}")
        print(f"RMSD: {train_metrics['rmsd']:.4f}")
        print(f"Confidence Correlation: {train_metrics['confidence_correlation']:.4f}")
        
        # Calculate improvements
        tm_improvement = train_metrics['tm_score'] - test_metrics['tm_score']
        tm_relative = (tm_improvement / test_metrics['tm_score']) * 100
        
        rmsd_improvement = test_metrics['rmsd'] - train_metrics['rmsd']  # Lower is better
        rmsd_relative = (rmsd_improvement / test_metrics['rmsd']) * 100
        
        print("\nPerformance Differences:")
        print(f"TM-Score Improvement: +{tm_improvement:.4f} ({tm_relative:.1f}%)")
        print(f"RMSD Improvement: +{rmsd_improvement:.4f} ({rmsd_relative:.1f}%)")
        
        # Scientific interpretation
        print("\nScientific Interpretation:")
        if tm_relative > 20.0:
            print("⚠️ WARNING: Model shows strong dependence on pseudo-dihedral angles.")
            print("   Consider teacher-student model to distill this knowledge.")
        elif tm_relative > 10.0:
            print("ℹ️ Model shows moderate dependence on pseudo-dihedral angles.")
            print("   Multi-task learning appears to be working well.")
        else:
            print("✓ Model shows minimal dependence on pseudo-dihedral angles.")
            print("   Performance should translate well to the test set.")
    
        # Generate visualization
        _generate_comparison_plot(test_metrics, train_metrics, results_file)
    
def _generate_comparison_plot(test_metrics, train_metrics, results_file):
    """Generate a comparison plot of per-target metrics."""
    # Implementation details omitted for brevity
    pass
```

## 6. Early Sharing Prize Development Timeline

Here's a practical timeline for maximizing your chances at the early sharing prize:

| Week | Primary Focus | Validation Approach | Expected Outcomes |
|------|---------------|---------------------|------------------|
| 1: Foundation | Implement data pipeline and model architecture | **Test-mode only** on Technical Subset (5 sequences) | Working end-to-end pipeline with basic structure prediction capability |
| 2: Core Training | Implement loss functions and basic training | **Both modes** on Technical Subset | Understanding of dihedral impact; model learns basic structural patterns |
| 3: Performance Tuning | Optimize hyperparameters; refine loss weights | **Both modes** on Scientific Subset (15 sequences) | Improved TM-score; quantified benefit of auxiliary learning |
| 4: Submission Prep | Finalize model; generate Kaggle submission | **Test-mode** on Comprehensive Subset (30 sequences) | Reliable leaderboard score estimation; submission-ready model |

### 6.1 Implementation Plan by Week

#### Week 1: Foundation
- **Monday**: Implement `ValidationDataset` with dual-mode support
- **Tuesday**: Create validation subsets and feature loading logic
- **Wednesday**: Integrate model with validation pipeline
- **Thursday**: Run first end-to-end test-mode validation
- **Friday**: Debug and fix any pipeline issues

#### Week 2: Core Training
- **Monday**: Implement training loop with loss functions
- **Tuesday**: Add auxiliary angle prediction loss
- **Wednesday**: Compare test/train modes on Technical Subset
- **Thursday**: Refine model based on validation insights
- **Friday**: Implement model checkpointing and version tracking

#### Week 3: Performance Tuning
- **Monday**: Extend validation to Scientific Subset
- **Tuesday-Wednesday**: Hyperparameter optimization
- **Thursday**: Loss weights optimization
- **Friday**: Performance analysis across model versions

#### Week 4: Submission Prep
- **Monday**: Run comprehensive validation
- **Tuesday**: Final model selection and ensemble creation
- **Wednesday**: Kaggle submission notebook preparation
- **Thursday**: Performance verification and documentation
- **Friday**: Submit to Kaggle for early sharing prize

## 7. Version Tracking Implementation

Track performance across versions with automatic logging:

```python
# validation/version_tracking.py

def save_validation_results(version, results, config, notes=""):
    """
    Save validation results with version tracking.
    
    Args:
        version: Version string (e.g., "0.1.2")
        results: Validation results dictionary
        config: Model configuration
        notes: Optional notes about this version
    """
    # Create metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "config": config,
        "notes": notes
    }
    
    # Create directory if needed
    os.makedirs("experiments/validation_results", exist_ok=True)
    
    # Save to JSON file
    file_path = f"experiments/validation_results/v{version}_results.json"
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved validation results to {file_path}")
    
    # Generate performance summary
    performance_summary = {
        "version": version,
        "tm_score_test": results["test_mode"]["tm_score"],
        "rmsd_test": results["test_mode"]["rmsd"],
        "timestamp": metadata["timestamp"],
        "notes": notes
    }
    
    if "train_mode" in results:
        performance_summary["tm_score_train"] = results["train_mode"]["tm_score"]
        performance_summary["rmsd_train"] = results["train_mode"]["rmsd"]
        
        # Calculate improvement
        tm_delta = results["train_mode"]["tm_score"] - results["test_mode"]["tm_score"]
        performance_summary["tm_score_improvement"] = tm_delta
    
    # Append to summary CSV
    summary_path = "experiments/validation_results/performance_summary.csv"
    if not os.path.exists(summary_path):
        # Create with headers
        with open(summary_path, 'w') as f:
            headers = ["version", "timestamp", "tm_score_test", "rmsd_test"]
            if "train_mode" in results:
                headers.extend(["tm_score_train", "rmsd_train", "tm_score_improvement"])
            headers.append("notes")
            f.write(",".join(headers) + "\n")
    
    # Append this version's data
    with open(summary_path, 'a') as f:
        values = [
            version,
            metadata["timestamp"],
            f"{results['test_mode']['tm_score']:.4f}",
            f"{results['test_mode']['rmsd']:.4f}"
        ]
        
        if "train_mode" in results:
            values.extend([
                f"{results['train_mode']['tm_score']:.4f}",
                f"{results['train_mode']['rmsd']:.4f}",
                f"{performance_summary.get('tm_score_improvement', 0):.4f}"
            ])
        
        values.append(notes.replace(",", ";"))
        f.write(",".join(values) + "\n")
```

## 8. Post-Early-Sharing-Prize Enhancements

After securing the early sharing prize, enhance your validation framework with:

### 8.1 K-Fold Cross Validation

```python
# validation/advanced/cross_validation.py

def k_fold_cross_validation(model_class, config, data_dir, k=5, temporal_stratified=True):
    """
    Perform k-fold cross validation with temporal stratification.
    
    Args:
        model_class: Model class to instantiate
        config: Model configuration
        data_dir: Data directory
        k: Number of folds
        temporal_stratified: Whether to use temporal stratification
    
    Returns:
        Dictionary of cross-validation results
    """
    # Implementation steps:
    # 1. Load all sequences and sort by temporal_cutoff
    # 2. Create k folds with temporal stratification
    # 3. For each fold:
    #    a. Train on k-1 folds
    #    b. Validate on the held-out fold
    #    c. Run both test-mode and train-mode validation
    # 4. Aggregate results across folds
```

### 8.2 Ablation Studies

```python
# validation/advanced/ablation.py

def run_feature_ablation_study(model_class, config, data_dir):
    """
    Perform ablation study to determine feature importance.
    
    Tests model performance with different feature combinations:
    - All features
    - Only thermodynamic features
    - Only MI matrix features
    - Thermodynamic + MI (test-mode)
    
    Args:
        model_class: Model class to instantiate
        config: Base model configuration
        data_dir: Data directory
    
    Returns:
        Dictionary of ablation results
    """
    # Implementation details omitted for brevity
```

### 8.3 Teacher-Student Evaluation

```python
# validation/advanced/teacher_student.py

def evaluate_teacher_student_distillation(teacher_model, student_model, data_dir):
    """
    Evaluate teacher-student distillation effectiveness.
    
    Compares:
    1. Teacher model (with pseudo-dihedrals)
    2. Student model (without pseudo-dihedrals)
    3. Direct model (without pseudo-dihedrals, no distillation)
    
    Args:
        teacher_model: Model with access to all features
        student_model: Model trained through distillation
        data_dir: Data directory
        
    Returns:
        Comparative analysis of all three approaches
    """
    # Implementation details omitted for brevity
```

## 9. Key Scientific Insights & Recommendations

1. **Dual-Mode Measurement is Crucial**
   - Measuring both test-mode and train-mode performance quantifies the value of pseudo-dihedral information
   - The gap between modes indicates how effectively your model leverages auxiliary supervision
   - A large gap signals potential leaderboard disappointment; a small gap suggests robustness

2. **Feature Importance Hierarchy**
   - Thermodynamic features provide essential secondary structure information
   - MI matrices capture critical long-range interactions
   - Pseudo-dihedral angles offer direct backbone geometry guidance
   - Through ablation studies, quantify the contribution of each feature type

3. **Architectural Priorities for V1**
   - Prioritize robust thermodynamic and MI feature processing
   - Ensure multi-task learning optimally balances coordinate and angle prediction
   - Focus first on test-mode performance as this translates to Kaggle success

4. **Long-Term Scientific Value**
   - The dual-mode framework provides insights beyond the competition:
   - Measures structural information content in different feature types
   - Quantifies the model's capacity to learn structural patterns
   - Informs future developments in RNA 3D structure prediction methodology

5. **Development Strategy**
   - Begin with test-mode validation only (fastest path to working model)
   - Add train-mode validation to measure auxiliary learning effectiveness
   - Use the gap between modes to guide teacher-student model development
   - Document dual-mode validation in your Kaggle submission for scientific credibility

## 10. Implementation Checklist

To implement this framework, complete these steps:

- [ ] Create validation directory structure (`validation/`)
- [ ] Implement `NPZFeatureLoader` with test-mode filtering
- [ ] Implement `ValidationDataset` with dual-mode support
- [ ] Create validation subset selection functions
- [ ] Implement `ValidationRunner` with test/train mode execution
- [ ] Add performance analysis and visualization tooling
- [ ] Create version tracking and experiment documentation system
- [ ] Integrate with model training workflow
- [ ] Create validation notebooks for each tier
- [ ] Test entire pipeline on a small dataset

## Conclusion

This dual-mode validation framework addresses the fundamental scientific challenge of feature availability mismatch between training and testing while supporting efficient development toward the early sharing prize. By systematically measuring the impact of pseudo-dihedral features on model performance, it provides both practical guidance for model development and valuable scientific insights into RNA structure prediction.

The implementation plan provided here offers a clear path forward with specific code scaffolding, a practical timeline, and prioritized enhancements for post-competition scientific rigor.
