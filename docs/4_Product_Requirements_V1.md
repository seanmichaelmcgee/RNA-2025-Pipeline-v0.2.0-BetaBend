# Product Requirements Document (PRD): RNA Folding Model - V1 Architecture Implementation

**Version:** 1.2
**Date:** 2024-04-09 (Updated for Path Parameterization & Architecture vApr9)
**Author:** AI Assistant (based on user input & Revised Architecture Plan Apr 9)

## 1. Introduction

This document outlines the requirements for the initial implementation phase (V1) of the RNA 3D Folding project's core machine learning components, based on the architecture defined in `Revised-architecture-plan-Apr9.md`. This phase focuses on establishing a robust PyTorch data loading pipeline and implementing the foundational Transformer-based fusion architecture, including auxiliary prediction heads and simplified update mechanisms. The goal is to create a runnable, testable system capable of processing precomputed features and generating initial 3D coordinate predictions, serving as the base for subsequent refinement and optimization towards the Kaggle competition goal.

## 2. Goals

*   Implement a PyTorch `Dataset` and `DataLoader` capable of loading, preprocessing, and batching precomputed RNA features (`6.Feature-Specification...`) and corresponding labels.
*   Implement a PyTorch `nn.Module` representing the V1 fusion model architecture (`Revised-architecture-plan-Apr9.md`), including embeddings, Transformer backbone (with standard attention and simplified pair updates), placeholder structure module, and auxiliary heads.
*   Define and implement V1 loss functions (simplified FAPE proxy, confidence loss, auxiliary angle prediction loss).
*   Ensure the implementation adheres to project design principles (Docker, reproducibility, **modularity**, **strict path parameterization**, configurability) outlined in `1.Proj-Cont-and-Setup...`, critical for **Kaggle Notebook compatibility**.
*   Provide a foundation for integrating a full training loop and more complex components (e.g., functional IPA module, refined losses).

## 3. Scope

### In Scope (V1 Implementation)

*   **Data Loading:** (Requirements DL-01 to DL-08)
    *   `RNADataset` implementation loading features specified in `6.Feature-Specification...` (.npz files: dihedral, thermo, evolutionary MI).
    *   Loading sequence data and ground truth C1' coordinates.
    *   Temporal cutoff logic for train/validation splits.
    *   `collate_fn` handling variable sequence lengths via padding and masking.
    *   Support for `DistributedSampler` integration.
*   **Model Architecture (V1 - Based on `Revised-architecture-plan-Apr9.md`):** (Requirements MA-01 to MA-11)
    *   Input embedding layers (sequence, positional, relative positional).
    *   Linear projection layers for combined input features (residue & pair).
    *   Implementation of `TransformerBlock` with standard MHA and simplified pair update mechanism.
    *   Stacking multiple `TransformerBlock` instances (initially scaled-down number/dims).
    *   **Placeholder** implementation for the Structure Module (`IPAModule` stub predicting coordinates linearly).
    *   Output head for predicted 3D coordinates (C1' atoms).
    *   Output head for predicted per-residue confidence scores.
    *   **Auxiliary Output Head** for predicting pseudo-dihedral angles (eta, theta) via multi-task learning.
    *   Model configuration driven by `config/default_config.yaml`.
*   **Loss Functions (V1 - Basic Proxies & Aux):** (Requirements LF-01 to LF-04)
    *   Implementation of a simplified coordinate loss (`compute_fape_loss` proxy: clamped L2).
    *   Implementation of a confidence prediction loss (`compute_confidence_loss`: MSE/BCE vs. derived lDDT proxy).
    *   Implementation of an auxiliary pseudo-dihedral angle prediction loss (`compute_angle_loss`).
*   Adherence to Non-Functional Requirements (NF-01 to NF-08).

### Out of Scope (for this V1 phase)

*   Full, functional implementation of the Invariant Point Attention (IPA) module.
*   Full implementation of pair-bias injection into attention mechanism.
*   Full implementation of Triangle Attention/Multiplication pair updates.
*   Implementation of the complete training and validation loop (`train.py`).
*   Integration and execution of the official TM-score evaluation during training/validation.
*   Advanced/refined loss implementations (full FAPE, accurate lDDT targets).
*   Teacher-Student distillation pipeline.
*   Hyperparameter optimization beyond initial scaling.
*   Inference script (`predict.py`) finalization, including the 5-prediction generation strategy and exact **Kaggle `submission.csv` formatting**.
*   Detailed performance profiling and memory optimization (beyond basic checks for VRAM).
*   Multi-GPU training setup and testing.

## 4. Target Users

*   Machine Learning Engineer/Developer implementing and training the V1 model.
*   Future team members extending/maintaining the codebase.

## 5. Requirements

### 5.1. Data Loading (`src/data_loading.py`)

| ID    | Requirement                                                                                                | Priority | Verification Method              | Notes                                             |
| :---- | :--------------------------------------------------------------------------------------------------------- | :------- | :------------------------------- | :------------------------------------------------ |
| DL-01 | Implement `RNADataset` class inheriting from `torch.utils.data.Dataset`.                                   | Must     | Code review, Unit tests          |                                                   |
| DL-02 | Constructor accepts `sequences_csv_path`, `labels_csv_path`, `features_dir`, `temporal_cutoff` arguments.    | Must     | Unit tests (instantiation)       | Paths must be arguments                           |
| DL-03 | Implement logic to filter training sequences based on `temporal_cutoff` in `train_sequences.csv`.            | Must     | Unit tests (cutoff logic)        | Use `pd.to_datetime`                              |
| DL-04 | Use `validation_sequences.csv` and `validation_labels.csv` entirely when `use_validation_set` is True.       | Must     | Unit tests (validation mode)     | Ignore `temporal_cutoff`                          |
| DL-05 | Implement `__len__` method returning the number of eligible samples.                                         | Must     | Unit tests                       |                                                   |
| DL-06 | Implement `__getitem__` method:                                                                              | Must     |                                  |                                                   |
| DL-06a| \- Loads sequence string for the given index.                                                                | Must     | Unit tests                       |                                                   |
| DL-06b| \- Loads precomputed features (`dihedral`, `thermo`, `evolutionary`) from `.npz` files in `features_dir`.    | Must     | Unit tests (shapes, types)       | Use helper `load_precomputed_features`          |
| DL-06c| \- Handles missing feature files gracefully (e.g., warning + zero tensor or error during debug).           | Should   | Test case (missing files)        | Return default zero tensors of correct shape      |
| DL-06d| \- Loads ground truth C1' coordinates (`x_1, y_1, z_1`) from `labels_csv` using helper `load_coordinates`. | Must     | Unit tests (coords loading)      |                                                   |
| DL-06e| \- Performs basic consistency checks (e.g., sequence length vs. coordinate length vs. feature lengths).    | Must     | Test case (inconsistent data)    | Raise error or log warning                        |
| DL-06f| \- Converts all loaded data into appropriately typed PyTorch tensors.                                        | Must     | Unit tests (output dict types)   | `float32` for features/coords, `long` for seq_int |
| DL-07 | Implement `collate_fn` function for batching:                                                                | Must     |                                  |                                                   |
| DL-07a| \- Identifies the maximum sequence length (`max_len`) within a batch.                                      | Must     | Unit tests (variable lengths)    |                                                   |
| DL-07b| \- Pads all sequence-length-dependent tensors (1D, 2D-N, 2D-NxN) to `max_len` using `F.pad`.                | Must     | Unit tests (batch shapes, padding) | Pad value = 0                                     |
| DL-07c| \- Generates a boolean attention mask tensor (`mask`, shape `(B, L)`) indicating valid positions.            | Must     | Unit tests (mask correctness)    | `True` for valid, `False` for padded            |
| DL-07d| \- Stacks all tensors correctly into batch dimension using `torch.stack`.                                    | Must     | Unit tests (final batch structure) | Handle non-tensor items (e.g., `target_id`)     |
| DL-08 | DataLoader setup design must be compatible with `DistributedSampler` for future DDP integration.             | Must     | Code review                      | No hardcoded shuffling logic conflicting w/ sampler |

### 5.2. Model Architecture (`src/models/`)

| ID      | Requirement                                                                                                                              | Priority | Verification Method                | Notes                                                            |
| :------ | :--------------------------------------------------------------------------------------------------------------------------------------- | :------- | :--------------------------------- | :--------------------------------------------------------------- |
| MA-01   | Implement `RNAFoldingModel` class inheriting from `torch.nn.Module`.                                                                     | Must     | Code review, Instantiation test    |                                                                  |
| MA-02   | Model hyperparameters (dimensions, layers, heads, dropout) loaded from a configuration object/dict (`config`).                           | Must     | Unit test (instantiation w/ config)  | Allows easy scaling                                              |
| MA-03   | Implement input embedding layers: `SequenceEmbedding`, `PositionalEncoding`, `RelativePositionalEncoding`.                               | Must     | Unit tests (embedding shapes)      | In `src/models/embeddings.py`                                    |
| MA-04   | Implement input linear projection layers for residue features (seq, dihedral, pair status, etc.) & pair features (pair probs, MI, rel pos). | Must     | Code review, Shape checks in test  | Calculate `in_features` dimension based on concatenated inputs |
| MA-05   | Implement `TransformerBlock` module containing: LayerNorm, **standard Multi-Head Attention** (`batch_first=True`), FFN, and **simplified pair update MLP**. | Must     | Unit test (block I/O shapes)       | In `src/models/transformer_block.py`                             |
| MA-06   | The main model backbone (`RNAFoldingModel.__init__`) stacks multiple `TransformerBlock` instances using `nn.ModuleList`.                   | Must     | Code review (`__init__`)           | Number of blocks from config                                     |
| MA-07   | Implement a **placeholder** `IPAModule` that accepts residue features and outputs 3D coordinates linearly (shape `(B, L, 3)`).             | Must     | Code review, Shape checks in test  | In `src/models/ipa_module.py`. Document clearly as placeholder.  |
| MA-08   | Implement a `confidence_head` (`nn.Sequential`) projecting final residue features to a scalar confidence score per residue (shape `(B, L)`). | Must     | Code review, Shape checks in test  | Output logits for BCE/MSE loss                                   |
| MA-09   | Implement an auxiliary `angle_prediction_head` (`nn.Sequential`) projecting final residue features to predicted sin/cos eta/theta (shape `(B, L, 4)`). | Must     | Code review, Shape checks in test  | For multi-task learning                                          |
| MA-10   | `RNAFoldingModel.forward` method correctly processes a batch, passing data through all components in sequence.                           | Must     | Integration test (data->model->loss) | Embeddings -> Projections -> Backbone -> IPA-Stub -> Heads     |
| MA-11   | Model `forward` output dictionary includes keys: `"pred_coords"`, `"pred_confidence"`, and `"pred_angles"`.                              | Must     | Unit/Integration tests             | Check tensor shapes match requirements                           |

### 5.3. Loss Functions (`src/losses.py`)

| ID      | Requirement                                                                                                          | Priority | Verification Method              | Notes                                                        |
| :------ | :------------------------------------------------------------------------------------------------------------------- | :------- | :------------------------------- | :----------------------------------------------------------- |
| LF-01   | Implement `compute_fape_loss` function (**simplified proxy**: clamped L2 distance between predicted and true coords).  | Must     | Unit tests (scalar output, mask) | Clamp distance error (e.g., `torch.clamp(dist, max=10.0)`) |
| LF-02   | Implement `compute_confidence_loss` function (**proxy**: MSE or BCEWithLogitsLoss vs. derived lDDT proxy target).      | Must     | Unit tests (scalar output, mask) | Calculate target inside `torch.no_grad()`                  |
| LF-03   | Implement `compute_angle_loss` function (auxiliary loss comparing predicted vs. true sin/cos angle features).        | Must     | Unit tests (scalar output, mask) | Use `1 - F.cosine_similarity` or MSE on sin/cos values. Handle NaNs. |
| LF-04   | All loss functions must correctly ignore contributions from padded sequence positions using the input `mask`.      | Must     | Unit tests (masked inputs)       | Apply mask before reduction (sum/mean)                   |

### 5.4. Non-Functional Requirements

| ID      | Requirement                                                                                                      | Priority | Verification Method                | Notes                                                 |
| :------ | :--------------------------------------------------------------------------------------------------------------- | :------- | :--------------------------------- | :---------------------------------------------------- |
| NF-01   | Code organized according to the defined project structure (`src/`, `scripts/`, `tests/`, etc.).                  | Must     | Code review                        | Ref: `1.Proj-Cont-and-Setup...`                     |
| NF-02   | Implementation uses PyTorch as the primary deep learning framework.                                              | Must     | Code review                        |                                                       |
| NF-03   | Code includes reasonable type hints (`typing` module) and docstrings for major classes and functions.            | Should   | Code review                        | Improves maintainability                              |
| NF-04   | Design allows easy containerization via Docker (no dependencies outside `environment.yml`, uses relative imports). | Must     | Code review, Successful Docker build |                                                       |
| NF-05   | Initial V1 implementation runs on the prototype workstation GPU (RTX 4070 Ti, 16GB VRAM) with a small batch size (e.g., 1-4) without Out-Of-Memory (OOM) errors. | Must     | Integration test run, `nvidia-smi` | Start with scaled-down config params                |
| NF-06   | Adherence to Python code style guidelines (e.g., PEP 8).                                                         | Should   | Linting tool (flake8/black)      | Improves readability                                  |
| NF-07   | All code and referenced algorithms respect the knowledge cutoff date (September 18, 2024).                         | Must     | Code review, Algorithm justification | Check against `Revised-architecture-plan-Apr9.md` |
| NF-08   | **Strict Path Parameterization:** Core logic modules in `src/` must accept necessary file paths as arguments and contain **no hardcoded paths**. | **Must** | **Code review, Tests w/ varied paths** | **Critical for Kaggle/Portability**               |

## 6. Success Metrics (V1)

*   `RNADataset` correctly loads, filters, and tensorizes all specified features from `.npz` and `.csv` files.
*   `DataLoader` produces batches with correctly padded tensors (1D, 2D-N, 2D-NxN) and boolean masks.
*   `RNAFoldingModel` (V1, scaled-down) instantiates from config and performs a forward pass on a batch without crashing on the target GPU.
*   Output tensors (`pred_coords`, `pred_confidence`, `pred_angles`) have the expected shapes (`B,L,3`, `B,L`, `B,L,4` respectively).
*   Loss functions (`fape_proxy`, `confidence_proxy`, `angle_aux`) compute scalar, non-negative values and respect padding masks.
*   Unit tests for implemented `src/` components pass.
*   Basic integration test (data -> model -> loss) runs successfully.
*   Code adheres to structural and path parameterization requirements.

## 7. Future Considerations (Post V1)

*   Replace placeholder IPA with a functional implementation.
*   Develop the full training loop (`scripts/train.py`) including optimization, scheduling, logging, and validation steps.
*   Refine loss functions (implement full FAPE, calculate accurate lDDT targets).
*   Integrate official TM-score calculation (e.g., using US-align via subprocess) for validation.
*   Implement the 5-prediction generation strategy for Kaggle submission (`scripts/predict.py`).
*   Scale up model parameters (layers, dimensions) and perform hyperparameter tuning.
*   Consider architectural enhancements (pair bias, triangle updates).
*   Evaluate Teacher-Student distillation as an alternative to multi-task angle prediction.
*   Implement multi-GPU training (DDP) if performance requires it.
*   Implement performance optimizations (mixed-precision, gradient checkpointing).
