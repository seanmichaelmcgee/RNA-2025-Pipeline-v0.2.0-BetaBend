# Roadmap: RNA Folding - V1 Foundational Build

**Version:** 1.3
**Date:** 2024-04-09
**Purpose:** This document outlines the high-level development plan and milestones for implementing Version 1 (V1) of the PyTorch-based RNA 3D structure prediction pipeline. V1 focuses on establishing the core data loading infrastructure and a functional, albeit simplified, version of the target architecture defined in `3_Architecture_Specification.md`. This provides a testable foundation compatible with our Dockerized workflow (`1_Context_and_Setup.md`) and prepares for subsequent V2+ iterations aiming for full Kaggle competition readiness.

## 1. V1 Goal

The primary goal of V1 is to create a **runnable end-to-end pipeline** (data loading -> model forward pass -> loss calculation) based on the foundational architecture. This includes:
*   A working PyTorch `Dataset` and `DataLoader` for precomputed features (`2_Feature_Specification.md`).
*   A V1 `RNAFoldingModel` implementing the core Transformer structure with scaled-down parameters, simplified pair updates, a placeholder structure module (IPA stub), and auxiliary heads (confidence, angle prediction) as specified in `4_Product_Requirements_V1.md`.
*   Basic proxy loss functions (FAPE, confidence, angle).
*   Verification through unit tests and a basic integration test demonstrating data flow.
*   Strict adherence to modularity and path parameterization principles for Kaggle compatibility.

## 2. High-Level Pipeline Overview (V1 Implementation)

1.  **Data Ingestion & Feature Loading (`src/data_loading.py`):** Implement `Dataset` and `DataLoader` to load sequences, labels, and precomputed `.npz` features, handling padding/masking via `collate_fn`. Apply temporal cutoff logic.
2.  **Feature Embedding (`src/models/embeddings.py`):** Implement modules for sequence, positional, and relative positional embeddings.
3.  **Fusion Backbone (V1) (`src/models/transformer_block.py`, `src/models/rna_folding_model.py`):** Implement scaled-down Transformer backbone with standard Multi-Head Attention and simplified pair updates.
4.  **Structure Module (Placeholder) (`src/models/ipa_module.py`):** Implement a simple linear layer predicting coordinates as a placeholder for the full IPA module.
5.  **Output Heads & Loss Calculation (`src/models/rna_folding_model.py`, `src/losses.py`):** Implement heads for coordinate, confidence, and auxiliary angle prediction. Implement V1 proxy loss functions.
6.  **Orchestration (Basic Test):** Create a test script/notebook (`scripts/test_pipeline.py` or similar) to verify the flow from data loading through loss calculation.

*(Full training and prediction scripts (`train.py`, `predict.py`) are part of V2+ development).*

## 3. Estimated Timeline & Milestones (V1 Foundational Build: ~2-3 Weeks)

This timeline assumes focused effort (a few hours per day) following the detailed steps in `6_Tactical_Plan_V1.md`.

| **Week**       | **Focus**                                                                                  | **Key Deliverables**                                                                                                   | **PRD Sections**    |
| :------------- | :----------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :------------------ |
| **Week 1**     | Setup, Data Loading Pipeline, Embeddings                                                   | - Docker env stable & verified <br/> - `RNADataset`, `collate_fn` implemented & unit-tested <br/> - Embedding modules implemented & unit-tested | DL-01 to DL-08, MA-03 |
| **Week 2**     | Model Backbone (V1), Placeholders, Auxiliary Heads                                         | - `TransformerBlock` (V1) implemented & unit-tested <br/> - `IPAModule` placeholder functional <br/> - `RNAFoldingModel` (V1) structure implemented <br/> - Confidence & Angle heads implemented & unit-tested | MA-01, 02, 04-11    |
| **Week 3**     | V1 Loss Functions, Integration Testing                                                     | - V1 Loss functions (FAPE proxy, Conf proxy, Angle aux) implemented & unit-tested <br/> - Basic Integration test (data->model->loss) runs successfully on target hardware (GPU) without OOM errors (small batch). <br/> - Code reviewed for modularity & path parameterization. | LF-01 to LF-04, NF-* |

## 4. Transition to V2 and Beyond

*   **V1 Completion:** Successful execution of the V1 integration test marks the completion of this foundational phase.
*   **V2 Planning:** Following V1 completion, a review will inform the specific requirements and tactical plan for V2. Key V2+ goals include:
    *   Implementing the full training loop (`scripts/train.py`) with optimization, logging, and validation.
    *   Replacing the `IPAModule` placeholder with a functional implementation.
    *   Refining loss functions (implementing full FAPE, accurate lDDT targets).
    *   Integrating TM-score calculation for validation.
    *   Implementing the 5-prediction strategy and finalizing `scripts/predict.py` for Kaggle submission format.
    *   Scaling up model parameters (layers, dimensions) based on performance and hardware capabilities.
    *   Potentially integrating more advanced architectural features (pair bias, triangle updates).
*   **Multi-GPU:** Implementation of multi-GPU training (DDP) is deferred. It will only be considered in later phases (V3+) **if** single-GPU training performance proves to be a significant bottleneck preventing necessary experimentation within the competition timeline. The V1/V2 design ensures readiness but avoids the upfront complexity.

## 5. Alignment with Kaggle

*   The focus on Docker, pinned dependencies (`environment.yml`), modularity (`src/` vs. `scripts/`), and strict path parameterization in V1 is explicitly designed to minimize friction when creating the final Kaggle Notebook submission.
*   While V1 does not produce the final `submission.csv`, it builds the core, reusable components (`src/`) that will be orchestrated within the Kaggle environment.

## Conclusion

This V1 roadmap focuses on rapidly building a working, testable foundation for the RNA folding pipeline using PyTorch. By starting with simplified components and scaled-down parameters within a reproducible Dockerized environment, we establish a solid base that adheres to architectural plans and Kaggle compatibility requirements. This allows for iterative refinement and scaling in subsequent V2+ phases towards a competitive final model.
