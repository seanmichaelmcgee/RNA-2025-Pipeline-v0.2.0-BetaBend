# Tactical Implementation Guide: RNA Folding - V1 Architecture (v1.3)

**Objective:** Implement the foundational V1 PyTorch data loading pipeline and Transformer-based fusion model architecture, as defined in `Revised-architecture-plan-Apr9.md` and `8.Product-Reqs-v1.2.md`. Includes multi-task angle prediction and starts with scaled-down parameters.

**Phase:** `Dev` Environment (Local Workstation, RTX 4070 Ti / 16GB VRAM focus)

**IMPORTANT NOTE:** Throughout implementation within `src/` modules, strictly adhere to `10.RULES-v3-for-kaggle-focus.md`:
*   **Rule 2.4 (Modularity):** Keep core logic in `src/`, use `scripts/` or notebooks for orchestration only.
*   **Rule 7.2 (Path Parameterization):** Pass all file/directory paths needed by `src/` functions/classes as arguments. **NO hardcoded paths** inside `src/`.

---

## I. Environment & Setup Verification (PRD NF-01, NF-02, NF-04)

*   `[ ]` **Verify Environment:** Activate `rna-3d-folding`. Check packages (`pytorch`, `pandas`, `numpy`, `pyyaml`) per `environment.yml`.
*   `[ ]` **Project Structure:** Ensure directories (`src`, `tests`, `data/processed`, `config`) exist as per `1.Proj-Cont-and-Setup-Apr2-25-v1.2.md`.
*   `[ ]` **Config File:** Create/update `config/default_config.yaml`. Define **scaled-down V1 parameters** first (e.g., `residue_embed_dim: 64`, `pair_embed_dim: 32`, `seq_embed_dim: 32`, `num_transformer_blocks: 4`, `num_attention_heads: 4`, `ffn_dim: 256`, `dropout: 0.1`). Include placeholder data paths (to be filled by orchestrator/args). Define `loss_weights` section (e.g., `fape: 1.0`, `confidence: 0.1`, `angle: 0.5`).

## II. Data Loading Implementation (`src/data_loading.py`) (PRD DL-01 to DL-08)

*   `[ ]` **Implement `load_coordinates` Helper:**
    *   Input: `labels_df: pd.DataFrame`, `target_id: str`.
    *   Output: `coords: np.ndarray` (`(N, 3)`, `float32`), `resnames: List[str]`. Error check.
*   `[ ]` **Implement `load_precomputed_features` Helper:**
    *   Input: `target_id: str`, `features_dir: str` (**Argument**).
    *   Action: Load `.npz` files (`dihedral`, `thermo`, `evolutionary`/`mi`). Handle errors/missing files (return default zero arrays or `None`).
    *   Output: `Dict[str, Dict[str, np.ndarray]]` or `Dict[str, Optional[Dict]]`.
*   `[ ]` **Implement `RNADataset.__init__`:**
    *   Args: `sequences_csv_path`, `labels_csv_path`, `features_dir` (**Arguments**), `temporal_cutoff`, `use_validation_set`.
    *   Action: Load sequences CSV, filter by `temporal_cutoff`. Store `target_ids`, `sequences`. Load full `labels_df`. Define `nuc_to_int`. Store `features_dir`.
*   `[ ]` **Implement `RNADataset.__len__`:**
    *   Action: Return `len(self.target_ids)`.
*   `[ ]` **Implement `RNADataset.__getitem__`:**
    *   Action: Get `target_id`, `sequence_str`. Call `load_precomputed_features(target_id, self.features_dir)`. Call `load_coordinates(self.labels_df, target_id)`.
    *   Action: Perform length consistency checks.
    *   Action: Assemble feature dictionary (`sequence_int`, `dihedral_features` (handle NaNs, length mismatch), `pairing_status`, `pairing_probs`, `mi_matrix`, `delta_g`, `coordinates`, `length`). Convert to tensors. Use defaults for missing features.
    *   Action: Wrap loading in `try...except`.
    *   Output: Dictionary of tensors + metadata.
*   `[ ]` **Implement `collate_fn`:**
    *   Input: `batch: List[Dict]`.
    *   Action: Find `max_len`. Pad 1D, 2D-N, 2D-NxN tensors using `F.pad`. Stack tensors. Generate boolean `mask`.
    *   Output: Batch dictionary.
*   `[ ]` **Write Unit Tests (`tests/test_data_loading.py`):**
    *   Test instantiation (train/val modes, cutoff).
    *   Test `__getitem__` output dict keys, shapes, types; mock helpers. Test missing feature handling.
    *   Test `collate_fn` with variable lengths; check padding, mask, shapes.
*   `[ ]` **Run Tests & Debug:** `pytest tests/test_data_loading.py`. Fix failures.
*   `[ ]` **Commit Prompt:** *AI prompts user:* "`RNADataset` and `collate_fn` implemented and tests pass. Ready to commit? Suggest: `feat(data): Implement RNADataset and collate_fn for V1 features`"

## III. Model Architecture Implementation (`src/models/`) (PRD MA-01 to MA-11)

1.  `[ ]` **Implement Embedding Layers (`src/models/embeddings.py`):** (PRD MA-03)
    *   Implement `SequenceEmbedding`.
    *   Implement `PositionalEncoding`.
    *   Implement `RelativePositionalEncoding`.
    *   Write unit tests (`tests/test_embeddings.py`) checking output shapes.
    *   **Commit Prompt:** *AI prompts user:* "Embedding layers implemented and tests pass. Ready to commit? Suggest: `feat(model): Implement sequence, positional, relative positional embeddings`"
2.  `[ ]` **Implement Transformer Block (`src/models/transformer_block.py`):** (PRD MA-05)
    *   Implement `TransformerBlock(config)`:
        *   Use **`nn.MultiheadAttention`** (`batch_first=True`).
        *   Implement residue update path.
        *   Implement **simplified pair update path** (Outer Product Prep -> MLP). Calculate MLP input dim: `2 * config['residue_embed_dim'] + config['pair_embed_dim']`.
    *   Write unit tests (`tests/test_transformer_block.py`) checking I/O shapes.
    *   **Commit Prompt:** *AI prompts user:* "`TransformerBlock` implemented and tests pass. Ready to commit? Suggest: `feat(model): Implement TransformerBlock with std MHA and simplified pair update`"
3.  `[ ]` **Implement IPA Placeholder (`src/models/ipa_module.py`):** (PRD MA-07)
    *   Implement placeholder `IPAModule(config)` with `nn.Linear` outputting `(B, L, 3)`. Document clearly.
    *   **(No dedicated commit unless complex placeholder warrants it)**
4.  `[ ]` **Implement Main Model (`src/models/rna_folding_model.py`):** (PRD MA-01, MA-02, MA-04, MA-06, MA-08, MA-09, MA-10, MA-11)
    *   Implement `RNAFoldingModel(config)`:
        *   `__init__`: Instantiate embeddings, input projections (calculate `IN_RES_DIM`, `IN_PAIR_DIM` based on `config` and features used), `nn.ModuleList` of `TransformerBlock`s, IPA placeholder, `confidence_head`, `angle_prediction_head`.
        *   `forward(batch)`: Implement data flow through components. Return dict: `"pred_coords"`, `"pred_confidence"`, `"pred_angles"`. Apply masks correctly after backbone and potentially before heads.
    *   Write unit tests (`tests/test_model.py`): Test instantiation, forward pass with dummy batch, verify output shapes.
    *   **Commit Prompt:** *AI prompts user:* "`RNAFoldingModel` V1 implemented (with placeholders/aux heads) and tests pass. Ready to commit? Suggest: `feat(model): Implement V1 RNAFoldingModel structure`"

## IV. Loss Function Implementation (`src/losses.py`)** (PRD LF-01, LF-02, LF-03, LF-04)

1.  `[ ]` **Implement `compute_fape_loss` (Simplified Proxy):** (PRD LF-01)
    *   Action: Clamped L2 distance, averaged over valid residues/batch.
2.  `[ ]` **Implement `compute_confidence_loss` (Proxy):** (PRD LF-02)
    *   Action: Calculate proxy lDDT target (distance-based) inside `torch.no_grad()`. Compute MSE loss (`sigmoid` output) or `BCEWithLogitsLoss` (logit output). Average over valid residues/batch.
3.  `[ ]` **Implement `compute_angle_loss` (Auxiliary):** (PRD LF-03)
    *   Action: Compare predicted vs. true sin/cos angle features (e.g., `1 - F.cosine_similarity`). Average over angles and valid residues/batch. Handle NaNs in true angles (mask out).
4.  `[ ]` **Write Unit Tests (`tests/test_losses.py`):**
    *   Test each loss function: scalar output, non-negative, mask handling.
5.  `[ ]` **Run Tests & Debug:** `pytest tests/test_losses.py`.
6.  `[ ]` **Commit Prompt:** *AI prompts user:* "V1 loss functions (FAPE proxy, conf proxy, angle aux) implemented and tests pass. Ready to commit? Suggest: `feat(loss): Implement V1 loss functions`"

## V. Basic Integration Test (`scripts/test_pipeline.py` or Notebook)** (PRD NF-05, NF-08)

1.  `[ ]` **Create Test Script/Notebook (Orchestrator):**
    *   Import from `src/`. Load config. Set `device`.
    *   **Define/Load Paths:** Explicitly define `data_dir`, `features_dir`, `sequences_csv_path`, `labels_csv_path` for this test.
    *   Instantiate `RNADataset` (passing paths).
    *   Instantiate `DataLoader` (small `batch_size`).
    *   Instantiate `RNAFoldingModel(config).to(device)`.
    *   Fetch batch, move tensors to `device`.
    *   Run `outputs = model(batch)`.
    *   Extract true targets from batch, move to `device`.
    *   Calculate individual losses using functions from `src/losses.py`.
    *   Calculate `total_loss` using `config['loss_weights']`.
    *   Print key shapes and loss values.
2.  `[ ]` **Execute & Verify:** Run script. Check for runtime errors (shapes, device, OOM). Ensure losses are computed. Check VRAM (`nvidia-smi`). If OOM, reduce config parameters (layers/dims).
3.  `[ ]` **Commit (Optional):** *AI prompts user:* "Basic integration test (data->model->loss) passes. Ready to commit script? Suggest: `test(integration): Add basic script testing V1 data->model->loss flow`"

---

**Completion Check (V1 Foundation):** Successful execution of the integration test signifies the completion of this phase. The system includes a working data pipeline, a foundational V1 model (scaled-down, aux heads, placeholders), and V1 losses, adhering to modularity and strict path parameterization. Ready for training loop implementation and iterative refinement (V2+).
