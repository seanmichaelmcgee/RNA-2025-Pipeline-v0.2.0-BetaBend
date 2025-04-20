```markdown
## 15. Data Loading: V2 Readiness & V1 Risk Mitigation

As we prepare for the full V2/IPA‑powered pipeline, it’s critical to identify where our V1 loader could lose information or block future extensions—and to build in safeguards now.

---

### 15.1 High‑Risk Areas

| ID | Risk | Impact |
|:---|:-----|:-------|
| **A** | **Feature drop‑in vs. leakage for pseudo‑dihedrals**<br>Auxiliary dihedral inputs only during training may be accidentally removed or reshaped incorrectly in V2. | Models might over‑rely on training‑only features or break when angles are omitted in inference/teacher modes. |
| **B** | **Ignoring uniform MI matrices**<br>Short RNAs with constant MI may teach the model to ignore that channel entirely. | When fed rich MI later, the network under‑utilizes a key signal. |
| **C** | **Rigid `collate_fn` padding**<br>Current batcher pads only 1D and 2D inputs; V2 needs 3D frames or coordinate inits. | Hard to add new per‑residue or per‑pair tensors without rewriting collate logic. |
| **D** | **Fixed temporal‑cutoff logic**<br>Dataset handles only “before/after” splits. | Limits future cross‑validation, sliding‑window evaluation, or multi‑stage splits. |
| **E** | **Test coverage gaps**<br>Edge cases like “dihedrals present but thermo missing” or very long sequences aren’t covered. | Undetected bugs when features are partially missing or sequence lengths balloon. |

---

### 15.2 Mitigation Strategies

| Risk | Mitigation |
|:-----|:-----------|
| **A. Dihedral handling** | • Always emit **both** `dihedral_input` and `dihedral_target` tensors of shape `(L,4)` (sin/cos), even if zero at inference.<br> • Document their use in a shared helper:  
  ```python
  def get_dihedral_tensors(target_id):
      # returns (input_angles: Tensor[L,4],
      #          target_angles: Tensor[L,4])
  ``` |
| **B. MI channel gating** | • Emit a boolean `has_msa: bool` per sequence in the batch metadata.<br> • In the embedder, concatenate `has_msa` so the model knows when to trust MI. |
| **C. Composable padding** | Refactor `collate_fn` into small utilities—e.g.:  
  ```python
  def pad_1d(x: Tensor, max_len: int) -> Tensor: ...
  def pad_2d(x: Tensor, max_len: int) -> Tensor: ...
  def pad_frames(frames: Tensor, max_len: int) -> Tensor: ...
  ```  
  Then build batches by chaining these. |
| **D. Pluggable split functions** | In `RNADataset.__init__`, accept a `split_fn: Callable[[DataFrame], DataFrame]` instead of hard‑coded modes. Supply defaults for train/val now; add k‑fold later without touching Dataset. |
| **E. Broaden tests** | Add pytest cases for:<br>1. Mixed‑presence features (e.g., dihedral=present, thermo=missing).<br>2. Long sequences (e.g. L=500) to verify padding and memory warnings.<br>3. Partial MI or NaN‑only channels. |

---

### 15.3 Roadmap for V2‑Ready Design

1. **Abstract Embedders & Modules**  
   - Define base classes and a registry:
     ```python
     REGISTRY = {}
     def register_module(name):
         def decorator(cls):
             REGISTRY[name] = cls
             return cls
         return decorator

     @register_module("ipa_stub")
     class IPAModuleStub(nn.Module):
         def forward(self, res: Tensor, pair: Tensor) -> Tuple[Tensor,Tensor]:
             # returns coords: (B,L,3), frames: (B,L,3,3)
             ...
     ```
   - V1 uses `ipa_stub`; V2 can register `ipa_full` seamlessly.

2. **Composable Padding Utilities**  
   - Centralize padding in `src/utils/padding.py`:
     ```python
     def pad_tensor(x: Tensor, target_shape: Tuple[int,...], pad_value=0):
         # Generic N‑D padding
     ```
   - Collate builds on these, so adding `frames` or new pair features is trivial.

3. **Config‑Driven Component Graph**  
   - In `config/default_config.yaml`:
     ```yaml
     model:
       residue_embedder: v1_simple
       pair_embedder: v1_simple
       backbone: v1_transformer
       structure_module: ipa_stub
     ```
   - Training script instantiates each via the registry, eliminating hard‑coded imports.

4. **Metadata Flags in Batches**  
   - `collate_fn` returns extra keys:
     ```python
     batch["meta"] = {
         "has_dihedrals": Tensor[B],
         "has_msa":     Tensor[B]
     }
     ```
   - Future losses or attention heads can condition on these.

5. **Integration Smoke Test**  
   - Add `tests/test_integration_smoke.py`:
     ```python
     def test_pipeline_smoke():
         loader = create_data_loader(...)
         batch = next(iter(loader))
         model = RNAFoldingModel(config)
         coords, conf, angles = model(batch)
         assert coords.shape == (batch_size, L, 3)
         assert conf.shape   == (batch_size, L)
         assert angles.shape == (batch_size, L, 2)
     ```

By building these abstractions and safeguards now, the V1 data loader remains lean, well‑tested, and ready to absorb the full V2/IPA machinery with minimal refactoring.  
```
