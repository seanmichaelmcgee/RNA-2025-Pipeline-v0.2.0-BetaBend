# Kaggle Inference Notebook Execution Log
Generated: 2025-04-23 22:12:22

## Environment Information
Device used: cuda
Python version: 3.10.17 | packaged by conda-forge | (main, Apr 10 2025, 22:19:12) [GCC 13.3.0]
PyTorch version: 2.5.1

## Configuration
```python
{
  "MODEL_PATHS": {
    "final_model": "../results/final_model/run_20250423-072601/checkpoints/best_model.pt",
    "tuning_lr_0.001": "../results/tuning_run_1/lr_0.001/run_20250423-072437/checkpoints/best_model.pt",
    "tuning_lr_0.0005": "../results/tuning_run_1/lr_0.0005/run_20250423-072448/checkpoints/best_model.pt",
    "tuning_lr_0.0001": "../results/tuning_run_1/lr_0.0001/run_20250423-072458/checkpoints/best_model.pt",
    "production_run_1": "../results/production_run_1/run_20250423-072209/checkpoints/best_model.pt"
  },
  "SELECTED_MODEL": null,
  "USE_ENSEMBLE": false,
  "BATCH_SIZE": 1,
  "NUM_SAMPLES": 5,
  "TEMPERATURE": 0.1
}
```

## Cell Execution Results
## Errors Detected
Most recent error traceback:
```

```
## Model Information
Number of models loaded: 1
### final_model
**Metrics:**
- val_rmsd: 7.593239451519988
- epoch: 30

## Model Loading/Inference Debug Info
models dictionary has 1 entries.
results dictionary has 12 entries.
First target (R1107) has 5 samples.
First sample keys: ['target_id', 'sample_id', 'coords', 'confidence']
coords shape: (69, 3)
confidence shape: (69,)
## Submission Information
Submission shape: (2515, 18)
Kaggle format submission
Target IDs: 12
Total residues: 2515

**Sample entries:**
```
        ID resname  resid       x_1       y_1       z_1       x_2       y_2  \
0  R1107_1       G      1  1.222656 -1.901367  0.580566  1.378906 -1.816406   
1  R1107_2       G      2  1.593750 -1.538086  0.437500  1.306641 -1.627930   
2  R1107_3       G      3  1.571289 -1.467773  0.323242  2.068359 -1.057617   

        z_2       x_3       y_3       z_3       x_4       y_4       z_4  \
0  0.395508  1.887695 -2.134766  0.760742  1.230469 -1.813477  0.008163   
1  0.279297  1.480469 -1.171875  0.390137  1.356445 -1.491211  1.068359   
2  0.029617  1.253906 -1.079102 -0.144409  1.708984 -1.044922  0.471680   

        x_5       y_5       z_5  
0  1.026367 -1.032227  1.064453  
1  1.314453 -1.241211  0.012543  
2  1.800781 -0.996094 -0.268311  
```