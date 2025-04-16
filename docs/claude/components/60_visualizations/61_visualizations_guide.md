Okay, let's create the `guide.md` for the new Visualization component. This guide will incorporate the visualization functions discussed previously and follow the standard structure.

```markdown
# START OF FILE: docs/claude/components/60_visualization/guide.md

# Visualization Implementation Guide

## Component Overview

The Visualization component provides tools for generating plots and visual representations of training progress, model performance, and predicted structures. These visualizations are crucial for monitoring experiments, debugging issues, analyzing results, and communicating findings. This component aims to encapsulate common visualization tasks into reusable functions.

## Requirements Reference

While not explicitly listed as separate components in the initial PRD, visualization fulfills aspects of monitoring, testing, and analysis implicit in model development and evaluation. Key functional requirements include:

-   **VIZ-01**: Plot training and validation loss curves over epochs/steps.
-   **VIZ-02**: Visualize the relative contributions of different loss components.
-   **VIZ-03**: Plot confidence calibration diagrams (reliability diagrams).
-   **VIZ-04**: Visualize the training trajectory in parameter space (using PCA).
-   **VIZ-05**: Generate basic 3D scatter plots of predicted RNA coordinates.
-   **VIZ-06**: Functions should handle optional dependencies gracefully (e.g., matplotlib).
-   **VIZ-07**: Functions should allow saving plots to specified file paths.

## Technical Background

-   **Libraries**: Primarily uses `matplotlib` for plotting. May use `numpy` for data manipulation and potentially `scikit-learn` for PCA (`visualize_training_trajectory`). `py3Dmol` or similar could be used for more advanced interactive structure visualization (currently out of scope for direct implementation but `visualize_coordinates` provides a basic alternative).
-   **Data Sources**: Functions typically consume data tracked during training (e.g., loss histories from a `LossTracker` object), model checkpoints, or model prediction outputs (coordinates, confidence scores).
-   **Execution Context**: Visualization functions might be called from training scripts (e.g., at the end of epochs), separate analysis scripts, or Jupyter notebooks.

## Interfaces

### Input Interface

Inputs vary depending on the specific visualization function:

```python
# For plot_loss_components / visualize_loss_breakdown
loss_history: Dict[str, List[float]] # e.g., from LossTracker.history
loss_weights: Dict[str, float]
# OR
tracker: LossTracker # An instance of the LossTracker class

# For visualize_confidence_calibration
analysis_results: Dict # Output from analyze_confidence_prediction_quality

# For visualize_training_trajectory
model_checkpoints: List[str] # List of paths to .pt files
model_class: type # e.g., RNAFoldingModel
model_config: Dict # Configuration used to instantiate the model

# For visualize_coordinates
coords: Union[torch.Tensor, np.ndarray] # Shape (N, 3) or (B, N, 3)
mask: Optional[Union[torch.Tensor, np.ndarray]] = None # Shape (N,) or (B, N)
sequence: Optional[str] = None # Sequence string of length N
```

### Output Interface

The primary output is typically a plot saved to a file or displayed.

```python
# Example output
output_path: str # Path where the plot image (e.g., .png) is saved.
# Functions may also return matplotlib figure/axis objects for further customization.
```

## Implementation Steps

Create a new file `src/utils/visualization.py` to house these functions. Ensure optional imports are handled with `try/except`.

1.  **Import Handling and Setup**:
    Set up the file with necessary imports and handle optional dependencies.

    ```python
    # src/utils/visualization.py
    import os
    import logging
    import torch
    import numpy as np
    from typing import Dict, List, Optional, Tuple, Union, Any

    # Optional imports
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        plt = None
        Axes3D = None
        MATPLOTLIB_AVAILABLE = False
        logging.warning("matplotlib not found. Visualization functions will be limited.")

    try:
        from sklearn.decomposition import PCA
        SKLEARN_AVAILABLE = True
    except ImportError:
        PCA = None
        SKLEARN_AVAILABLE = False
        # No warning needed unless PCA function is called

    # Assumes LossTracker class exists, potentially in src/utils/tracking.py
    # from src.utils.tracking import LossTracker # Example import
    class LossTracker: # Placeholder if not defined elsewhere
        def __init__(self, component_names, window_size=100): self.history={k:[] for k in component_names}; self.step=0; self.window_size=window_size; self.weights_history=[]
        def get_relative_contributions(self): return {k:0.5 for k in self.history if k != 'total'}


    def _check_matplotlib():
        """Helper to check if matplotlib is available."""
        if not MATPLOTLIB_AVAILABLE:
            logging.error("Matplotlib is required for this visualization function but was not found.")
            return False
        return True
    ```

2.  **Implement `plot_loss_components`**:
    Visualize running average loss history and relative contributions. This function was previously included in `52_losses_examples.md`.

    ```python
    # src/utils/visualization.py

    def plot_loss_components(tracker: LossTracker, output_path: Optional[str] = None) -> None:
        """
        Plot loss component history (running average) and relative contributions.

        Args:
            tracker: LossTracker instance with recorded data.
            output_path: Path to save plot. If None, defaults to 'loss_components.png'.
        """
        if not _check_matplotlib(): return
        if not tracker.history or tracker.step == 0:
            logging.warning("No loss history in tracker to plot.")
            return

        if output_path is None:
            output_path = 'loss_components.png'
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
            ax1, ax2 = axes # Unpack axes

            # Plot 1: Running average loss history
            steps = list(range(1, tracker.step + 1))
            num_steps = len(steps)

            component_names_to_plot = [name for name in tracker.history if name != 'total' and tracker.history[name]]

            for name in component_names_to_plot:
                full_history = tracker.history[name]
                avg_history = []
                # Calculate running average correctly
                for k in range(1, len(full_history) + 1):
                    window = full_history[max(0, k - tracker.window_size):k]
                    avg_history.append(np.mean(window) if window else np.nan)

                # Only plot if we have valid data
                if len(avg_history) > 0:
                    # Ensure avg_history length matches steps length if possible
                    plot_steps = steps[:len(avg_history)]
                    ax1.plot(plot_steps, avg_history, label=f'{name} (avg {tracker.window_size})', alpha=0.8)

            if not component_names_to_plot:
                 ax1.text(0.5, 0.5, 'No loss components recorded', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax1.set_title(f'Running Average Loss ({tracker.window_size}-step window)')
            ax1.set_ylabel('Loss Value')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            try: # Set yscale carefully
                 if all(v > 0 for history_list in tracker.history.values() for v in history_list if not np.isnan(v)):
                     ax1.set_yscale('log')
            except Exception:
                pass # Keep linear scale if log fails

            # Plot 2: Relative contributions (bar chart of latest values)
            contributions = tracker.get_relative_contributions()
            comp_names = list(contributions.keys())
            comp_values = list(contributions.values())

            if comp_names: # Check if contributions were calculated
                ax2.bar(comp_names, comp_values)
                ax2.set_title('Latest Relative Weighted Loss Contributions')
                ax2.set_xlabel('Loss Component')
                ax2.set_ylabel('Relative Contribution')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, axis='y', alpha=0.3)
                ax2.set_ylim(0, max(1.05, np.max(comp_values)*1.1) if comp_values else 1.05) # Adjust ylim slightly
            else:
                 ax2.text(0.5, 0.5, 'No relative contributions available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)


            plt.suptitle('Loss Component Analysis', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            plt.savefig(output_path)
            plt.close(fig) # Close the figure to free memory
            logging.info(f"Loss visualization saved to {output_path}")

        except Exception as e:
            logging.error(f"Error generating loss plot: {e}")
            plt.close(fig) # Ensure figure is closed on error
    ```

3.  **Implement `visualize_confidence_calibration`**:
    Plot reliability diagrams for confidence scores. This function was previously included in `52_losses_examples.md`.

    ```python
    # src/utils/visualization.py

    def visualize_confidence_calibration(
        analysis_results: Optional[Dict],
        output_path: str = 'confidence_calibration.png'
    ) -> None:
        """
        Visualize confidence calibration (reliability diagram).

        Args:
            analysis_results: Dictionary containing calibration analysis, typically output
                              from a function like analyze_confidence_prediction_quality.
                              Expected keys: 'bins' (list of dicts with 'accuracy', 'confidence'), 'ece'.
            output_path: Path to save the plot.
        """
        if not _check_matplotlib(): return
        if not analysis_results or 'bins' not in analysis_results or not analysis_results['bins']:
             logging.warning("Invalid or empty analysis results provided for calibration plot.")
             return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))

            bins_data = analysis_results['bins']
            # Filter out bins with zero samples if necessary, or handle confidence calculation appropriately
            valid_bins = [b for b in bins_data if b.get('count', 1) > 0] # Assume count key exists or default to >0

            if not valid_bins:
                 logging.warning("No valid bins with samples found for calibration plot.")
                 ax.text(0.5, 0.5, 'No valid data bins', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else:
                 accuracies = [b['accuracy'] for b in valid_bins]
                 confidences = [b['confidence'] for b in valid_bins] # Use average confidence in bin

                 # Reliability diagram
                 ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                 ax.plot(confidences, accuracies, 'o-', label='Model Calibration', markersize=8)

                 # Optional: Bar chart showing gap per bin (can clutter plot)
                 # gaps = np.abs(np.array(confidences) - np.array(accuracies))
                 # bin_centers = [(b['lower'] + b['upper']) / 2 for b in valid_bins]
                 # ax.bar(bin_centers, gaps, width=0.1, alpha=0.3, color='red', label='Calibration Gap')

            ece = analysis_results.get('ece', float('nan'))
            ax.set_xlabel('Average Predicted Confidence in Bin')
            ax.set_ylabel('Average Actual Accuracy in Bin')
            ax.set_title(f"Reliability Diagram (ECE: {ece:.4f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            logging.info(f"Confidence calibration visualization saved to {output_path}")

        except Exception as e:
            logging.error(f"Error generating confidence calibration plot: {e}")
            if 'fig' in locals(): plt.close(fig) # Ensure figure is closed on error
    ```

4.  **Implement `visualize_training_trajectory`**:
    Plot model parameter trajectory using PCA. This function was previously included in `52_losses_examples.md`.

    ```python
    # src/utils/visualization.py
    from src.models.rna_folding_model import RNAFoldingModel # Assuming model class import

    def visualize_training_trajectory(
        model_checkpoints: List[str], # List of paths to saved model state_dicts
        model_class: type, # The class of the model (e.g., RNAFoldingModel)
        model_config: Dict, # Config dict needed to instantiate the model
        pca_components: int = 2, # Number of PCA components (2 or 3)
        param_groups_to_include: Optional[List[str]] = None, # Filter parameters
        output_path: str = 'training_trajectory.png'
    ) -> None:
        """ Visualize the training trajectory in parameter space using PCA. """
        if not _check_matplotlib(): return
        if PCA is None:
            logging.error("scikit-learn is required for PCA visualization but not found.")
            return
        if pca_components not in [2, 3]:
             raise ValueError("pca_components must be 2 or 3")

        output_dir = os.path.dirname(output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param_vectors = []
        valid_ckpt_paths = [] # Keep track of successfully loaded checkpoints

        logging.info(f"Loading {len(model_checkpoints)} checkpoints...")
        for i, ckpt_path in enumerate(model_checkpoints):
            if not os.path.exists(ckpt_path):
                 logging.warning(f"Checkpoint file not found: {ckpt_path}. Skipping.")
                 continue
            try:
                # Instantiate model structure without loading weights initially
                # This avoids potential issues if config changed between checkpoints
                model = model_class(model_config).to(device)
                checkpoint = torch.load(ckpt_path, map_location=device)

                # Determine the key for the state dict
                state_dict_key = None
                if 'model_state_dict' in checkpoint: state_dict_key = 'model_state_dict'
                elif 'state_dict' in checkpoint: state_dict_key = 'state_dict'
                else: # Assume checkpoint is the state_dict
                    state_dict = checkpoint
                if state_dict_key: state_dict = checkpoint[state_dict_key]

                # Handle potential 'module.' prefix from DDP saving
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                model.eval()

                # Extract and flatten relevant parameters
                current_params = []
                with torch.no_grad():
                    for name, param in model.named_parameters():
                         is_included = True
                         if param_groups_to_include:
                              is_included = any(group in name for group in param_groups_to_include)
                         if is_included and param.requires_grad:
                              current_params.append(param.detach().view(-1))

                if not current_params:
                     logging.warning(f"No parameters included for checkpoint {ckpt_path}. Skipping.")
                     continue

                param_vectors.append(torch.cat(current_params).cpu().numpy())
                valid_ckpt_paths.append(ckpt_path) # Add path if loaded successfully

            except Exception as e:
                logging.error(f"Failed to load or process checkpoint {ckpt_path}: {e}")
                continue

        if len(param_vectors) < 2:
            logging.error("Need at least 2 valid checkpoints to visualize trajectory.")
            return

        logging.info(f"Performing PCA on {len(param_vectors)} parameter vectors...")
        param_matrix = np.array(param_vectors)
        try:
            pca = PCA(n_components=pca_components)
            reduced_params = pca.fit_transform(param_matrix)
            explained_variance = pca.explained_variance_ratio_
            logging.info(f"PCA explained variance: {explained_variance}")
        except Exception as e:
             logging.error(f"PCA failed: {e}")
             return


        # --- Plotting ---
        logging.info("Plotting trajectory...")
        fig = plt.figure(figsize=(10, 8))
        indices = np.arange(len(reduced_params))
        cmap = plt.cm.viridis

        if pca_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(reduced_params[:, 0], reduced_params[:, 1], c=indices, cmap=cmap, s=50, alpha=0.7)
            ax.plot(reduced_params[:, 0], reduced_params[:, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel(f'PCA Component 1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PCA Component 2 ({explained_variance[1]:.2%})')
        else: # 3D
            if Axes3D is None: # Check again in case it wasn't available initially
                 logging.error("matplotlib 3D toolkit not available for 3D plot.")
                 plt.close(fig)
                 return
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_params[:, 0], reduced_params[:, 1], reduced_params[:, 2], c=indices, cmap=cmap, s=50, alpha=0.7)
            ax.plot(reduced_params[:, 0], reduced_params[:, 1], reduced_params[:, 2], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel(f'PCA Component 1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PCA Component 2 ({explained_variance[1]:.2%})')
            ax.set_zlabel(f'PCA Component 3 ({explained_variance[2]:.2%})')

        # Add annotations for start and end points
        ax.scatter(reduced_params[0, 0], reduced_params[0, 1], reduced_params[0, 2] if pca_components==3 else None,
                   marker='o', color='lime', s=100, edgecolor='black', label='Start')
        ax.scatter(reduced_params[-1, 0], reduced_params[-1, 1], reduced_params[-1, 2] if pca_components==3 else None,
                   marker='*', color='red', s=150, edgecolor='black', label='End')

        # Add colorbar
        cbar = fig.colorbar(scatter, label='Checkpoint Index (Epoch Order)')

        title_str = 'Training Trajectory in Parameter Space (PCA)'
        if param_groups_to_include:
            title_str += f'\n(Parameters containing: {", ".join(param_groups_to_include)})'
        ax.set_title(title_str)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        logging.info(f"Training trajectory visualization saved to {output_path}")
    ```

5.  **Implement `visualize_coordinates`**:
    Generate a basic 3D scatter plot of coordinates. This function was previously included in `52_losses_examples.md`.

    ```python
    # src/utils/visualization.py

    def visualize_coordinates(
        coords: Union[torch.Tensor, np.ndarray],
        sequence: Optional[str] = None,
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        title: str = "Predicted RNA Structure",
        output_path: Optional[str] = None,
        highlight_indices: Optional[List[int]] = None,
        highlight_color: str = 'red'
    ) -> None:
        """
        Create a simplified 3D scatter plot visualization of predicted coordinates.

        Args:
            coords: Coordinates of shape (N, 3) or (B, N, 3). If batch, plots first sample.
            sequence: Optional nucleotide sequence string (length N).
            mask: Optional boolean mask of shape (N,) or (B, N). If provided, only plots valid points.
            title: Plot title.
            output_path: Path to save plot. If None, displays plot interactively (if supported).
            highlight_indices: List of 0-based indices to highlight.
            highlight_color: Color for highlighted points.
        """
        if not _check_matplotlib(): return
        if Axes3D is None:
            logging.error("matplotlib 3D toolkit not available for coordinate visualization.")
            return

        # --- Data Preparation ---
        # Ensure coords is numpy array on CPU
        if isinstance(coords, torch.Tensor):
            if coords.dim() == 3: # Batch dimension present
                if coords.shape[0] > 1:
                    logging.warning("Input has batch dimension > 1, visualizing only the first sample.")
                coords = coords[0]
            coords = coords.detach().cpu().numpy()
        elif not isinstance(coords, np.ndarray):
             raise TypeError(f"coords must be torch.Tensor or np.ndarray, got {type(coords)}")

        seq_len = coords.shape[0]

        # Prepare mask
        valid_indices = np.arange(seq_len)
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 2: mask = mask[0] # Use first sample's mask if batch
                mask = mask.detach().cpu().numpy()
            if mask.shape[0] != seq_len:
                 raise ValueError(f"Mask length ({mask.shape[0]}) does not match coordinate length ({seq_len})")
            valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
             logging.warning("No valid residues to plot based on mask.")
             return

        # Filter coordinates and sequence based on mask
        coords_to_plot = coords[valid_indices]
        if sequence is not None:
            if len(sequence) != seq_len:
                 logging.warning(f"Sequence length ({len(sequence)}) doesn't match coordinate length ({seq_len}). Ignoring sequence labels.")
                 sequence_to_plot = None
            else:
                 sequence_to_plot = "".join(sequence[i] for i in valid_indices)
        else:
             sequence_to_plot = None


        # --- Plotting ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        # Default color
        colors = ['blue'] * len(coords_to_plot)
        sizes = [50] * len(coords_to_plot)

        # Highlight specific points if requested
        if highlight_indices:
            for idx_to_highlight in highlight_indices:
                 # Find the corresponding index in the filtered coordinates
                 try:
                      plot_idx = np.where(valid_indices == idx_to_highlight)[0][0]
                      colors[plot_idx] = highlight_color
                      sizes[plot_idx] = 100 # Make highlighted points larger
                 except IndexError:
                      logging.warning(f"Highlight index {idx_to_highlight} not found among valid residues.")

        # Scatter plot
        ax.scatter(coords_to_plot[:, 0], coords_to_plot[:, 1], coords_to_plot[:, 2],
                   c=colors, s=sizes, alpha=0.7, depthshade=True)

        # Plot backbone connections (only between consecutively valid residues)
        plotted_indices = list(valid_indices) # Convert to list for index finding
        for i in range(len(plotted_indices) - 1):
             current_orig_idx = plotted_indices[i]
             next_orig_idx = plotted_indices[i+1]
             # Only plot if they were adjacent in the original sequence
             if next_orig_idx == current_orig_idx + 1:
                  ax.plot(
                      [coords_to_plot[i, 0], coords_to_plot[i+1, 0]],
                      [coords_to_plot[i, 1], coords_to_plot[i+1, 1]],
                      [coords_to_plot[i, 2], coords_to_plot[i+1, 2]],
                      c='gray', alpha=0.5, linewidth=1)

        # Add labels if sequence is provided
        if sequence_to_plot is not None:
            for i, nuc in enumerate(sequence_to_plot):
                ax.text(coords_to_plot[i, 0], coords_to_plot[i, 1], coords_to_plot[i, 2], nuc, size=8)

        # --- Styling ---
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title)

        # Auto-scale axes to keep aspect ratio roughly equal
        if len(coords_to_plot) > 0:
            max_range = np.array([coords_to_plot[:, i].max() - coords_to_plot[:, i].min() for i in range(3)]).max() / 2.0
            mid_x = (coords_to_plot[:, 0].max() + coords_to_plot[:, 0].min()) * 0.5
            mid_y = (coords_to_plot[:, 1].max() + coords_to_plot[:, 1].min()) * 0.5
            mid_z = (coords_to_plot[:, 2].max() + coords_to_plot[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.grid(True, alpha=0.2)

        # --- Output ---
        plt.tight_layout()
        if output_path:
             output_dir = os.path.dirname(output_path)
             if output_dir: os.makedirs(output_dir, exist_ok=True)
             plt.savefig(output_path)
             logging.info(f"Coordinate visualization saved to {output_path}")
        else:
             plt.show() # Display interactively

        plt.close(fig) # Close the figure
    ```

## Critical Aspects

-   **Optional Dependencies**: Functions relying on `matplotlib` or `sklearn` should check for their availability (`if MATPLOTLIB_AVAILABLE:`) and log a warning or raise an informative error if they are missing. They should not cause the entire module to fail importing.
-   **Saving Plots**: Provide an `output_path` argument to save plots. Ensure the directory exists before saving (`os.makedirs(..., exist_ok=True)`). Close figures (`plt.close(fig)`) after saving to free memory, especially important in long-running scripts or notebooks.
-   **Clarity and Labeling**: Ensure all plots have clear titles, axis labels, and legends where appropriate.
-   **Data Handling**: Functions should accept standard data structures used in the project (e.g., `LossTracker` objects, dictionaries of tensors) and handle potential missing data or edge cases gracefully.
-   **Device Handling**: Visualization usually requires data on the CPU. Ensure tensors are moved correctly (`.detach().cpu().numpy()`) before passing to plotting libraries.
-   **Modularity**: Keep visualization functions separate from core model logic and training loops. They should be callable utilities.

## Testing Requirements

Testing visualization code primarily involves ensuring the functions run without errors and produce output files, rather than pixel-perfect plot verification.

1.  **Function Execution**: Test that each visualization function runs without crashing given valid input data structures (e.g., a mock `LossTracker` object, sample coordinates).
2.  **Output Generation**: Verify that if an `output_path` is provided, the corresponding file is created.
3.  **Dependency Handling**: Test the behavior when optional dependencies (matplotlib, sklearn) are *not* installed. The functions should log warnings or skip plotting gracefully, not raise `ImportError`.
4.  **Input Validation**: Test with edge case inputs (e.g., empty history for `plot_loss_components`, zero valid points for `visualize_coordinates`) to ensure robust error handling or graceful skipping.
5.  **Masking Verification (for `visualize_coordinates`)**: Test that providing a mask correctly filters the plotted points.

```python
# Example Test Structure (tests/test_visualization.py)
import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Assume visualization functions are in src.utils.visualization
from src.utils.visualization import (
    plot_loss_components,
    visualize_confidence_calibration,
    visualize_training_trajectory,
    visualize_coordinates,
    # Mock LossTracker for testing
    LossTracker
)

# Mock availability flags for testing dependency handling
MOCK_MATPLOTLIB_AVAILABLE = True
MOCK_SKLEARN_AVAILABLE = True

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """Mock optional dependencies for testing."""
    monkeypatch.setattr('src.utils.visualization.MATPLOTLIB_AVAILABLE', MOCK_MATPLOTLIB_AVAILABLE)
    monkeypatch.setattr('src.utils.visualization.SKLEARN_AVAILABLE', MOCK_SKLEARN_AVAILABLE)
    # Mock plt.savefig and plt.show to prevent actual plotting during tests
    monkeypatch.setattr('matplotlib.pyplot.savefig', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.show', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.close', MagicMock())


class TestPlotLossComponents:
    @pytest.fixture
    def sample_tracker(self):
        tracker = LossTracker(['fape', 'confidence', 'angle', 'total'], window_size=10)
        # Add some sample data
        for i in range(20):
            tracker.update({
                'fape': 1.0 / (i + 1),
                'confidence': 0.5 * (1.0 / (i + 1)),
                'angle': 0.8 * (1.0 / (i + 1)),
                'total': 2.3 * (1.0 / (i + 1))
            }, weights={'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}) # Need weights for contribution plot
        return tracker

    def test_runs_without_error(self, sample_tracker, tmp_path):
        """Test that the function executes without raising errors."""
        output_file = tmp_path / "loss_plot.png"
        plot_loss_components(sample_tracker, output_path=str(output_file))
        assert output_file.exists() # Check if file was created

    def test_handles_empty_tracker(self, tmp_path):
        """Test behavior with an empty tracker."""
        empty_tracker = LossTracker(['fape', 'total'])
        output_file = tmp_path / "empty_loss_plot.png"
        # Should run without error, potentially logging a warning
        plot_loss_components(empty_tracker, output_path=str(output_file))
        # File might not be created if there's nothing to plot, or might be empty
        # assert not output_file.exists() # Or assert it exists but is blank? Depends on desired behavior.

# Add similar test classes for visualize_confidence_calibration,
# visualize_training_trajectory, visualize_coordinates, testing:
# - Execution without errors
# - File creation
# - Handling of missing optional dependencies (by setting MOCK_..._AVAILABLE = False)
# - Handling of edge case inputs (e.g., empty data, single data point)
# - Correct mask filtering in visualize_coordinates
```

## Example Usage

```python
# --- In a training script (e.g., scripts/train.py) ---
from src.utils.visualization import plot_loss_components
from src.utils.tracking import LossTracker # Assuming LossTracker is here

# Initialize tracker
loss_tracker = LossTracker(['fape', 'confidence', 'angle', 'total'])

# Inside training loop (after each epoch)
# epoch_losses = {'train_fape': avg_train_fape, 'val_fape': avg_val_fape, ...}
# loss_tracker.update(epoch, epoch_losses) # Update with epoch averages

# After training loop
plot_loss_components(loss_tracker, output_path=os.path.join(output_dir, "training_losses.png"))

# --- In an analysis notebook ---
from src.utils.visualization import visualize_coordinates, visualize_confidence_calibration
import torch

# Load model predictions and true data
predictions = torch.load("predictions.pt") # Example: {'coords': tensor, 'confidence': tensor}
true_data = torch.load("true_data.pt")   # Example: {'coords': tensor, 'mask': tensor}

# Visualize a predicted structure (first sample in batch)
visualize_coordinates(
    coords=predictions['coords'][0],
    mask=true_data['mask'][0],
    title="Sample Prediction",
    output_path="sample_structure_viz.png"
)

# Analyze and visualize confidence
# Assume analyze_confidence_prediction_quality exists and returns results dict
# calibration_results = analyze_confidence_prediction_quality(...)
# visualize_confidence_calibration(calibration_results, output_path="calibration_plot.png")
```

## Related Documentation

-   **Losses Guide**: `docs/claude/components/50_losses/guide.md` (Provides context for `LossTracker`)
-   **Debugging Guide**: `docs/claude/workflows/80_debugging.md` (Visualization is a key debugging tool)
-   **Model Integration**: `docs/claude/workflows/60_model_integration.md` (Shows where outputs visualized here are generated)

## Next Steps

1.  Create the `src/utils/visualization.py` file.
2.  Implement the functions outlined above, ensuring optional dependencies are handled.
3.  Create the corresponding test file `tests/test_visualization.py` with tests covering execution, file output, dependency handling, and edge cases.
4.  Integrate calls to these visualization functions into training scripts (`scripts/train.py`), evaluation scripts, or analysis notebooks as needed.

```
