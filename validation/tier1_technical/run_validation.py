#!/usr/bin/env python
"""
Run Tier 1 technical validation for the RNA 3D folding model.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run RNA 3D folding model technical validation")
    parser.add_argument("--notebook", default="validation_technical.ipynb", 
                        help="Path to validation notebook (default: validation_technical.ipynb)")
    parser.add_argument("--output", default="validation_results.html",
                        help="Output file path (default: validation_results.html)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (optional)")
    args = parser.parse_args()
    
    # Get current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    notebook_path = current_dir / args.notebook
    output_path = current_dir / "results" / args.output
    
    # Create results directory if it doesn't exist
    os.makedirs(current_dir / "results", exist_ok=True)
    
    # Set environment variable for checkpoint if provided
    env = os.environ.copy()
    if args.checkpoint:
        env["MODEL_CHECKPOINT"] = args.checkpoint
        print(f"Using model checkpoint: {args.checkpoint}")
    
    # Run notebook
    print(f"Running validation notebook: {notebook_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        cmd = [
            "jupyter", "nbconvert", 
            "--ExecutePreprocessor.timeout=600",
            "--to", "html", 
            "--execute", 
            str(notebook_path),
            "--output", str(output_path.name),
            "--output-dir", str(output_path.parent)
        ]
        
        subprocess.run(cmd, check=True, env=env)
        print(f"Validation completed successfully. Results saved to {output_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running validation notebook: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())