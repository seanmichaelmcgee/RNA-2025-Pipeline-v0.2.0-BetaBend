#!/usr/bin/env python
"""
Export Jupyter Notebook Results Tool

This script extracts the outputs from a Jupyter notebook and exports them
to a Markdown file, preserving the structure and formatting while focusing
on the outputs rather than the code.

Usage:
    python export_notebook_results.py notebook.ipynb [output_file.md]

If output_file is not specified, it will save to [notebook_name]_results.md
"""

import json
import sys
import os
import re
from datetime import datetime

def extract_notebook_outputs(notebook_path, output_path=None):
    """
    Extract outputs from a Jupyter notebook and save them to a Markdown file.
    
    Args:
        notebook_path: Path to the Jupyter notebook
        output_path: Path to save the Markdown output (optional)
    
    Returns:
        Path to the output file
    """
    # Determine output file path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(notebook_path))[0]
        output_dir = os.path.dirname(notebook_path)
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"{base_name}_results.md")
    
    # Load the notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook {notebook_path}: {e}")
        return None
    
    # Extract notebook title from the first markdown cell
    notebook_title = "Notebook Results"
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            match = re.search(r'^#\s+(.+)$', source, re.MULTILINE)
            if match:
                notebook_title = match.group(1)
                break
    
    # Start building the Markdown output
    md_lines = [
        f"# Results: {notebook_title}",
        "",
        f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"*Source notebook: {os.path.basename(notebook_path)}*",
        "",
        "## Contents",
        ""
    ]
    
    # First pass: build table of contents
    section_count = 0
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            heading_match = re.search(r'^(#+)\s+(.+)$', source, re.MULTILINE)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                section_count += 1
                
                # Add to table of contents
                indent = "  " * (level - 1)
                md_lines.append(f"{indent}- [{title}](#{section_count})")
    
    md_lines.append("")
    md_lines.append("---")
    
    # Second pass: extract content
    section_count = 0
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        cell_type = cell.get('cell_type')
        
        # Handle markdown cells (section headers)
        if cell_type == 'markdown':
            source = ''.join(cell.get('source', []))
            heading_match = re.search(r'^(#+)\s+(.+)$', source, re.MULTILINE)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                section_count += 1
                
                # Add section header with anchor
                md_lines.append("")
                md_lines.append(f"<a id='{section_count}'></a>")
                md_lines.append(f"{'#' * level} {title}")
                md_lines.append("")
        
        # Handle code cells (outputs)
        elif cell_type == 'code':
            # Get cell execution count
            execution_count = cell.get('execution_count')
            if execution_count is not None:
                exec_str = f"[{execution_count}]"
            else:
                exec_str = "[*]"
            
            outputs = cell.get('outputs', [])
            
            # Skip cells with no outputs
            if not outputs:
                continue
            
            # Process each output
            for output in outputs:
                output_type = output.get('output_type')
                
                # Handle stream output (stdout/stderr)
                if output_type == 'stream':
                    stream_name = output.get('name', 'stdout')
                    text = ''.join(output.get('text', []))
                    
                    if text.strip():  # Only add non-empty outputs
                        md_lines.append(f"**Output {exec_str}:**")
                        md_lines.append("```")
                        md_lines.append(text.rstrip())
                        md_lines.append("```")
                        md_lines.append("")
                
                # Handle display data (e.g., matplotlib figures)
                elif output_type in ('display_data', 'execute_result'):
                    # Get data from the output
                    data = output.get('data', {})
                    
                    # Handle text/plain (most common)
                    if 'text/plain' in data:
                        text = ''.join(data['text/plain'])
                        if text.strip():
                            md_lines.append(f"**Output {exec_str}:**")
                            md_lines.append("```")
                            md_lines.append(text.rstrip())
                            md_lines.append("```")
                            md_lines.append("")
                    
                    # Handle images - would need to save them separately
                    if 'image/png' in data:
                        md_lines.append(f"**Output {exec_str}:** *[Image output]*")
                        md_lines.append("")
                
                # Handle error output
                elif output_type == 'error':
                    ename = output.get('ename', 'Error')
                    evalue = output.get('evalue', '')
                    traceback = '\n'.join(output.get('traceback', []))
                    
                    # Clean up ANSI escape codes
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    traceback = ansi_escape.sub('', traceback)
                    
                    md_lines.append(f"**Error {exec_str}:**")
                    md_lines.append("```python")
                    md_lines.append(f"{ename}: {evalue}")
                    if traceback:
                        md_lines.append(traceback)
                    md_lines.append("```")
                    md_lines.append("")
    
    # Add a footer
    md_lines.append("---")
    md_lines.append("")
    md_lines.append(f"*End of results - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write the Markdown file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"Results exported to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        return None

def extract_multiple_notebook_results(tier_dir):
    """
    Extract results from all notebooks in a tier directory.
    
    Args:
        tier_dir: Path to the tier directory containing notebooks
    """
    notebooks = [f for f in os.listdir(tier_dir) if f.endswith('.ipynb')]
    
    if not notebooks:
        print(f"No notebooks found in {tier_dir}")
        return
    
    for notebook in notebooks:
        notebook_path = os.path.join(tier_dir, notebook)
        extract_notebook_outputs(notebook_path)

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} notebook.ipynb [output_file.md]")
        print(f"   or: {sys.argv[0]} --tier tier_directory")
        return 1
    
    if sys.argv[1] == '--tier':
        if len(sys.argv) < 3:
            print(f"Usage: {sys.argv[0]} --tier tier_directory")
            return 1
        tier_dir = sys.argv[2]
        extract_multiple_notebook_results(tier_dir)
    else:
        notebook_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        extract_notebook_outputs(notebook_path, output_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())