#!/bin/bash
# Create folder structure for RNA 3D Folding project

# Create docs structure
mkdir -p docs/claude/components
mkdir -p docs/claude/integration
mkdir -p docs/claude/reference
mkdir -p docs/claude/testing
mkdir -p docs/claude/workflows

# Create source code structure
mkdir -p src/models
mkdir -p tests

# Create claude instruction files
touch docs/claude/master.md

# Component instructions
touch docs/claude/components/data_loading.md
touch docs/claude/components/embeddings.md
touch docs/claude/components/transformer_block.md
touch docs/claude/components/ipa_module.md
touch docs/claude/components/losses.md

# Integration guides
touch docs/claude/integration/pipeline_flow.md
touch docs/claude/integration/model_integration.md

# Reference materials
touch docs/claude/reference/feature_formats.md
touch docs/claude/reference/architecture.md
touch docs/claude/reference/code_patterns.md

# Testing guidelines
touch docs/claude/testing/unit_tests.md
touch docs/claude/testing/integration_tests.md

# Workflow documents
touch docs/claude/workflows/debugging.md
touch docs/claude/workflows/performance.md
touch docs/claude/workflows/kaggle_submission.md

# Source code files (empty placeholders)
touch src/data_loading.py
touch src/losses.py
touch src/models/embeddings.py
touch src/models/transformer_block.py
touch src/models/ipa_module.py
touch src/models/rna_folding_model.py

# Create basic test placeholder
touch tests/test_data_loading.py

echo "Folder structure created successfully!"
echo "Next steps:"
echo "1. Populate the Claude instruction markdown files"
echo "2. Begin implementation following the systematic approach"
