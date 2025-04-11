 We are iteratively moving through generating guideline documentation for this Claude Code project. At each step, I am going to get you to generate one artifact. We'll continue generating high-quality artifacts until we run out of context window. We are using a lot of context to generate really high-quality artifacts.

Your task in response to this prompt is to generate a single, comprehensive GUIDE markdown file for the TRANSFORMER component of our RNA 3D folding project.

# Document Specifications
- **Target File:** `docs/claude/components/30_transformer_block/transformer_examples.md`
- **Document Type:** EXAMPLES MARKDOWN
- **Document Purpose:** Following the structured examples of prior code examples, we will build out the transformer code examples needed for this component to assist the downstream engineer.
- **Previous Component Files:** transformer_guide.md
- **Related Component Files:** transformer_guide.md ; transforemr_testing.md (NOT YET GENREATED, NEXT TO BE GENERATED)

# Project Structure Reference

## Essential Documentation to Review
Before generating this documentation, please review these key documents:

1. **Primary Specifications:**
   - `docs/1_Context_and_Setup.md` - Project setup, containerization, and architecture
   - `docs/2_Feature_Specification.md` - RNA feature data formats and processing
   - `docs/3_Architecture_Specification.md` - Detailed model architecture design
   - `docs/4_Product_Requirements_V1.md` - Component requirements (reference specific requirement IDs)
   - `docs/6_Tactical_Plan_V1.md` - Implementation steps and guidelines

2. **Core Guidelines:**
   - `docs/claude/00_master_guide.md` - Implementation roadmap and overview
   - `docs/claude/01_implementation_principles.md` - Project-wide patterns and principles
   - `docs/claude/reference/feature_formats.md` - Feature file formats reference
   - `docs/claude/reference/pytorch_patterns.md` - PyTorch implementation patterns

## Documentation Structure
The project documentation follows this hierarchical structure:

```
docs/
├── 1_Context_and_Setup.md              # Existing detailed specification
├── 2_Feature_Specification.md          # Existing detailed specification
├── 3_Architecture_Specification.md     # Existing detailed specification
├── 4_Product_Requirements_V1.md        # Existing detailed specification
├── 5_Roadmap_V1.md                     # Existing detailed specification
├── 6_Tactical_Plan_V1.md               # Existing detailed specification
├── 7_AI_Agent_Rules.md                 # Existing detailed specification
├── claude/                             # Claude instruction folder
│   ├── 00_master_guide.md              # Entry point with implementation roadmap
│   ├── 01_implementation_principles.md # Core principles and patterns across components
│   │
│   ├── components/                     # Task-focused implementation guides by component
│   │   ├── 10_data_loading/
│   │   │   ├── guide.md                # Main implementation guide
│   │   │   ├── examples.md             # Code examples and patterns
│   │   │   └── testing.md              # Testing strategy and examples
│   │   │
│   │   ├── 20_embeddings/
│   │   │   ├── guide.md                # Main implementation guide
│   │   │   ├── examples.md             # Code examples and patterns
│   │   │   └── testing.md              # Testing strategy and examples
│   │   │
│   │   ├── 30_transformer_block/
│   │   │   ├── guide.md                # Main implementation guide
│   │   │   ├── examples.md             # Code examples and patterns
│   │   │   └── testing.md              # Testing strategy and examples
│   │   │
│   │   ├── 40_ipa_module/
│   │   │   ├── guide.md                # Main implementation guide
│   │   │   ├── examples.md             # Code examples and patterns
│   │   │   └── testing.md              # Testing strategy and examples
│   │   │
│   │   └── 50_losses/
│   │       ├── guide.md                # Main implementation guide
│   │       ├── examples.md             # Code examples and patterns
│   │       └── testing.md              # Testing strategy and examples
│   │
│   ├── workflows/                      # Integration and common workflow guides
│   │   ├── 60_model_integration.md     # Combining components into full model
│   │   ├── 70_pipeline_testing.md      # End-to-end pipeline testing
│   │   ├── 80_debugging.md             # Common debugging patterns
│   │   └── 90_kaggle_submission.md     # Preparing for Kaggle submission
│   │
│   └── reference/                      # Simplified references and examples
│       ├── feature_formats.md          # Clear examples of feature file formats
│       ├── configuration.md            # Configuration options and usage
│       └── pytorch_patterns.md         # Recommended PyTorch patterns
```

## Implementation Structure
The target implementation should follow this structure:

```
src/
├── data_loading.py                    # Data loading and processing
├── models/                            # Model components
│   ├── __init__.py
│   ├── embeddings.py                  # Embedding layers
│   ├── transformer_block.py           # Transformer block implementation
│   ├── ipa_module.py                  # IPA module (placeholder for V1)
│   └── rna_folding_model.py           # Main model integration
├── losses.py                          # Loss function implementations
└── utils/                             # Utility functions
    ├── __init__.py
    └── ...

tests/                                 # Test directory
├── test_data_loading.py
├── test_embeddings.py
├── test_transformer_block.py
├── test_ipa_module.py
├── test_losses.py
└── test_integration.py
```

Remember to adhere strictly to the path parameterization principle (no hardcoded paths) and maintain clear separation between core logic (in `src/`) and orchestration (in `scripts/`).
```


# Document Requirements
The markdown file should:
1. Follow the structure and tone of existing documentation
2. Include clear section headers, code examples, and explanations
3. Reference relevant portions of the Architecture Specification and Product Requirements
4. Focus on practical implementation guidance rather than theoretical explanations
5. Include proper cross-references to other documentation when appropriate
6. Emphasize core principles like path parameterization and modularity

# Specific Content Requirements
{SPECIFIC_REQUIREMENTS_FOR_THIS_DOCUMENT}
