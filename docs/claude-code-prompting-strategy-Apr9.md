# Updated Prompting Strategy for Revised Structure

## Initial Project Introduction

For your first conversation with Claude, set the context for the entire project:

```
I'd like your help implementing the RNA 3D structure prediction project. Please first review:

1. docs/claude/00_master_guide.md for an overview of the implementation approach
2. docs/claude/01_implementation_principles.md for critical cross-cutting principles

After reviewing these guides, let me know if you have any questions before we begin implementation.
```

## Component Implementation Requests

When requesting implementation of a specific component:

```
I need your help implementing the [component name] for the RNA 3D structure prediction project.

Please refer to these guides in order:
1. docs/claude/components/[XX_component_folder]/guide.md for implementation instructions
2. docs/claude/components/[XX_component_folder]/examples.md for code patterns

Based on these guides, please:
1. Implement the required functions/classes in a code artifact
2. Provide unit tests in a separate code artifact
3. Explain any design decisions you made that weren't explicitly specified

I'm particularly concerned about [specific aspect], so please pay special attention to that.
```

## Cross-Component Integration

For integration tasks that span multiple components:

```
Now that we've implemented [component A] and [component B], I need your help integrating them following:

1. docs/claude/workflows/60_model_integration.md for integration guidance
2. docs/claude/components/[XX_component_A]/guide.md (section on interfaces)
3. docs/claude/components/[YY_component_B]/guide.md (section on interfaces)

Please review these guides and then:
1. Implement the necessary integration code
2. Highlight any potential issues or inconsistencies between components
3. Suggest any optimizations we might make
```

## Implementation Issue Resolution

When encountering implementation problems:

```
We're facing an issue with the [component] implementation. The specific error is:

[Error details]

Please review:
1. docs/claude/components/[XX_component_folder]/guide.md
2. docs/claude/workflows/80_debugging.md for troubleshooting guidance

Then analyze the issue and recommend a solution, including any code changes needed.
```

## Reference Only Requests

For consultative questions that don't require implementation:

```
I'd like you to help me understand the [technical concept] in the context of our project.

Please review:
1. docs/claude/reference/[relevant_reference].md
2. docs/[X_Original_Specification].md (section [Y.Z])

Then explain [technical concept] in the specific context of our RNA 3D folding implementation.
```
