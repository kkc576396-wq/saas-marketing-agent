# Workflow

Contains the LangGraph state, nodes, edges, and graph assembly for the
AnySearch research workflow.

Current flow:

```text
planner -> search -> analyzer -> verifier -> (search | save)
```

The verifier evaluates relevance, evidence availability, and confidence, and
only verified insights are retained for the saved output.
