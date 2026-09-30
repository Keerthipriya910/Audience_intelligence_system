# Build Validation

The packaged source was validated with:

- Python bytecode compilation across the project.
- Unit tests for EWTC confidence behavior and no-evidence identity behavior.
- End-to-end Demo Mode execution over the bundled multilingual sample dataset.
- Demo threshold-sweep experiment execution.
- Paper-results Markdown generation from an experiment report.

Research Mode was not executed during package construction because it requires downloading external Hugging Face model weights. Research Mode is intentionally strict: it does not silently substitute Demo Mode sentiment, emotion, topic, or FAISS behavior if a required research dependency fails.
