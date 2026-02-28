"""
This submodule holds the mathematical definitions of the Machine Learning models (specifically PyTorch), 
their training loops, wrappers, and evaluative architectures.

Highlights:

*   **Wrappers**: Standardized interface for models (`PyTorchWrapper`) allowing smooth ingestion within the 
    Simulator and FeatureCreation pipelines.
*   **Training & Evaluation**: Automated metric tracking and gradient scaling procedures.
*   **Architectures**: Contains specific folder-level implementations, such as Transformer-based sequence models.
"""
