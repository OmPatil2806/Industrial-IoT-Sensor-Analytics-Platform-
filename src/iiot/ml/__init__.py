"""Machine learning: failure prediction, component diagnosis, anomaly detection and RUL."""

import os

# Keep the pipeline logs clean: MLflow prints a hint on import
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# and a progress bar for every artifact it reads back.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
