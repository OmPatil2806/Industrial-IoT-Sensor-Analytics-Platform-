"""Machine learning: failure prediction, component diagnosis, anomaly detection and RUL."""

import os

# MLflow prints an advertising hint on import; keep the pipeline logs clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
