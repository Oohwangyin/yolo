# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .extended_metrics import ExtendedDetectionValidator

# Use the extended validator for every standard model.val() call. The implementation remains in a separate module so
# user-facing validation scripts do not need metric-specific logic.
DetectionValidator = ExtendedDetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "ExtendedDetectionValidator"
