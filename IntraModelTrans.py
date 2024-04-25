from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.classification import LightGBMClassifier
from art.estimators.classification import XGBoostClassifier
from art.estimators.classification.pytorch import PyTorchClassifier

class IntraModelTransfer:
    def __init__(self) -> None:
        self.