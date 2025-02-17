from sklearn.ensemble import AdaBoostRegressor
import numpy as np

def train_tradaboost(source_features, source_labels, target_features, target_labels):
    """
    Implements Transfer-Adaptive Boosting (TrAdaBoost) to refine generalization to unseen sensors.

    Args:
        source_features (array): Features extracted from source domain (TacTip sensor 1).
        source_labels (array): Corresponding 6DoF pose labels.
        target_features (array): Features extracted from target domain (TacTip sensor 2).
        target_labels (array): Corresponding pose labels for target sensor.

    Returns:
        AdaBoostRegressor: Trained TrAdaBoost model.
    """
    tradaboost = AdaBoostRegressor(n_estimators=50)
    
    # Pre-train on source domain data
    tradaboost.fit(source_features, source_labels)

    # Fine-tune on target domain data
    tradaboost.fit(target_features, target_labels)

    return tradaboost
