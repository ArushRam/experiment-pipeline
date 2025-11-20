import torch
import torch.nn.functional as F


class Losses:
    """Namespace + registry for loss functions."""

    # --- registry populated after method definitions ---
    _registry = {}

    @staticmethod
    def mse(preds, targets):
        return torch.nn.MSELoss()(preds, targets)

    @staticmethod
    def gaussian_nll(preds, targets, preds_var):
        return torch.nn.GaussianNLLLoss()(preds, targets, torch.exp(preds_var))

    @staticmethod
    def bce(preds, targets):
        return torch.nn.BCELoss()(preds, targets)

    @staticmethod
    def bce_with_weights(preds, targets, weights):
        return torch.nn.BCELoss(weight=weights)(preds, targets)

    @staticmethod
    def bce_logits(preds, targets):
        return torch.nn.BCEWithLogitsLoss()(preds, targets)

    @staticmethod
    def bce_logits_with_weights(preds, targets, weights):
        return torch.nn.BCEWithLogitsLoss(weight=weights)(preds, targets)

    @staticmethod
    def kl_divergence(preds, targets):
        return (
            F.binary_cross_entropy(preds, targets)
            - F.binary_cross_entropy(targets, targets)
        )

    @staticmethod
    def kl_divergence_logits(preds, targets):
        return (
            F.binary_cross_entropy_with_logits(preds, targets)
            - F.binary_cross_entropy(targets, targets)
        )

    # -----------------------
    # Registry + Getter
    # -----------------------
    @classmethod
    def get(cls, name: str):
        """
        Retrieve a loss function by name.
        """
        if name not in cls._registry:
            raise KeyError(f"Loss '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]


# Populate registry after defining methods
Losses._registry = {
    "mse": Losses.mse,
    "gaussian_nll": Losses.gaussian_nll,
    "bce": Losses.bce,
    "bce_with_weights": Losses.bce_with_weights,
    "bce_logits": Losses.bce_logits,
    "bce_logits_with_weights": Losses.bce_logits_with_weights,
    "kl_divergence": Losses.kl_divergence,
    "kl_divergence_logits": Losses.kl_divergence_logits,
}