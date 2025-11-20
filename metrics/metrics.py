import torch
import scipy.stats


class Metrics:
    """Namespace + registry for metric functions."""

    _registry = {}

    @staticmethod
    def mse(preds, labels):
        return torch.mean(torch.square(preds - labels)).item()

    @staticmethod
    def mse_logits(preds, labels):
        return torch.mean(torch.square(torch.special.expit(preds) - labels)).item()

    @staticmethod
    def pearson_r(preds, labels):
        return scipy.stats.pearsonr(preds, labels).statistic

    @staticmethod
    def pearson_r_pval(preds, labels):
        return scipy.stats.pearsonr(preds, labels).pvalue

    @staticmethod
    def pearson_logits_r(preds, labels):
        return scipy.stats.pearsonr(torch.special.expit(preds), labels).statistic

    @staticmethod
    def pearson_logits_r_pval(preds, labels):
        return scipy.stats.pearsonr(torch.special.expit(preds), labels).pvalue

    # -----------------------
    # Registry + Getter
    # -----------------------
    @classmethod
    def get(cls, name: str):
        """
        Retrieve a metric function by name.
        """
        if name not in cls._registry:
            raise KeyError(f"Metric '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]


# Populate registry after defining staticmethods
Metrics._registry = {
    "mse": Metrics.mse,
    "mse-logits": Metrics.mse_logits,
    "pearson-r": Metrics.pearson_r,
    "pearson-r-pval": Metrics.pearson_r_pval,
    "pearson-logits-r": Metrics.pearson_logits_r,
    "pearson-logits-r-pval": Metrics.pearson_logits_r_pval,
}
