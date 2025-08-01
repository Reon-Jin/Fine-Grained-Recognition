import torch
import torch.nn.functional as F

class NoiseHandler:
    """Implements core routines for Co‑Teaching."""

    @staticmethod
    def co_teaching_batch(logits1, logits2, targets, forget_rate: float):
        """Select small‑loss instances for each network and return indices.

        Args:
            logits1 (Tensor): B×C logits from model1
            logits2 (Tensor): B×C logits from model2
            targets (Tensor): B labels
            forget_rate (float): fraction of samples to drop
        Returns:
            idx1 (Tensor): indices kept for model1 loss (selected by model2)
            idx2 (Tensor): indices kept for model2 loss (selected by model1)
        """
        losses1 = F.cross_entropy(logits1, targets, reduction='none')
        losses2 = F.cross_entropy(logits2, targets, reduction='none')

        remember_rate = 1.0 - forget_rate
        num_remember  = max(1, int(remember_rate * targets.size(0)))

        idx1_sorted = torch.argsort(losses1)
        idx2_sorted = torch.argsort(losses2)

        idx1 = idx1_sorted[:num_remember]
        idx2 = idx2_sorted[:num_remember]
        return idx1, idx2
