from typing import Literal

import torch
import torch.nn.functional as F


class DivergenceUtils:
    def kullback_leibler(self, P: torch.Tensor, Q: torch.Tensor):
        epsilon = torch.finfo(P.dtype).eps
        kl_div_tensor = P * (torch.log(P + epsilon) - torch.log(Q + epsilon))
        kl_divergence = torch.sum(kl_div_tensor, dim=-1, keepdim=True, dtype=P.dtype)
        return kl_divergence

    def jensen_shannon(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        return_value: Literal["distance", "divergence"] = "distance",
        is_probability: bool = True,
    ):
        """
        Computes the Jensen-Shannon Divergence or Distance between two sets of logits or probability distributions.

        Refer - `Jensen-Shannon Divergence <https://en.wikipedia.org/wiki/Jensen-Shannon_divergence>`__

        Args:
            P (torch.Tensor):
                A tensor typically of shape (batch_size, num_classes), containing logits or probability values.
            Q (torch.Tensor):
                A tensor typically of shape (batch_size, num_classes), containing logits or probability values.
            return_value (Literal["distance", "divergence"], optional):
                Specifies whether to return the Jensen-Shannon "distance" (square root of divergence) or "divergence".
                Defaults to "distance".
            is_probability (bool, optional):
                If True, the inputs are treated as probabilities. If False, the inputs are considered logits and
                are converted to probabilities using softmax. Defaults to True.

        Returns:
            torch.Tensor:
                The computed Jensen-Shannon distance (if return_value is `distance`) or divergence
                (if return_value is `divergence`) between distributions P and Q.
        """
        if not is_probability:
            P = F.softmax(P, dim=-1)
            Q = F.softmax(Q, dim=-1)

        M = 0.5 * (P + Q)

        kl_PM = self.kullback_leibler(P, M)
        kl_QM = self.kullback_leibler(Q, M)

        result = 0.5 * (kl_PM + kl_QM)  # Jensen Shannon Divergence

        if return_value == "distance":
            result = torch.sqrt(result)  # Jensen Shannon Distance

        return result
