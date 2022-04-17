from typing import Optional, Any, Callable

import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from torchmetrics.functional.regression.mean_squared_error import _mean_squared_error_update, \
    _mean_squared_error_compute


class MaskMeanSquaredError(Metric):
    r"""
    Computes `mean squared error`_ (MSE):

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        squared:
            If True returns MSE value, if False returns RMSE value.

    """
    is_differentiable = True
    higher_is_better = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
            squared: bool = True,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: A bool matrix the same shape as preds where True is indication for missing data and False is for
            observed. We are interest in measuring only the mse over missing data
        """
        if mask is not None:
            n_obs = mask.sum()
            sum_squared_error = torch.sum(mask * (preds - target) ** 2)
        else:
            sum_squared_error, n_obs = _mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)
