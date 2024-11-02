from typing import Any

import torch


class TensorUtils:
    __DEBUG_MODE = True

    @staticmethod
    def log_details(torch_tensor: Any, tensor_name: str) -> None:
        if not TensorUtils.__DEBUG_MODE:
            return

        if isinstance(torch_tensor, torch.Tensor):
            print(tensor_name.ljust(24), type(torch_tensor).__name__, torch_tensor.dtype, torch_tensor.shape, sep="\t")
        else:
            print(f"{tensor_name} is not a tensor, rather it is of type `{type(torch_tensor).__name__}`")

    @staticmethod
    def print_nan_count(torch_tensor: torch.Tensor, tensor_name: str):
        nan_count = torch.sum(torch.isnan(torch_tensor).int()).item()
        if TensorUtils.__DEBUG_MODE:
            print("nan count", tensor_name.ljust(16), nan_count)
        return nan_count
