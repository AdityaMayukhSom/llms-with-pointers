from typing import Any, Optional

import torch


class TensorUtils:
    _DEBUG = True

    _TMPL_TNSR_INFO = "{name:<24}\t{type}\t{dtype}\t{shape}"
    _TMPL_TYPE_INFO = "{name} type is `{type}`, len is {length}"
    _TMPL_ELEM_INFO = "elem type is {elem_type}"
    _TMPL_NON_TNSR_INFO = "{name} not of type torch.Tensor, type is `{type}`"

    _STR_EMPTY_LIST_INFO = "does not contain any element"

    @classmethod
    def set_debug_mode(cls, debug: bool):
        cls._DEBUG = debug

    @classmethod
    def inspect_details(cls, torch_tensor: Any, tensor_name: str) -> None:
        if not cls._DEBUG:
            return

        tensor_type = type(torch_tensor).__name__

        if isinstance(torch_tensor, torch.Tensor):
            tensor_info = cls._TMPL_TNSR_INFO.format(
                name=tensor_name,
                type=tensor_type,
                dtype=torch_tensor.dtype,
                shape=torch_tensor.shape,
            )
            print(tensor_info)

        elif isinstance(torch_tensor, (list, tuple)):
            length = len(torch_tensor)
            type_info = cls._TMPL_TYPE_INFO.format(
                name=tensor_name,
                type=tensor_type,
                length=length,
            )

            if length > 0:
                elem_type = type(torch_tensor[0]).__name__
                elem_info = cls._TMPL_ELEM_INFO.format(elem_type=elem_type)
                print(type_info, elem_info)
            else:
                print(type_info, cls._STR_EMPTY_LIST_INFO)

        else:
            non_tensor_info = cls._TMPL_NON_TNSR_INFO.format(
                name=tensor_name,
                type=tensor_type,
            )
            print(non_tensor_info)

    @classmethod
    def nan_count(cls, torch_tensor: torch.Tensor, tensor_name: str):
        nan_count = torch.sum(torch.isnan(torch_tensor).int()).item()
        if cls._DEBUG:
            print("nan count", tensor_name, nan_count)
        return nan_count

    @classmethod
    def inf_count(cls, torch_tensor: torch.Tensor, tensor_name: str):
        inf_count = torch.sum(torch.isinf(torch_tensor).int()).item()
        if cls._DEBUG:
            print("inf count", tensor_name, inf_count)
        return inf_count

    @classmethod
    def inspect_min_max(cls, torch_tensor: torch.Tensor, tensor_name: str, dim: int):
        if cls._DEBUG:
            print(f"{tensor_name} minimum value", torch.min(torch_tensor, dim=dim, keepdim=True).values.tolist())
            print(f"{tensor_name} maximum value", torch.max(torch_tensor, dim=dim, keepdim=True).values.tolist())

    @classmethod
    def count_nan_and_inf(cls, torch_tensor: torch.Tensor, tensor_name: str):
        nan_count = cls.nan_count(torch_tensor, tensor_name)
        inf_count = cls.inf_count(torch_tensor, tensor_name)
        return nan_count, inf_count

    @classmethod
    def be_omniscient(cls, torch_tensor: Any, tensor_name: str, dim: Optional[int] = None):
        cls.inspect_details(torch_tensor, tensor_name)
        if isinstance(torch_tensor, torch.Tensor):
            cls.count_nan_and_inf(torch_tensor, tensor_name)
            if dim is not None:
                cls.inspect_min_max(torch_tensor, tensor_name, dim)
