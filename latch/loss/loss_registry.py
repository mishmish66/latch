from typing import Dict, Type

from .loss_func import LossFunc

loss_dict: Dict[str, Type[LossFunc]] = {}


def register_loss(name: str):
    def decorator(cls):
        loss_dict[name] = cls
        return cls

    return decorator


def get_loss_name(query_loss: Type[LossFunc]):
    # Do a linear search to find the name of the loss
    for name, loss_func in loss_dict.items():
        if loss_func == query_loss:
            return name
    raise ValueError(f"Loss {query_loss} not found in loss_dict")
