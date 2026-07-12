from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch


def fedavg_aggregate(
    client_states: List[OrderedDict], sample_counts: List[int]
) -> OrderedDict:
    total_samples = sum(sample_counts)
    agg_state = OrderedDict()
    for key in client_states[0].keys():
        agg_state[key] = sum(
            state[key] * (n_i / total_samples) for state, n_i in zip(client_states, sample_counts)
        )
    return agg_state


def fedprox_aggregate(
    client_states: List[OrderedDict], sample_counts: List[int]
) -> OrderedDict:
    return fedavg_aggregate(client_states, sample_counts)


def scaffold_aggregate(
    global_state: OrderedDict,
    client_states: List[OrderedDict],
    sample_counts: List[int],
    c_global: Dict[str, torch.Tensor],
    updated_c_locals: List[Dict[str, torch.Tensor]],
) -> Tuple[OrderedDict, Dict[str, torch.Tensor]]:
    new_global_state = fedavg_aggregate(client_states, sample_counts)
    if updated_c_locals:
        n = len(updated_c_locals)
        for key in c_global.keys():
            c_global[key] = sum(ci[key] for ci in updated_c_locals) / n
    return new_global_state, c_global

