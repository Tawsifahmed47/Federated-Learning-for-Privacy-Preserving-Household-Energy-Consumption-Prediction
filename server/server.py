from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from server.strategies import fedavg_aggregate, fedprox_aggregate, scaffold_aggregate


class FLServer:
    def __init__(self, model, algorithm: str) -> None:
        self.model = model
        self.algorithm = algorithm.lower()
        self.c_global = {
            k: torch.zeros_like(v) for k, v in self.model.state_dict().items()
        }
        self.c_locals: Dict[str, Dict[str, torch.Tensor]] = {}

    def get_global_state(self) -> OrderedDict:
        return self.model.state_dict()

    def sample_clients(self, all_client_ids: List[str], fraction: float) -> List[str]:
        n_total = len(all_client_ids)
        n_pick = max(1, int(fraction * n_total))
        return random.sample(all_client_ids, n_pick)

    def aggregate(
        self,
        selected_ids: List[str],
        client_states: List[OrderedDict],
        sample_counts: List[int],
        updated_c_locals: List[Dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if self.algorithm == "fedavg":
            new_global = fedavg_aggregate(client_states, sample_counts)
        elif self.algorithm == "fedprox":
            new_global = fedprox_aggregate(client_states, sample_counts)
        elif self.algorithm == "scaffold":
            new_global, self.c_global = scaffold_aggregate(
                self.get_global_state(),
                client_states,
                sample_counts,
                self.c_global,
                updated_c_locals or [],
            )
            for cid, c_local in zip(selected_ids, updated_c_locals or []):
                self.c_locals[cid] = c_local
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        self.model.load_state_dict(new_global)

    def get_client_control_variate(self, client_id: str) -> Dict[str, torch.Tensor]:
        if client_id not in self.c_locals:
            self.c_locals[client_id] = {
                k: torch.zeros_like(v) for k, v in self.model.state_dict().items()
            }
        return self.c_locals[client_id]

