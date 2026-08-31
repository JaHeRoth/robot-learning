"""Rollout-protocol wrapper for the from-scratch ACT: action queue + the
normalization boundary (dataset-stats in, unnormalized actions out)."""
from collections import deque

import torch
from torch import Tensor
from torch.nn import Module

from scripts.act import ACT


class MyACTPolicy(Module):
    def __init__(self, act: ACT, stats: dict, n_action_steps: int = 16):
        super().__init__()
        self.act = act
        self.n_action_steps = n_action_steps
        for key, feature in [("state", "observation.state"), ("action", "action")]:
            for stat in ["mean", "std"]:
                self.register_buffer(
                    f"{key}_{stat}",
                    torch.as_tensor(stats[feature][stat], dtype=torch.float32),
                )
        self.reset()

    def reset(self):
        self._queue = deque(maxlen=self.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if not self._queue:
            img = batch["observation.image"].unsqueeze(1)  # add camera dim
            proprio = (batch["observation.state"] - self.state_mean) / self.state_std
            chunk_pred, _, _ = self.act(img, proprio, chunk=None)
            actions = chunk_pred[:, : self.n_action_steps] * self.action_std + self.action_mean
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()
