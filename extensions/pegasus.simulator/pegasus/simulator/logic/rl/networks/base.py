"""
Base interface para todas as redes neuronais.
Qualquer rede passada ao runner tem de herdar esta classe.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseActorCritic(nn.Module, ABC):

    @abstractmethod
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Amostra uma ação da política.
        Args:
            obs: [N, obs_dim]
        Returns:
            actions:   [N, act_dim]
            log_probs: [N]
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Avalia ações para o update PPO.
        Args:
            obs:     [N, obs_dim]
            actions: [N, act_dim]
        Returns:
            values:    [N]
            log_probs: [N]
            entropy:   [N]
        """
        ...

    @abstractmethod
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Ação determinística sem gradiente — usado em play/eval.
        Args:
            obs: [N, obs_dim]
        Returns:
            actions: [N, act_dim]
        """
        ...