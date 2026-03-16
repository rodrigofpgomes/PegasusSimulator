# custom_actor_critic.py
from rsl_rl.modules.actor_critic import ActorCritic
from isaaclab_rl.rsl_rl import AxisDecoupledTanhActor  # importa o actor acima
import torch.nn as nn

class AxisDecoupledActorCritic(ActorCritic):
    def __init__(self, num_obs, num_act, device, **kwargs):
        # evita crash no MLP base do ActorCritic
        kwargs.setdefault("actor_hidden_dims", [64])
        super().__init__(num_obs, num_act, device, **kwargs)

        # 🔁 envolver em Sequential para o exporter ONNX poder fazer self.actor[0].in_features
        self.actor = nn.Sequential(AxisDecoupledTanhActor(num_obs, num_act, act_func=False))
        print(f"[AxisDecoupledActorCritic] Actor substituído (obs={num_obs}, act={num_act})")
