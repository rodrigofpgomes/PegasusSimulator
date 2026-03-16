# custom_actor_critic.py
from rsl_rl.modules.actor_critic import ActorCritic
from isaaclab_rl.rsl_rl import AxisDecoupledPDActor, AxisDecoupledPDActorLinear, QuadraticFormCritic, LQRCritic
import torch.nn as nn

class AxisDecoupledActorCritic(ActorCritic):
    def __init__(self, num_obs, num_act, device, **kwargs):
        kwargs.setdefault("actor_hidden_dims", [64])
        super().__init__(num_obs, num_act, device, **kwargs)

        self.actor = nn.Sequential(AxisDecoupledPDActorLinear(num_obs, num_act, act_func=False))
        print(f"[AxisDecoupledActorCritic] Actor substituído (obs={num_obs}, act={num_act})")
