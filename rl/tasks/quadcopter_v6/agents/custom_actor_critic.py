# custom_actor_critic.py
from rsl_rl.modules.actor_critic import ActorCritic
from isaaclab_rl.rsl_rl import AxisDecoupledPDActor, AxisDecoupledPDActorLinear, QuadraticFormCritic, LQRCritic
import torch.nn as nn

class NetworkActorCritic(ActorCritic):
    def __init__(self, num_obs, num_act, device, **kwargs):
        super().__init__(num_obs, num_act, device, **kwargs)

        # Sequential para o exporter ONNX poder fazer self.actor[0].in_features
        self.actor = nn.Sequential(AxisDecoupledPDActorLinear(num_obs, num_act, act_func=False))
        print(f"[AxisDecoupledPDActorLinear] Actor substituído (obs={num_obs}, act={num_act})")

        self.critic = nn.Sequential(QuadraticFormCritic(num_obs))
        print(f"[QuadraticFormCritic] Critic substituído (obs={num_obs}, act=1)")
