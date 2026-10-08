import copy
import math

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# ---------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------

class Policy(GaussianMixin, Model):
    """
    Política Gaussiana equivalente à MlpPolicy usada no repositório
    optimal_quad_control_RL.

    Observação: 26
    Ação: 4 comandos normalizados dos rotores em [-1, 1]
    Arquitetura: 64-64-64, ReLU
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        # Stable-Baselines3 utiliza um log_std global por ação.
        # log_std_init=0 -> std inicial = exp(0) = 1.
        self.log_std_parameter = nn.Parameter(
            torch.Tensor(self.num_actions).fill_(torch.tensor(-1.0)),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inicialização próxima da inicialização ortogonal usada pelo SB3.
        """
        linear_layers = [
            module
            for module in self.net
            if isinstance(module, nn.Linear)
        ]

        # Camadas escondidas
        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(
                layer.weight,
                gain=math.sqrt(2.0),
            )
            nn.init.zeros_(layer.bias)

        # Output da política com ganho pequeno
        nn.init.orthogonal_(
            linear_layers[-1].weight,
            gain=0.01,
        )
        nn.init.zeros_(linear_layers[-1].bias)

    def compute(self, inputs, role):
        mean_actions = self.net(inputs["states"])

        return mean_actions, self.log_std_parameter, {}


# ---------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------

class Value(DeterministicMixin, Model):
    """
    Função de valor V(s), separada da policy.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        linear_layers = [
            module
            for module in self.net
            if isinstance(module, nn.Linear)
        ]

        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(
                layer.weight,
                gain=math.sqrt(2.0),
            )
            nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(
            linear_layers[-1].weight,
            gain=1.0,
        )
        nn.init.zeros_(linear_layers[-1].bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def make_models(observation_space, action_space, device):
    return {
        "policy": Policy(
            observation_space,
            action_space,
            device,
        ),
        "value": Value(
            observation_space,
            action_space,
            device,
        ),
    }


# ---------------------------------------------------------------------
# Reward scaling
# ---------------------------------------------------------------------

def reward_shaper(rewards, timestep, timesteps):
    """
    O teu ambiente tem rewards cerca de 10 vezes maiores:

    reward normal: até aproximadamente +1.5
    morte: -100

    No repositório inicial, as rewards normais são normalmente da ordem
    de 0.01–0.1 e a colisão dá -10.

    Multiplicar por 0.1 preserva exatamente a política ótima:

    +1.5  -> +0.15
    -100  -> -10
    """
    return 0.1 * rewards


# ---------------------------------------------------------------------
# PPO configuration
# ---------------------------------------------------------------------

def make_cfg():
    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)

    # ---------------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------------

    # Igual a n_steps=1000 no Stable-Baselines3
    cfg["rollouts"] = 1000

    # Igual a n_epochs=10
    cfg["learning_epochs"] = 10

    # 100 envs × 1000 steps = 100 000 amostras por rollout.
    # 100 000 / 20 = minibatches de 5000 amostras.
    cfg["mini_batches"] = 20

    # ---------------------------------------------------------------
    # Returns e GAE
    # ---------------------------------------------------------------

    cfg["discount_factor"] = 0.999
    cfg["lambda"] = 0.95

    # ---------------------------------------------------------------
    # Otimizador
    # ---------------------------------------------------------------

    # Default usado no PPO do Stable-Baselines3
    cfg["learning_rate"] = 3e-4

    # O repositório inicial não usa scheduler KL adaptativo
    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    cfg["grad_norm_clip"] = 0.5

    # ---------------------------------------------------------------
    # PPO clipping
    # ---------------------------------------------------------------

    cfg["ratio_clip"] = 0.2

    # SB3 não usa value-function clipping por defeito
    cfg["clip_predicted_values"] = False
    cfg["value_clip"] = 0.2

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------

    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 0.5

    # Sem early stopping por KL, semelhante ao SB3 utilizado
    cfg["kl_threshold"] = 0.0

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------

    # O repositório inicial não utiliza VecNormalize.
    # O teu SAC também treina com observações raw.
    cfg["state_preprocessor"] = None
    cfg["state_preprocessor_kwargs"] = {}

    cfg["value_preprocessor"] = None
    cfg["value_preprocessor_kwargs"] = {}

    # Ajusta a escala da tua reward à reward do repositório inicial
    cfg["rewards_shaper"] = reward_shaper

    # ---------------------------------------------------------------
    # Inicialização
    # ---------------------------------------------------------------

    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    # O episódio terminar aos 5 segundos é truncatura e não morte.
    # Deve ser usado bootstrap da função de valor.
    cfg["time_limit_bootstrap"] = True

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

    cfg["experiment"] = {
        "directory": "",
        "experiment_name": "",
        "write_interval": 1000,

        # Checkpoint a cada 10 rollouts:
        # 10 × 1000 passos vetorizados
        "checkpoint_interval": 4_000,

        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }

    return cfg


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------

PRESETS = {
    # Teste inicial mais curto
    "test": {
        "models": make_models,
        "cfg": make_cfg(),

        # 50 rollouts:
        # 50 000 × 100 envs = 5 milhões de transições
        "timesteps": 50_000,

        "seed": 10,
        #"expected_num_envs": 100,
    },

    # Aproximação do treino completo do artigo
    "paper": {
        "models": make_models,
        "cfg": make_cfg(),

        # 1 000 000 passos vetorizados × 100 envs
        # = 100 milhões de transições
        "timesteps": 1_000_000,

        "seed": 10,
        #"expected_num_envs": 100,
    },
}