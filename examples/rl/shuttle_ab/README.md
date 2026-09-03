# Shuttle_glider — pacote de experiências A/B/C do 5.º rotor

| Ficheiro | O que é |
|---|---|
| `quadcopter_env_v2.py` | Env instrumentado. Presets `baseline`/`heading`, ablação limpa, clip corrigido, heading por look-ahead, métricas de diagnóstico. |
| `trajectory_v2.py` | Drop-in do `trajectory.py` com `lookahead()`, `set_generator()` (referências deterministas) e `is_langevin()`. |
| `eval_recorder.py` | Escreve métricas **por episódio** num CSV. Agnóstico ao runner. |
| `eval_tost.py` | Cluster bootstrap sobre seeds + **teste de equivalência TOST**. Sem scipy. |
| `make_fake_results.py` | CSV sintético para testar o `eval_tost.py` sem gastar GPU. |

---

## 0. Instalação

```bash
cp quadcopter_env_v2.py trajectory_v2.py eval_recorder.py <.../raptor_pretrain>/
```

O env importa `from .trajectory_v2 import RaptorLikeTrajectory` e
`from .eval_recorder import EpisodeRecorder`. Se preferires substituir o
`trajectory.py` original, renomeia e ajusta o import.

## 1. Testar a análise estatística agora (30 s, sem GPU)

```bash
python3 make_fake_results.py --out fake_null.csv --effect 0.00
python3 make_fake_results.py --out fake_eff.csv  --effect 0.25

python3 eval_tost.py --csv fake_null.csv --by traj_type --plot ab_null.png
python3 eval_tost.py --csv fake_eff.csv  --by traj_type
```

Esperado: o primeiro dá **EQUIVALENCIA ESTABELECIDA**, o segundo dá
**DIFERENCA RELEVANTE**. Se for isso, o pipeline está bom.

## 2. Passo 0 obrigatório — validar a física antes de treinar

O env chama `assert_physics_sane()` no fim do `setup()` e imprime:

```
[diag] rotor_axes_body = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,0]]
[diag] thrust5 @max = 104.72 N | @neutro(a=0) = 26.18 N
```

**Se o eixo do rotor 5 não for ~`[1,0,0]`, para tudo.** O teu env baseline de
28D **não tinha** a chave `rotor_axes_body`; se o simulador assumiu `[0,0,1]`,
o rotor 5 estava a empurrar para **cima** à frente do CoM (105 N + momento de
pitch enorme) — e zerá-lo era a única solução racional. Nesse caso, todas as
conclusões anteriores sobre o rotor 5 estão contaminadas.

Validação em malha aberta (5 min, vale mais que 5 h de treino): fixa
`a = [0,0,0,0, u5]` com `u5 ∈ {-1, -0.5, 0, 0.5, 1}`, mede `a_x` no frame do
corpo e compara com `kf5·ω5²/m`. Se não bater, o bug é no simulador, não na RL.

## 3. E1 — ablação limpa (a prova formal)

Três condições, ≥5 seeds cada:

```python
from quadcopter_env_v2 import cfg_E1
cfg = cfg_E1("A_active",  seed=s, csv="results_E1.csv")  # 5 ações
cfg = cfg_E1("B_dead",    seed=s, csv="results_E1.csv")  # 5 ações, canal 5 = -1
cfg = cfg_E1("C_removed", seed=s, csv="results_E1.csv")  # 4 ações  <- controlo limpo
```

Porque três e não duas: a tua ablação antiga (`omega[:,4]=0` só no
`_apply_action`) é a condição **B mal feita** — deixava o canal 5 a poluir o
custo Δaction e a realimentar-se na observação, e no SAC a dimensão morta
funciona como sumidouro de entropia (`target_entropy = −dim(A)`), reduzindo a
exploração efetiva nos 4 canais úteis. **B vs C isola exatamente esse
artefacto** — é a explicação das oscilações extra que observaste, não
compensação de força (com `omega5 = 0` não há força nem torque para compensar).

Avaliação: fixa as referências para as três condições verem as mesmas
trajetórias.

```python
cfg.test_mode = True
cfg.trajectory_eval_seed = 12345   # igual em A, B e C
```

Análise:

```bash
python3 eval_tost.py --csv results_E1.csv --baseline A_active --ablation C_removed \
    --metric rmse_pos --margin-frac 0.10 --by traj_type --min-episode-idx 2 --plot E1.png
python3 eval_tost.py --csv results_E1.csv --baseline B_dead --ablation C_removed --by traj_type
```

## 4. E2 — contrafactual sem retreinar (barato, muito convincente)

Carrega o checkpoint de **A** e avalia com `cfg.rotor5_mode = "dead"`.

* degradação ≈ 0 → a solução aprendida **não é load-bearing** no rotor 5;
* degradação grande → o rotor 5 fazia algo (bias de pitch) e o retreino de C
  simplesmente reaprendeu sem ele.

São afirmações diferentes; queres as duas na tese.

## 5. E3 — env de heading reparado

```python
from quadcopter_env_v2 import cfg_E3
cfg = cfg_E3(seed=s, w_heading=0.0, csv="results_E3.csv")   # sanity check
cfg = cfg_E3(seed=s, w_heading=0.5, csv="results_E3.csv")   # com rampa
```

**Corre primeiro com `w_heading=0.0`.** Se com heading a zero o RMSE não
recuperar o nível do baseline, o problema não é o termo de heading — é uma das
outras alterações que fizeste ao mesmo tempo (`mixture_langevin_prob 0.5→0.9`,
`max_rotor_velocity[4] 3500→1000`, obs em body-frame, `advance()`). Só depois
liga a rampa.

O que foi corrigido em relação ao teu env de heading:

1. **`pos_cost` fora do clip** (`clip_mode="shaping_only"`). Era este o bug
   principal: com os termos de shaping somados, `cost` batia em `cost_clip=3.0`
   e `∂r/∂pos_error = 0` → zero sinal de posição. É exatamente por isso que
   deixava de voar para pontos fora da zona de treino.
2. **Heading por look-ahead filtrado + histerese** em vez do `vel_ref`
   instantâneo. Com `γ=1, ω=2, σ=0.5` o Langevin dá `σ_v ≈ 0.35 m/s` e
   `σ_x ≈ 0.18 m`: a direção de `vel_ref` é quase puro ruído, e o replay
   ping-pong inverte a velocidade a meio (`vel = -vel`) → flips instantâneos de
   180°. O alvo era fisicamente inatingível.
3. **Rampa de aquecimento** para `w_heading` (aprende a voar primeiro).
4. `trajectory_mixture_langevin_prob` de volta a **0.5** (0.9 removia o
   curriculum fácil do hover).
5. `advance()` removido (era código morto: `current()` recebe sempre
   `step_ids = episode_length_buf`).
6. Reset consistente: `_last_action` e `_action_history_obs` passam a ser o
   mesmo vetor (antes, `hist=reset_rotor_norm` e `last=0` davam um primeiro
   Δaction artificialmente enorme).
7. `max_rotor_velocity[4]` de volta a **3500** (baixar para 1000 não resolve
   nada e reduz a autoridade).

## 6. E4 — mudança de envelope (onde o rotor 5 passa a ser útil)

Nada disto faz o rotor 5 ser útil se o envelope não mudar. Com
`max_lin_vel_per_axis = 2.0 m/s`:

* força da deriva vertical: `q = ½·1.225·2² = 2.45 Pa` →
  `F ≈ 2.45 × 0.0221 × 0.8 ≈ 0.04 N` = **0.15 % do peso** (~30 N). Com 5 m/s de
  vento, ~0.27 N. Por isso as tuas três versões aerodinâmicas deram o mesmo
  resultado — a mudança está abaixo do ruído numérico;
* o rotor 5 dá **105 N** ao máximo e **26 N** já na ação neutra (`a=0`), o que
  produz ≈8 m/s² → viola o limite de 2 m/s em ~0.25 s. O atuador é
  estruturalmente inutilizável neste envelope: o ótimo é `u5 = −1`, onde o
  gradiente do tanh desaparece.

Para o rotor 5 ser a solução ótima precisas de:

* modelar a **asa** (`L = q·S_wing·CL(α)`) e o arrasto do fuselage;
* `max_lin_vel_per_axis` para **15–25 m/s**, episódios de **15–30 s**,
  referências com retas longas / raios grandes;
* **custo de potência** (`w_power·Σ(kf·ωi²)^{3/2}` ou `Σωi³`) — sem ele,
  inclinar o quadrotor é gratuito;
* penalizar/limitar a inclinação do quadrotor;
* só depois adicionar o heading.

Alternativa forçada (se quiseres o comportamento já, sem esperar pela
emergência): `u5 = u5_ff(v_ref) + Δ_policy` (feedforward + residual), curriculum
de alocação, bónus de alinhamento multiplicativo **fora do clip**, e redesenhar
o mapeamento da ação 5 (com `min_w=0`, empuxo nulo exige `a=−1`, na saturação
do tanh).

## 7. Métricas que o env já registra

TensorBoard: `Metrics/rmse_pos`, `rmse_vel`, `thrust5_mean_N`,
`thrust5_impulse_frac`, `tail_force_mean_N`, `clip_frac`,
`heading_valid_frac`, `tilt_zz_mean`, `power_mean`, `a5_mean`, `a5_std`.

Critérios de leitura:

* `clip_frac > 0.2` → o clip está a matar gradiente. Baixa pesos ou sobe
  `shaping_clip`.
* `heading_valid_frac < 0.3` → o termo de heading quase nunca liga; o
  look-ahead ou a trajetória são curtos demais.
* `a5_mean ≈ −1` e `a5_std ≈ 0` → policy convergiu para desligar o rotor 5.
* `a5_std` grande em `B_dead`/`removed` → ainda há canal morto a injetar ruído.
* `thrust5_impulse_frac` → a fração do impulso total entregue pelo rotor 5. É
  este o número que fecha o argumento na tese.
