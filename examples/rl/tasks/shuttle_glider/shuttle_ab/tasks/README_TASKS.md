# Tasks A/B/C do 5.o rotor -- instalacao e uso

## O que sao

Tres condicoes experimentais para a ablacao do 5.o rotor do shuttle_glider:

| Task | rotor5_mode | Acoes | Obs | O que testa |
| --- | --- | --- | --- | --- |
| `sg_ab_A_active` | `active` | 5 | 28 | Baseline. A policy controla o rotor 5. |
| `sg_ab_B_dead` | `dead` | 5 | 28 | Canal 5 forcado a -1 ANTES da reward e da obs. |
| `sg_ab_C_removed` | `removed` | 4 | 26 | A policy nem sequer tem a dimensao. CONTROLO LIMPO. |

B e C respondem a perguntas diferentes. B mantem a arquitetura (mesmas
dimensoes, mesmo checkpoint carregavel) e mede o que acontece quando o
rotor deixa de responder. C retreina sem a dimensao e mede se ela alguma vez
foi necessaria. Precisas dos dois.

> A tua versao antiga de "desligar o rotor 5" zerava apenas o omega dentro do
> `_apply_action`. Isso deixa tres canais parasitas: custo de `d_action`
> fantasma, ruido da propria policy realimentado na observacao via
> `_action_history_obs[:, 4]`, e um sumidouro de entropia no SAC
> (`target_entropy = -dim(A) = -5`). Nao e' um controlo limpo -- e' a origem
> das oscilacoes extra que observaste. O modo `dead` corrige isto.

## Instalacao

```bash
cp -r tasks/shuttle_glider/sg_ab_* ~/dev_skrl/examples/rl/tasks/shuttle_glider/
```

Ficas com quatro pastas novas: as tres condicoes mais `sg_ab_common`, que
contem o env verdadeiro (`env_core.py`), a trajetoria e o gravador. As tres
condicoes sao shims de ~20 linhas que so' fixam defaults do cfg.

### Porque ha' um `sg_ab_common`

O `_load_env()` do `skrl_agent_controller.py` faz:

```python
env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]
env_file = env_files[0]
...
return getattr(mod, cls_name), getattr(mod, cls_name + "Cfg")()
```

Duas consequencias:

1. **Um unico `*_env.py` por pasta de task.** Se houver dois, o escolhido
   depende da ordem do `os.listdir` -- nao deterministico. Por isso o nucleo
   chama-se `env_core.py` e vive fora das pastas de task.
2. **Nao ha' overrides de cfg vindos do manifesto JSON.** O cfg e'
   instanciado sem argumentos. E' exatamente por isso que cada condicao
   precisa da sua propria pasta, em vez de um campo no `ab_rotor5.json`.

### Um detalhe que teria partido a condicao C em silencio

`PegasusEnvCfg` e' um `@dataclass` **sem** `__post_init__`, portanto o
`__init__` gerado nunca chamaria o `__post_init__` da subclasse. Como o
`_load_env()` faz apenas `QuadcopterEnvCfg()`, o `action_space` e o
`observation_space` ficariam nos defaults de classe (5 e 28) e a condicao
`removed` -- que precisa de 4 e 26 -- carregaria o checkpoint com as
dimensoes erradas. O `env_core.py` define agora um `__init__` explicito que
chama o `__post_init__`. Nao mexas nos campos `action_space` e
`observation_space` a' mao: sao derivados.

## Treino

Cinco seeds por condicao, quinze corridas no total:

```bash
cd ~/dev_skrl
for cond in sg_ab_A_active sg_ab_B_dead sg_ab_C_removed; do
  for s in 0 1 2 3 4; do
    ./python.sh examples/rl/train.py --headless \
        --task shuttle_glider/$cond --algo sac --preset isaac_lab --seed $s
  done
done
```

Ajusta as flags ao teu `train.py`. Se quiseres que o `run_tag` do gravador
interno distinga seeds, poe `seed_tag` no shim -- mas para a avaliacao com o
`play_multi.py` isso e' irrelevante, porque quem define a seed e' a flag
`--seed` do `csv_to_tost.py`.

## Avaliacao

Preenche os tres `<RUN_?>` no `ab_rotor5.json` com os caminhos dos
checkpoints e corre as tres condicoes **na mesma simulacao**, para ficarem
emparelhadas no mesmo mundo:

```bash
for s in 0 1 2 3 4; do
  ./python.sh examples/rl/play_multi.py --headless \
      --vehicle shuttle_glider --config examples/rl/scripts/ab_rotor5.json \
      --trajectory lemniscate --traj_amplitude 1.5 --traj_period 8 --traj_z 1.5 \
      --episode_duration 10 --num_episodes_per_env 20 \
      --record --record_dir play_records/lemn_seed$s

  python3 csv_to_tost.py --records play_records/lemn_seed$s \
      --ref-type lemniscate --traj-amplitude 1.5 --traj-period 8 --traj-z 1.5 \
      --seed $s --out results_eval.csv --append
done

python3 eval_tost.py --csv results_eval.csv \
    --baseline A_active --ablation C_removed \
    --metric rmse_pos --margin-frac 0.10 --by traj_type --plot ab.png
```

O campo `name` de cada veiculo no manifesto vira a coluna `controller` do
CSV, que o `csv_to_tost.py` converte em `run_tag`. Nao lhes mudes os nomes
sem mudar tambem os argumentos `--baseline` / `--ablation`.

## Contrafactual (sem retreinar)

Para medir se a solucao aprendida em A e' load-bearing no rotor 5, aponta o
**mesmo** checkpoint de A para a task `sg_ab_B_dead`. Funciona porque `dead`
mantem 5 acoes e 28 observacoes.

- Degradacao proxima de zero -> a policy nunca dependeu do rotor 5.
- Degradacao grande -> o rotor 5 fazia alguma coisa, e o C limitou-se a
  reaprender sem ele.

Sao afirmacoes diferentes. Queres as duas na tese.

## Duas coisas a confirmar na primeira corrida a serio

1. **`ref_vx` / `ref_vy` nao podem vir todos a zero** num lemniscate. O
   `make_reference_provider()` do `play_multi.py` devolve velocidade
   identicamente nula, enquanto o `_update_goal()` escreve a velocidade
   analitica em `reset_manager._goal_vel`. Se o CSV trouxer zeros, o
   `rmse_vel` e o `heading_err_deg` nao significam nada.
2. **O `_prime_motors()` da' um pontape de ~26 N no rotor 5 em cada reset**
   de avaliacao. Com `--min-steps 50` o `csv_to_tost.py` ja' descarta o
   transitorio, mas confirma que o `thrust5_impulse_frac` nao esta' a ser
   dominado por esse arranque.
