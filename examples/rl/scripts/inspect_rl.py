"""
Corre este script no teu ambiente Isaac Sim para descobrir
o formato correcto do rsl_rl instalado.

Uso:
    python inspect_rslrl.py
"""
import sys

try:
    import rsl_rl
    print(f"rsl_rl location: {rsl_rl.__file__}")
    print(f"rsl_rl version:  {getattr(rsl_rl, '__version__', 'unknown')}")
except ImportError:
    print("rsl_rl not found")
    sys.exit(1)

print()

# Listar todos os módulos disponíveis
import pkgutil, importlib
print("=== Modules in rsl_rl ===")
for info in pkgutil.walk_packages(rsl_rl.__path__, prefix="rsl_rl."):
    print(f"  {info.name}")

print()

# Tentar importar ActorCritic de vários lugares
candidates = [
    "rsl_rl.modules.actor_critic",
    "rsl_rl.modules",
    "rsl_rl.models",
    "rsl_rl.models.actor_critic",
    "rsl_rl.actors",
    "rsl_rl.networks",
]
print("=== ActorCritic search ===")
for mod_path in candidates:
    try:
        mod = importlib.import_module(mod_path)
        classes = [name for name in dir(mod) if "Actor" in name or "Critic" in name or "MLP" in name]
        if classes:
            print(f"  {mod_path}: {classes}")
        else:
            print(f"  {mod_path}: (no Actor/Critic/MLP classes)")
    except ImportError:
        print(f"  {mod_path}: ImportError")

print()

# Ver o source de construct_algorithm para perceber o formato exacto
print("=== ppo.py construct_algorithm (lines 460-510) ===")
try:
    import inspect
    from rsl_rl.algorithms import ppo as rsl_ppo
    src = inspect.getsource(rsl_ppo)
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'construct_algorithm' in line or 'actor' in line.lower() or 'class_name' in line:
            print(f"  {i+1:4d}: {line}")
except Exception as e:
    print(f"  Error: {e}")

print()

# Ver o resolve_callable para perceber o que aceita
print("=== resolve_callable source ===")
try:
    from rsl_rl.utils.utils import resolve_callable
    print(inspect.getsource(resolve_callable))
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== MLPModel signature ===")
try:
    from rsl_rl.models import MLPModel
    import inspect
    sig = inspect.signature(MLPModel.__init__)
    print(f"  MLPModel.__init__{sig}")
    # Mostrar os primeiros 30 linhas do source
    src = inspect.getsource(MLPModel)
    for i, line in enumerate(src.split('\n')[:40]):
        print(f"  {i+1:3d}: {line}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== on_policy_runner.py __init__ ===")
try:
    from rsl_rl.runners import OnPolicyRunner
    import inspect
    src = inspect.getsource(OnPolicyRunner.__init__)
    print(src[:2000])
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== Distribution classes in rsl_rl.modules.distribution ===")
try:
    from rsl_rl.modules import distribution as dist_mod
    import inspect
    classes = [name for name in dir(dist_mod) if not name.startswith("_")]
    print(f"  All names: {classes}")
    for name in classes:
        obj = getattr(dist_mod, name)
        if inspect.isclass(obj):
            print(f"  CLASS: {name}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== mlp_model.py distribution section ===")
try:
    from rsl_rl.models import mlp_model
    import inspect
    src = inspect.getsource(mlp_model)
    for i, line in enumerate(src.split('\n')[60:100], start=61):
        print(f"  {i:3d}: {line}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== GaussianDistribution signature ===")
try:
    from rsl_rl.modules.distribution import GaussianDistribution
    import inspect
    sig = inspect.signature(GaussianDistribution.__init__)
    print(f"  GaussianDistribution.__init__{sig}")
    src = inspect.getsource(GaussianDistribution.__init__)
    print(src[:800])
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== on_policy_runner.py learn() full source ===")
try:
    from rsl_rl.runners import OnPolicyRunner
    import inspect
    src = inspect.getsource(OnPolicyRunner.learn)
    print(src)
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== Logger source ===")
try:
    from rsl_rl.utils.logger import Logger
    import inspect
    src = inspect.getsource(Logger)
    print(src[:4000])
except Exception as e:
    print(f"  Error: {e}")