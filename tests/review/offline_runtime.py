"""
| File: offline_runtime.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Load simulator-independent production modules without Isaac package initializers.
"""
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def enable():
    base=ROOT/'extensions/pegasus.simulator'
    for name in ('pegasus','pegasus.simulator','pegasus.simulator.logic','pegasus.simulator.logic.rl','pegasus.simulator.logic.rl.algorithms','pegasus.simulator.logic.dynamics'):
        module=types.ModuleType(name);module.__path__=[str(base.joinpath(*name.split('.')))];sys.modules.setdefault(name,module)
    sys.path.insert(0,str(ROOT/'examples/rl'))
