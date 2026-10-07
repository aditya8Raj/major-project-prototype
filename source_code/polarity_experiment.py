"""
Does the result depend on the leak events being ON events only?  (Chapter 7)

In simulator.py a leak event always has ON polarity. A classifier can use this: noise groups are almost all ON, moth groups
are mixed. Here the simulator source is copied in memory with one change: a leak event is ON or OFF with equal chance.
Everything else (scenes, seeds, pipeline, classifier, scoring) is the same as in run_experiments.py, which is run again in full
(training on 18 scenes with the new noise, testing on the 18 test scenes with the new noise).

Writes results_symmetric_leak/results.json.   Run:  python polarity_experiment.py
"""
import os, types
import simulator as S0
import run_experiments as R

src = open(S0.__file__.replace('.pyc', '.py') if S0.__file__.endswith('.pyc') else S0.__file__).read()
OLD = 'd = th + 1e-6'
assert OLD in src
src = src.replace(OLD, 'd = (th + 1e-6) if np.random.random() < 0.5 else -(th + 1e-6)').replace('cache=True', 'cache=False')
mod = types.ModuleType('simulator_symmetric_leak')
mod.__file__ = 'simulator_symmetric_leak.py'
exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
R.S = mod
R.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_symmetric_leak')
os.makedirs(R.OUT, exist_ok=True)
R.main()
