"""cifar.py with KronQ's basis refresh period set from the command line (refresh study, 2026-10-15). Does not edit cifar.py or kronq_lm.py:
sets KronQLM's refresh / adapt_refresh defaults (cifar.py passes neither), then runs cifar.py unchanged.
  python bench/cifar_kx.py --refresh_fixed N <cifar.py arguments>     N > 0: refresh every N steps (no adaptation); 0: KronQ's adaptive rule
"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'llm')]
import kronq_lm

argv = sys.argv[1:]; n = 0
if '--refresh_fixed' in argv:
    i = argv.index('--refresh_fixed'); n = int(argv[i + 1]); del argv[i:i + 2]
if n > 0:
    _init = kronq_lm.KronQLM.__init__
    def _patched(self, *a, **k):
        k.setdefault('refresh', n); k.setdefault('adapt_refresh', 0.0); _init(self, *a, **k)
    kronq_lm.KronQLM.__init__ = _patched
sys.argv = [os.path.join(HERE, 'cifar.py')] + argv
runpy.run_path(sys.argv[0], run_name='__main__')
