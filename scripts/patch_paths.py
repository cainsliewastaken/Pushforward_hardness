"""Make L63 scripts location-independent after copying them from $SCRATCH/l63_held_rough/code.

- replaces the literal /pscratch/.../l63_held_rough by L63_ROOT (env, default $SCRATCH/l63_held_rough);
- in optimizer/, extends the sys.path line so the shared modules in chain/ are importable;
- byte-compiles every script (run inside the Shifter image: the system python is too old for the f-strings).

usage: shifter --image=nersc/pytorch:26.01.01 python -I scripts/patch_paths.py <repo root>
"""
import re, sys, glob, os, py_compile
repo = sys.argv[1]
OLD = '/pscratch/sd/c/cainslie/l63_held_rough'
DEF = "import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch"
SYS_OLD = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
SYS_NEW = "sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late"
changed = []
for sub in ('chain', 'optimizer'):
    for f in sorted(glob.glob(os.path.join(repo, sub, '*.py'))):
        src = open(f).read(); orig = src
        if sub == 'optimizer' and SYS_OLD in src:
            src = src.replace(SYS_OLD, SYS_NEW)
        if OLD in src:
            src = re.sub(r"""(f['"])""" + re.escape(OLD), r"\1{L63_ROOT}", src)
            src = re.sub(r"""(?<!f)(['"])""" + re.escape(OLD) + r"""([^'"]*)(['"])""", r"L63_ROOT + \1\2\3", src)
            assert OLD not in src, f
            if 'L63_ROOT = ' not in src:
                lines = src.split('\n')
                for i, l in enumerate(lines):
                    if l.startswith('import ') or l.startswith('from '):
                        lines.insert(i + 1, DEF); break
                else:
                    raise SystemExit('no import line in ' + f)
                src = '\n'.join(lines)
        if src != orig:
            open(f, 'w').write(src); changed.append(os.path.relpath(f, repo))
        py_compile.compile(f, doraise=True)
print('patched:', len(changed)); print('\n'.join(changed))
