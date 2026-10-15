"""Shared low-level names for the siesta_python package.

Lives below the domain modules and the `siesta_python` facade so that extracted
free functions (e.g. sip_fdf) and the config dataclass can use these without
importing the facade, which would be circular. The facade imports and
re-exports them, so existing `from siesta_python.siesta_python import ossystem`
style imports keep working.
"""
import os

ld = os.listdir


def ossystem(string):
    """Run a shell command, unless SIESTA_PYTHON_DRYRUN is set."""
    if _DRYRUN:
        return
    os.system(string)


try:
    os.environ["SIESTA_PYTHON_DRYRUN"]
    _DRYRUN = True
except KeyError:
    _DRYRUN = False


# --- default executables / paths, used as SiPConfig field defaults ---
wd = os.getcwd()
PP_path = '../pp'

sp             = 'siesta '
tb_trans_exec  = 'tbtrans '
B              = 'gnubands '
pdos_exec      = 'fmpdos'
gulp_path      = 'gulp '
gen_basis      = 'gen-basis '
mix_ps         = 'mixps '
joblib_backend = 'multiprocessing'

M = ''
try:
    M = os.environ['mpi_exec']
except KeyError:
    pass
