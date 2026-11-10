"""Keep this task's guarded command tied to its immediate supervisor lifetime."""
import ctypes
import os
import signal
import sys

if len(sys.argv) < 2:
    raise SystemExit("Usage: task_exec.py COMMAND [ARG...]")
parent = os.getppid()
if parent == 1:
    raise SystemExit("Supervisor already gone")
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int
if libc.prctl(1, signal.SIGTERM, 0, 0, 0):  # PR_SET_PDEATHSIG
    raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
if os.getppid() != parent:
    raise SystemExit("Supervisor changed during startup")
os.execvp(sys.argv[1], sys.argv[1:])
