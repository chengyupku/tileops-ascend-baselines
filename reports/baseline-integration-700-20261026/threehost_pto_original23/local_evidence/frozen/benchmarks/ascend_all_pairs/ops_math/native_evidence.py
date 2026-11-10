"""Optional observational receipts for the actual resolved native API and calls."""
import ctypes as C
import json
import os
import time
from pathlib import Path


def _write(record):
    path = os.environ.get('OPS_MATH_NATIVE_API_RECORDS')
    if path:
        with open(path, 'a') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(),
                                         time_ns=time.time_ns(), **record)) + '\n')


def bound_symbol(library, name, function):
    if not os.environ.get('OPS_MATH_NATIVE_API_RECORDS'):
        return
    class DlInfo(C.Structure):
        _fields_ = [('fname', C.c_char_p), ('base', C.c_void_p),
                    ('sname', C.c_char_p), ('address', C.c_void_p)]
    dladdr = C.CDLL(None).dladdr
    dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dladdr.restype = C.c_int
    address = C.cast(function, C.c_void_p).value
    info = DlInfo()
    rc = dladdr(C.c_void_p(address), C.byref(info))
    _write(dict(event='native_symbol_resolved', library=library._name, symbol=name,
                address=address, dladdr_rc=rc,
                loaded_file=os.fsdecode(info.fname) if info.fname else None,
                loaded_symbol=os.fsdecode(info.sname) if info.sname else None))


def graph_call(graph, rc):
    # This records API completion, not driver consumed-byte attestation.
    _write(dict(event='aclopCompileAndExecute_returned', graph=graph, returncode=rc))
