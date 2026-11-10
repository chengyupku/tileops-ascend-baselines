"""Explicit process-wide evaluation variant. Never invoked by candidate factory.
Use only in a dedicated checker process, around the entire unchanged checker.
"""
from contextlib import contextmanager
import datetime,platform,os

@contextmanager
def evaluation_backend(torch, record):
    original=torch.backends.mkldnn.enabled
    bmm=torch.bmm
    def event(phase):
        record(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,
            setting='torch.backends.mkldnn.enabled=False',mkldnn_enabled=torch.backends.mkldnn.enabled,
            original_mkldnn_enabled=original,bmm_identity=id(torch.bmm),bmm_unchanged=torch.bmm is bmm,
            torch_version=str(torch.__version__),num_threads=torch.get_num_threads(),
            hostname=platform.node(),pid=os.getpid(),qualification='parent-approved explicit runtime variant only'))
    try:
        event('before_configuration')
        torch.backends.mkldnn.enabled=False
        event('configured')
        if torch.get_num_threads()!=4:raise RuntimeError('Evaluation requires preconfigured 4 CPU threads')
        yield
        if torch.bmm is not bmm:raise RuntimeError('Original torch.bmm identity changed')
    finally:
        torch.backends.mkldnn.enabled=original
        event('restored')
