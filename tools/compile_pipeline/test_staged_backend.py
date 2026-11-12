import concurrent.futures
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from staged_backend import DeferredKernel, DeferredLaunch, StagedBackend


class StagingTests(unittest.TestCase):
    def test_deferred_launch_never_waits_or_executes(self):
        future = concurrent.futures.Future()
        kernel = DeferredKernel({"name": "next", "enabled": False}, future)
        with self.assertRaises(DeferredLaunch):
            kernel(1)
        self.assertFalse(future.done())
        self.assertEqual(kernel.calls, 0)

    def test_consumption_uses_worker_artifact_and_reuses_loaded_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.so"
            path.write_bytes(b"worker-result")
            built = []
            called = []

            class Kernel:
                def __init__(self, *args):
                    built.append(args)

                def __call__(self, *args):
                    called.append(args)
                    return 9

            mod = types.ModuleType("mock_kernel")
            mod.NpuInductorKernel = Kernel
            auto = types.ModuleType("mock_autotune")
            auto.NpuInductorAutoTuneKernel = Kernel
            auto.KernelInfo = lambda **x: x
            future = concurrent.futures.Future()
            future.set_result(
                {
                    "lib_wrapper": "wrapper.so",
                    "lib_kernel": str(path),
                    "kernel_key": "key",
                    "artifact_hashes": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    },
                }
            )
            kernel = DeferredKernel({"name": "next", "enabled": True}, future)
            with patch.dict(
                "sys.modules",
                {
                    "torch_npu._inductor.ascendc.compiler._kernel": mod,
                    "torch_npu._inductor.ascendc.compiler._autotune": auto,
                },
            ):
                self.assertEqual(kernel(1), 9)
                self.assertEqual(kernel(2), 9)
            self.assertEqual(built, [("wrapper.so", str(path), "key")])
            self.assertEqual(called, [(1,), (2,)])
            self.assertEqual(kernel.calls, 2)

    def test_changed_artifact_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.so"
            path.write_bytes(b"changed")
            mod = types.ModuleType("mock_kernel")
            mod.NpuInductorKernel = lambda *x: self.fail("must not load")
            auto = types.ModuleType("mock_autotune")
            auto.NpuInductorAutoTuneKernel = None
            auto.KernelInfo = None
            future = concurrent.futures.Future()
            future.set_result({"artifact_hashes": {str(path): "wrong"}})
            kernel = DeferredKernel({"name": "next", "enabled": True}, future)
            with patch.dict(
                "sys.modules",
                {
                    "torch_npu._inductor.ascendc.compiler._kernel": mod,
                    "torch_npu._inductor.ascendc.compiler._autotune": auto,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    kernel(1)

    def test_late_compilation_is_not_silently_allowed(self):
        backend = object.__new__(StagedBackend)
        backend.active = None
        with self.assertRaisesRegex(RuntimeError, "outside graph preparation"):
            backend._compile(None, {})


if __name__ == "__main__":
    unittest.main()
