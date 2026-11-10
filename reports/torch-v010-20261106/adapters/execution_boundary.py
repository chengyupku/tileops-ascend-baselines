"""Explicit preparation/reset boundary; no device imports at module import.

Adapters must expose the complete mathematical operation in execute. This module
cannot decide which internal clones are protection versus required output work.
"""
from dataclasses import dataclass
from typing import Callable

@dataclass
class ExecutionBoundary:
    execute: Callable
    args: tuple
    reset: Callable
    description: dict

    def check_once(self, synchronize):
        self.reset()
        synchronize()
        result = self.execute(*self.args)
        synchronize()
        return result


def private_inputs(execute, args, indices):
    """Preserve stable tensor identities; reset copies run before profile opens."""
    indices = tuple(indices)
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate protected input index')
    working = list(args)
    for i in indices:
        if not 0 <= i < len(args):
            raise IndexError(i)
        working[i] = args[i].clone()
    def reset():
        for i in indices:
            working[i].copy_(args[i])
    return ExecutionBoundary(execute, tuple(working), reset,
                             {'protected_indices': indices,
                              'reset_outside_timing': True,
                              'internal_reference_protection_requires_separate_audit': True})


def saved_autograd(torch, forward, inputs, gradients, grad_outputs):
    """Prepare forward once; each invocation computes new gradients, never caches them.

Call only for contracts whose saved forward corresponds exactly to `inputs`.
No detach/cast or source input regeneration is performed here.
"""
    outputs = forward(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    def backward():
        return torch.autograd.grad(outputs, gradients, grad_outputs=grad_outputs,
                                   retain_graph=True, create_graph=False,
                                   allow_unused=False)
    return ExecutionBoundary(backward, (), lambda: None,
                             {'forward_outside_timing': True,
                              'backward_recomputed_each_call': True,
                              'saved_graph_reused': True})
