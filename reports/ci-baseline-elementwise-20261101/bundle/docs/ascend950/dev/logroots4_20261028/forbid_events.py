"""Reject Event construction while preserving framework class introspection."""

def install(torch, torch_npu):
    def forbidden(*args, **kwargs):
        raise AssertionError('Event construction/timing forbidden; use isolated msprof')

    # Dynamo interfaces may capture the class before torch.npu.Event is patched.
    # Patch the original Python subclass constructor, which also covers such
    # aliases, without requesting a lazily registered device interface early.
    event_type = torch_npu.npu.streams.Event
    event_type.__new__ = staticmethod(forbidden)
    torch.npu.Event = event_type
    torch_npu.npu.Event = event_type
    cuda_event = torch.cuda.Event
    if isinstance(cuda_event, type):
        cuda_event.__new__ = staticmethod(forbidden)
    else:
        torch.cuda.Event = forbidden
    return {'npu_event_class': event_type.__module__ + '.' + event_type.__qualname__,
            'construction_forbidden': True}
