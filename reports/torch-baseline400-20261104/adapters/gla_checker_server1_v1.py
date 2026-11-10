import torch
def chunkwise_verification(dtype, *, backward=False):
    from workloads.numerics import Custom, Exact, compare_outputs

    if not backward:
        tol = {torch.float32: 1e-2, torch.float16: 5e-2, torch.bfloat16: 1e-1}[dtype]

        def validate_forward(got, expected):
            compare_outputs(got, expected, Exact(atol=tol, rtol=tol))
            actuals = (got,) if isinstance(got, torch.Tensor) else got
            references = (expected,) if isinstance(expected, torch.Tensor) else expected
            for actual, reference in zip(actuals, references, strict=True):
                a, b = actual.double().flatten(), reference.double().flatten()
                if torch.count_nonzero(b):
                    # Absolute slack alone must not accept zero for low-amplitude outputs.
                    cosine = torch.dot(a, b) / (
                        torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                    )
                    assert cosine > 0.99

        return Custom(validate_forward, "forward error bound and nonzero signal alignment")

    def validate(got, expected):
        # Gradients are about 1e-2; bound each by 1% of its largest reference entry.
        for actual, reference in zip(got, expected, strict=True):
            torch.testing.assert_close(
                actual, reference, atol=1e-2 * reference.abs().max().item(), rtol=0
            )

    return Custom(validate, "gradient error bounded by 1% of each gradient's amplitude")

