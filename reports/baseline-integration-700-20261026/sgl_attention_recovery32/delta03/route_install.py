def install(provider):
    from benchmarks.ascend_all_pairs import sgl
    for op,factory in provider.PROVIDERS.items():
        sgl.PROVIDERS[op]=factory
