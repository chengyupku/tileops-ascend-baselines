"""Lossless CPU phase packets and exclusive, content-addressed receipts.

No runtime is imported at module scope. Packets contain primitive Python values
and CPU uint8 tensors only, and are read with torch.load(weights_only=True).
They are an audit format, not an authorization or a signature from another user.
"""
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    with Path(path).open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def packet(value):
    """Capture all bytes once per storage, including views, offsets and aliases.

    On device this performs D2H, never hashing or numerical comparisons. Callers
    must synchronize before publication. Unsupported layouts fail explicitly.
    """
    import torch
    storages, tensor_ids, storage_ids, tensors = [], {}, {}, []

    def visit(v):
        if isinstance(v, torch.Tensor):
            if type(v) is not torch.Tensor or v.layout != torch.strided or v.is_quantized:
                raise TypeError("phase packets require ordinary strided tensors")
            if v.is_conj() or v.is_neg() or v.requires_grad:
                raise TypeError("conjugate/negative/autograd views need the original checker")
            if id(v) in tensor_ids:
                return ["tensor", tensor_ids[id(v)]]
            storage = v.untyped_storage()
            key = (str(v.device), storage._cdata)
            if key not in storage_ids:
                storage_ids[key] = len(storages)
                raw = torch.empty(0, dtype=torch.uint8, device=v.device).set_(
                    storage, 0, (storage.nbytes(),), (1,))
                storages.append(raw.to("cpu", copy=True, non_blocking=False))
            i = len(tensors)
            tensor_ids[id(v)] = i
            tensors.append(dict(storage=storage_ids[key], shape=list(v.shape),
                                stride=list(v.stride()), offset=v.storage_offset(),
                                dtype=str(v.dtype).removeprefix("torch."),
                                device=str(v.device), layout=str(v.layout)))
            return ["tensor", i]
        if v is None:
            return ["none"]
        if type(v) in (bool, int, str):
            return [type(v).__name__, v]
        if type(v) is float:
            return ["float", v.hex()]
        if isinstance(v, torch.dtype):
            return ["dtype", str(v).removeprefix("torch.")]
        if isinstance(v, torch.device):
            return ["device", str(v)]
        if type(v) in (tuple, list):
            return [type(v).__name__, [visit(x) for x in v]]
        if type(v) is dict:
            return ["dict", [[visit(k), visit(x)] for k, x in v.items()]]
        raise TypeError(f"unsupported non-tensor argument/output: {type(v).__name__}")

    graph = visit(value)
    return dict(version=1, graph=graph, tensors=tensors, storages=storages)


def unpack(p, device="cpu"):
    """Reconstruct storage sharing, tensor identity, strides and typed scalars."""
    import torch
    if p["version"] != 1:
        raise ValueError("unknown packet version")
    bases = []
    for raw in p["storages"]:
        if raw.device.type != "cpu" or raw.dtype != torch.uint8 or raw.ndim != 1:
            raise ValueError("packet storage must be a flat CPU byte tensor")
        bases.append(raw.to(device, copy=True, non_blocking=False))
    tensors = []
    for meta in p["tensors"]:
        dtype = getattr(torch, meta["dtype"])
        tensors.append(torch.empty(0, dtype=dtype, device=device).set_(
            bases[meta["storage"]].untyped_storage(), meta["offset"],
            tuple(meta["shape"]), tuple(meta["stride"])))

    def visit(node):
        kind = node[0]
        if kind == "tensor":
            return tensors[node[1]]
        if kind == "none":
            return None
        if kind == "float":
            return float.fromhex(node[1])
        if kind == "dtype":
            return getattr(torch, node[1])
        if kind == "device":
            return torch.device(node[1])
        if kind in ("bool", "int", "str"):
            return node[1]
        if kind in ("tuple", "list"):
            items = [visit(x) for x in node[1]]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {visit(k): visit(v) for k, v in node[1]}
        raise ValueError("unknown packet node")
    return visit(p["graph"])


def contract(p):
    """Everything except device and bytes: preserves aliases and scalar values."""
    return dict(graph=p["graph"], tensors=[{k: v for k, v in m.items() if k != "device"}
                                         for m in p["tensors"]],
                storage_bytes=[s.numel() for s in p["storages"]])


def fingerprint(p):
    """Complete storage bytes plus complete metadata, computed only on CPU."""
    hashes = [hashlib.sha256(memoryview(s.numpy())).hexdigest() for s in p["storages"]]
    info = contract(p) | {"storage_sha256": hashes}
    return info | {"content_sha256": hashlib.sha256(canonical(info)).hexdigest(),
                   "source_devices": [m["device"] for m in p["tensors"]]}


def assert_preserved(before, after):
    """Reuse the original preservation check, then check layout and all bytes."""
    import torch
    from workloads.numerics import _require_unchanged, _unique
    if contract(before) != contract(after):
        raise AssertionError("input shape/dtype/stride/offset/alias/scalar contract changed")
    a, b = unpack(before), unpack(after)
    _require_unchanged(_unique(b), _unique(a))
    if not all(torch.equal(x, y) for x, y in zip(before["storages"], after["storages"], strict=True)):
        raise AssertionError("input storage bits changed (including signed zeros and padding)")


def replace_storage_in_place(target, source):
    """Same live device tensor identities for Add's cached-input edge check."""
    import torch
    # Metadata capture without D2H; storage traversal follows packet's order.
    seen = set()
    storages = []
    from workloads.numerics import _tensors
    for t in _tensors(target):
        key = (str(t.device), t.untyped_storage()._cdata)
        if key not in seen:
            seen.add(key)
            raw = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                t.untyped_storage(), 0, (t.untyped_storage().nbytes(),), (1,))
            storages.append(raw)
    if len(storages) != len(source["storages"]):
        raise ValueError("edge storage alias topology mismatch")
    for dst, src in zip(storages, source["storages"], strict=True):
        dst.copy_(src, non_blocking=False)


def save_packet(path, p):
    import torch
    with Path(path).open("xb") as f:
        torch.save(p, f)
        f.flush()
        os.fsync(f.fileno())


def load_packet(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def seal(directory, stage, metadata):
    """One exclusive receipt per stage; partial/unsealed dirs are never valid."""
    directory = Path(directory)
    if list(directory.glob("receipt-*.json")):
        raise FileExistsError("stage already sealed")
    # O_EXCL also excludes two concurrent sealers with different metadata/hashes.
    # An interrupted seal stays unsealed; retry in a NEW stage directory.
    write_new(directory / "SEAL.claim", canonical({"stage": stage}) + b"\n")
    files = {}
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
        if p.is_file():
            files[str(p.relative_to(directory))] = dict(sha256=sha_file(p), bytes=p.stat().st_size)
    body = dict(schema="ascend-pair-phases-v1", stage=stage, files=files, metadata=metadata)
    digest = hashlib.sha256(canonical(body)).hexdigest()
    path = directory / f"receipt-{digest}.json"
    write_new(path, canonical(body) + b"\n")
    return path


def read_receipt(path, stage=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("receipt symlinks are forbidden")
    body = json.loads(path.read_bytes())
    digest = hashlib.sha256(canonical(body)).hexdigest()
    if path.name != f"receipt-{digest}.json" or body["schema"] != "ascend-pair-phases-v1":
        raise ValueError("receipt hash/schema mismatch")
    if stage is not None and body["stage"] != stage:
        raise ValueError("unexpected artifact stage")
    actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob("*") if p.is_file()}
    if actual != set(body["files"]) | {path.name}:
        raise ValueError("artifact file inventory changed")
    for rel, info in body["files"].items():
        p = path.parent / rel
        if p.is_symlink() or not p.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("unsafe artifact path")
        if p.stat().st_size != info["bytes"] or sha_file(p) != info["sha256"]:
            raise ValueError(f"artifact hash mismatch: {rel}")
    return body


# Source fence for the selected route. No provider is imported or built here.
# Whole files are hashed, but dependency traversal follows the selected symbols,
# not every provider registered beside them. Unanticipated executed project code
# is rejected by execution_fence rather than silently accepted as a new dep.
class SourceClosure:
    def __init__(self, root):
        import ast
        self.ast = ast
        self.root = Path(root).resolve()
        self.bases = (self.root, self.root / 'src', self.root.parent, Path(__file__).resolve().parent)
        self.files, self.edges, self.seen, self.cache = {}, [], set(), {}

    def key(self, path):
        path = Path(path).resolve()
        if path.parent == Path(__file__).resolve().parent and path.name in (
                'ascend_pair_phases.py', 'ascend_phase_artifacts.py', 'ascend_pair_cpu.py'):
            return '@checker/' + path.name
        return str(path.relative_to(self.root.parent))

    def add_file(self, path, why):
        path = Path(path).resolve()
        key = self.key(path)
        self.files[key] = sha_file(path) if path.is_file() else None
        self.edges.append({'to': key, 'why': why})
        return key

    def module_path(self, module):
        parts = module.split('.')
        for base in self.bases:
            f, package = base.joinpath(*parts).with_suffix('.py'), base.joinpath(*parts, '__init__.py')
            if f.is_file(): return f
            if package.is_file(): return package
        return None

    def parsed(self, module):
        if module in self.cache: return self.cache[module]
        path = self.module_path(module)
        if path is None: return None
        tree = self.ast.parse(path.read_text(), filename=str(path))
        bindings = {}
        for n in tree.body:
            if isinstance(n, (self.ast.FunctionDef, self.ast.AsyncFunctionDef, self.ast.ClassDef)):
                bindings[n.name] = n
            elif isinstance(n, (self.ast.Assign, self.ast.AnnAssign)):
                targets = n.targets if isinstance(n, self.ast.Assign) else [n.target]
                for target in targets:
                    if isinstance(target, self.ast.Name): bindings[target.id] = n
            elif isinstance(n, (self.ast.Import, self.ast.ImportFrom)):
                for alias in n.names:
                    bindings[alias.asname or (alias.name.split('.')[0] if isinstance(n, self.ast.Import) else alias.name)] = (n, alias)
        result = path, tree, bindings
        self.cache[module] = result
        return result

    def import_target(self, module, node, alias):
        if isinstance(node, self.ast.Import):
            return alias.name, None
        target = node.module or ''
        if node.level:
            path = self.module_path(module)
            package = module if path.name == '__init__.py' else module.rpartition('.')[0]
            parts = package.split('.')
            target = '.'.join(parts[:len(parts) - node.level + 1] + ([target] if target else []))
        if self.module_path(target + '.' + alias.name):
            return target + '.' + alias.name, None
        return target, alias.name

    def follow(self, module, symbol=None, why='root'):
        identity = (module, symbol)
        if identity in self.seen: return
        self.seen.add(identity)
        parsed = self.parsed(module)
        if parsed is None: return  # torch/stdlib/runtime tracked independently
        path, tree, bindings = parsed
        self.add_file(path, why + ' -> ' + module + (':' + symbol if symbol else ''))
        parts = module.split('.')
        for i in range(1, len(parts)):
            parent = self.module_path('.'.join(parts[:i]))
            if parent and parent.name == '__init__.py':
                self.add_file(parent, 'package initializer for ' + module)
        if symbol is None:
            # Used for a directly imported local module (e.g. F). Its callable
            # dispatch cannot be resolved statically: follow every definition.
            for name in bindings:
                self.follow(module, name, 'module access ' + module)
            return
        node = bindings.get(symbol)
        if isinstance(node, tuple):
            self.follow(*self.import_target(module, *node), why=module + ':' + symbol)
            return
        if node is None:
            # Methods live in their class, and externally exported dynamic
            # symbols cannot be assumed safe. Runtime execution fence must still
            # find every project file in the precomputed set.
            return
        for n in self.ast.walk(node):
            if isinstance(n, self.ast.Name) and isinstance(n.ctx, self.ast.Load) and n.id in bindings:
                self.follow(module, n.id, module + ':' + symbol)
            elif isinstance(n, (self.ast.Import, self.ast.ImportFrom)):
                for alias in n.names:
                    self.follow(*self.import_target(module, n, alias), why=module + ':' + symbol + ' local import')

    def follow_imports(self, module):
        """Hash the actual import-time local closure, without following functions."""
        identity = (module, '@imports')
        if identity in self.seen: return
        self.seen.add(identity)
        parsed = self.parsed(module)
        if parsed is None: return
        path, tree, _ = parsed
        self.add_file(path, 'dynamic route import-time dependency')
        for node in tree.body:
            if isinstance(node, (self.ast.Import, self.ast.ImportFrom)):
                for alias in node.names:
                    target, _ = self.import_target(module, node, alias)
                    self.follow_imports(target)

    def resolve_symbol(self, module, symbol):
        seen = set()
        while (module, symbol) not in seen:
            seen.add((module, symbol))
            parsed = self.parsed(module)
            if not parsed: raise ValueError("selected factory is not local Python source")
            node = parsed[2].get(symbol)
            if isinstance(node, tuple):
                module, symbol = self.import_target(module, *node)
            elif isinstance(node, (self.ast.FunctionDef, self.ast.AsyncFunctionDef)):
                return {"module": module, "qualname": symbol,
                        "source": self.key(parsed[0])}
            elif isinstance(node, self.ast.Assign) and isinstance(node.value, self.ast.Name):
                symbol = node.value.id
            else:
                raise ValueError("dynamic selected factory requires original checker")
        raise ValueError("cyclic selected factory aliases")

    def assignment(self, module, name):
        parsed = self.parsed(module)
        if not parsed: raise ValueError('missing dispatch module: ' + module)
        node = parsed[2].get(name)
        if not isinstance(node, (self.ast.Assign, self.ast.AnnAssign)):
            raise ValueError('unreviewed dynamic dispatch declaration: ' + module + ':' + name)
        return node.value

    def dict_item(self, node, key):
        if not isinstance(node, self.ast.Dict):
            raise ValueError('unreviewed dynamic provider map; original checker required')
        for k, v in zip(node.keys, node.values):
            if k is None:
                raise ValueError('unreviewed expanded provider map; original checker required')
            if isinstance(k, self.ast.Constant) and k.value == key: return v
        return None



# Reviewed dynamic registration sources. A new program requires review, not exec.
_DYNAMIC_SOURCE_PINS = {'benchmarks/ascend_all_pairs/ops_math/__init__.py': '31db113fa68f24dd2767befbebccb096ebaf682a08004c72dd90ee980881b361',
 'benchmarks/ascend_all_pairs/ops_math/composition_specs.py': 'cfc59f6f98bbd59a899fd95ea7950854d5697622a7975e85747fc0599af648c7',
 'benchmarks/ascend_all_pairs/ops_math/semantic_limits.py': '2c304276b34a147ca35d676d18d6f3ea6e5707269db908ab49ecec9317cb0f51',
 'benchmarks/ascend_all_pairs/ops_math/specs.py': 'fc7aee2d97131136b4636e6ddf5e41687d76a9ff0185b1d5a3827b0b89614012',
 'benchmarks/ascend_all_pairs/ops_nn/__init__.py': ('2724e36f92bd0d288f1b96ab87a83d8ead1df9ca740ae4c89b895ba3cc07572a',
                                                    'c8e908c100d1a1be54ca4a12f97b76ca5a107a7d3cf386ca4e12be42cd2ec74a',
                                                    'cafe19fd73341e474524eac671fed52c77f11e70c58bad988f1c73b1c9f17c78'),
 'benchmarks/ascend_all_pairs/ops_nn/catalog.py': 'c1bcb5eb685308ae4fb03ba45e6e11b6a9d6bbff3b6ebcc66cf116a9ae5f8acf',
 'benchmarks/ascend_all_pairs/ops_nn/matrix.py': '3347572cf01123858b13d18b0dc0e21d890160ea83c705cb8761690f0c978950',
 'benchmarks/ascend_all_pairs/ops_nn/staging.py': ('dacaa8bbf62c631a5f2005b2970ca5f6d83cd622b8e66125ff72debc632f97d5',
                                                   '3e05c4cff6aa4a2d4f861b00a801e2be2f67eb1900cd04bfebe528a9b0e99012',
                                                   'd19081f918cb38774b1a5dc2ba83edd3662ca744cad5d756e4cf1119102e2123'),
 'benchmarks/ascend_all_pairs/ops_nn/conv2d_native.py': '4d5df64cd7bf5280430ea2b8751c1545a1130eed9c18de7be725800cb04bf7ac',
 'benchmarks/ascend_all_pairs/ops_nn/mean_axis_lift.py': '99fa4c6e234097b3a86e371a640a8075b40504922e03b8acf72614678e77017e',
 'benchmarks/ascend_all_pairs/registry.py': 'c45ef41fa96b819b19b0d077dd14c61d887cfded8885ff9d02db403c9f33ea25'}


def admission_files(package, continuation):
    if package == 'benchmarks.ascend_all_pairs.ops_math':
        return {'_enabled': ('enabled.json', 'list'), '_enabled_cases': ('enabled_cases.json', 'map')}
    if package != 'benchmarks.ascend_all_pairs.ops_nn':
        raise ValueError('unknown admission package')
    result = {'_accepted': ('accepted_cases.json', 'map'), '_pinned': ('accepted_pinned_cases.json', 'map')}
    if continuation:
        result['_continuation_data'] = ('accepted_continuation_cases.json', 'records')
    return result


def decode_admission(package, continuation, history):
    """Rebuild exact semantic identity from sealed original JSON bytes, cold-safe."""
    files = admission_files(package, continuation)
    if set(history) != set(files):
        raise ValueError('admission history inventory mismatch')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate provider JSON key')
            result[key] = value
        return result
    def strings(value):
        return type(value) is list and all(type(x) is str for x in value) and len(value) == len(set(value))
    data = {}
    for key, (name, kind) in files.items():
        item = history[key]
        wanted = 'TileOPs/' + package.replace('.', '/') + '/' + name
        if item['source'] != wanted:
            raise ValueError('admission source identity mismatch')
        raw = bytes.fromhex(item['raw_hex']) if item['raw_hex'] is not None else None
        if item['sha256'] != (hashlib.sha256(raw).hexdigest() if raw is not None else None):
            raise ValueError('admission raw bytes/hash mismatch')
        if raw is None:
            if package.endswith('.ops_math'): raise ValueError('missing dynamic admission data')
            value = {'records': []} if kind == 'records' else {}
        else:
            value = json.loads(raw, object_pairs_hook=unique)
        valid = strings(value) if kind == 'list' else (type(value) is dict and all(type(k) is str and strings(v) for k, v in value.items()))
        if kind == 'records':
            valid = (type(value) is dict and type(value.get('records')) is list and
                     all(type(r) is dict and type(r.get('op')) is str and type(r.get('case_id')) is str for r in value['records']))
            if valid and len({(r['op'], r['case_id']) for r in value['records']}) != len(value['records']):
                raise ValueError('duplicate continuation case identity')
        if not valid: raise ValueError('invalid dynamic admission data: ' + name)
        data[key] = value
    if continuation:
        data['_continuation'] = {}
        for record in data['_continuation_data']['records']:
            data['_continuation'].setdefault(record['op'], []).append(record['case_id'])
        data['_continuation'] = {k: sorted(v) for k, v in data['_continuation'].items()}
    return data


def selected_admission(package, op, case_id, data):
    """Only facts read by the reviewed selected resolver/factory branch.

    Record presence and all selected continuation fields are included, even
    though continuation execution remains held. Unselected rows cannot affect
    the pinned Add branch; file schema/duplicate failures still fail closed.
    """
    math = package.endswith('.ops_math')
    return {'package': package, 'op': op, 'case_id': case_id,
            'registered': op in data['_enabled' if math else '_accepted'],
            'accepted': case_id in data['_enabled_cases' if math else '_accepted'].get(op, []),
            'pinned': case_id in data.get('_pinned', {}).get(op, []),
            'continuation': [r for r in data.get('_continuation_data', {'records': []})['records']
                             if r['op'] == op and r['case_id'] == case_id]}


def read_admission(root, package, op, case_id, continuation):
    history = {}
    for key, (name, _) in admission_files(package, continuation).items():
        path = Path(root) / package.replace('.', '/') / name
        raw = path.read_bytes() if path.is_file() else None
        history[key] = {'source': str(path.relative_to(Path(root).parent)),
                        'sha256': hashlib.sha256(raw).hexdigest() if raw is not None else None,
                        'raw_hex': raw.hex() if raw is not None else None}
    data = decode_admission(package, continuation, history)
    selected = selected_admission(package, op, case_id, data)
    return {'version': 1, 'selected': selected, 'history': history}, data


def validate_admission(root, route):
    """Historical bytes prove the plan; fresh bytes prove the current selection."""
    old = route['admission']
    selected = old['selected']
    if old['version'] != 1 or selected['package'] != route['package'] or selected['op'] != route['op']:
        raise ValueError('selected admission plan identity mismatch')
    historical = decode_admission(route['package'], route['continuation_registry'], old['history'])
    if selected_admission(route['package'], route['op'], selected['case_id'], historical) != selected:
        raise ValueError('selected admission historical identity mismatch')
    current, data = read_admission(root, route['package'], route['op'], selected['case_id'], route['continuation_registry'])
    if current['selected'] != selected:
        raise ValueError('selected admission changed; hold current case')
    current['changed_unselected_sources'] = [item['source'] for key, item in current['history'].items()
                                            if item['sha256'] != old['history'][key]['sha256']]
    return current, data


def dynamic_provider(c, package, op, case_id=None):
    """Interpret only two reviewed comprehensions and their JSON admission data.

    No provider code is executed. Pins bind the surrounding assignments, loops,
    nested factory and late CANDIDATES updates, which a DictComp alone cannot prove.
    Selected plans bind admission semantics and retain full raw JSON history;
    callers without a case retain the legacy whole-file admission fence.
    """
    token = package.rsplit('.', 1)[-1]
    allowed = {'ops_math': {'AddFwdOp', 'AmaxFwdOp'},
               'ops_nn': {'AddFwdOp', 'AmaxFwdOp', 'AdaptiveMaxPool2dFwdOp',
                          'AdaptiveMaxPool2dIndicesFwdOp'}}
    if op not in allowed.get(token, set()):
        raise ValueError('unknown dynamic route; original checker required')
    paths = ['benchmarks/ascend_all_pairs/registry.py',
             'benchmarks/ascend_all_pairs/' + token + '/__init__.py']
    if token == 'ops_math':
        paths += ['benchmarks/ascend_all_pairs/ops_math/' + f for f in
                  ('specs.py', 'composition_specs.py', 'semantic_limits.py')]
    else:
        paths += ['benchmarks/ascend_all_pairs/ops_nn/' + f for f in
                  ('staging.py', 'catalog.py', 'matrix.py')]
    # This reviewed staging revision only adds an unselected Mean factory.
    # Its newly imported module must match the independently reviewed bytes.
    if token == 'ops_nn' and sha_file(c.root / 'benchmarks/ascend_all_pairs/ops_nn/staging.py') in ('3e05c4cff6aa4a2d4f861b00a801e2be2f67eb1900cd04bfebe528a9b0e99012', 'd19081f918cb38774b1a5dc2ba83edd3662ca744cad5d756e4cf1119102e2123'):
        paths.append('benchmarks/ascend_all_pairs/ops_nn/mean_axis_lift.py')
    if token == 'ops_nn' and sha_file(c.root / 'benchmarks/ascend_all_pairs/ops_nn/staging.py') == 'd19081f918cb38774b1a5dc2ba83edd3662ca744cad5d756e4cf1119102e2123':
        paths.append('benchmarks/ascend_all_pairs/ops_nn/conv2d_native.py')
    for rel in paths:
        path = c.root / rel
        accepted = _DYNAMIC_SOURCE_PINS[rel]
        accepted = (accepted,) if isinstance(accepted, str) else accepted
        if not path.is_file() or sha_file(path) not in accepted:
            raise ValueError('unreviewed dynamic source: ' + rel)
        c.add_file(path, 'reviewed dynamic dispatch program')
    continuation = (token == 'ops_nn' and sha_file(c.module_path(package)) in
                    ('c8e908c100d1a1be54ca4a12f97b76ca5a107a7d3cf386ca4e12be42cd2ec74a',
                     'cafe19fd73341e474524eac671fed52c77f11e70c58bad988f1c73b1c9f17c78'))
    expression = ("{op: partial(_public,op=op) for op in _enabled}" if token == 'ops_math'
                  else "{name:_provider(name,frozenset(ids),frozenset(_pinned.get(name,()))) for name,ids in _accepted.items()}")
    if continuation:
        expression = '{name:_provider(name,frozenset(ids),frozenset(_pinned.get(name,())),frozenset(_continuation.get(name,()))) for name,ids in _accepted.items()}'
    if c.ast.dump(c.assignment(package, 'PROVIDERS')) != c.ast.dump(c.ast.parse(expression, mode='eval').body):
        raise ValueError('unreviewed dynamic registration expression')

    admission, data = read_admission(c.root, package, op, case_id, continuation)
    for item in admission['history'].values():
        path = source_path(c.root, item['source'])
        if case_id is None:
            c.add_file(path, 'dynamic provider admission data')
        else:
            c.edges.append({'to': item['source'], 'why': 'selected-case semantic admission; raw bytes archived in plan'})
    registered = op in data['_enabled' if token == 'ops_math' else '_accepted']
    route = {'package': package, 'op': op, 'registered': registered,
             'data_sha256': {k: hashlib.sha256(canonical(v)).hexdigest() for k, v in data.items()},
             'accepted_ids': data['_enabled_cases' if token == 'ops_math' else '_accepted'].get(op, []),
             'pinned_ids': data.get('_pinned', {}).get(op, []),
             'continuation_ids': data.get('_continuation', {}).get(op, []),
             'continuation_registry': continuation, 'roots': []}
    if case_id is not None:
        route['admission'] = admission
    # Missing public entries follow the original registry's classic fallback.
    # Existing entries NEVER fall through because a case is refused.
    if not registered: return route
    if op != 'AddFwdOp':
        route['hold'] = 'registered non-Add dynamic route needs selected runtime/receipt dependency review'
    elif route['pinned_ids'] and (case_id is None or case_id in route['pinned_ids']):
        route['hold'] = 'pinned runtime external receipt closure requires independent review'
    else:
        if token == 'ops_math':
            route['roots'] = [('benchmarks.ascend_math_baselines', 'opsmath_add')]
            route['selected'] = {'module': package, 'qualname': '_public',
                                 'source': c.key(c.module_path(package))}
            # _screened(Add) reaches _add, semantic_reason and source_hashes;
            # it never reaches isolated.make or that child process's global map.
            route['roots'] += [(package, '_add'), (package + '.semantic_limits', 'reason'),
                               ('benchmarks.ascend_baselines', 'source_hashes')]
        else:
            route['roots'] = [(package + '.matrix', 'legacy'),
                               ('benchmarks.ascend_nn_direct', 'opsnn_inplace_add')]
            route['selected'] = {'module': package, 'qualname': '_provider.<locals>.factory',
                                 'source': c.key(c.module_path(package))}
            # legacy's import_module(LEGACY[op][0]) is intentionally explicit.
            legacy = c.ast.literal_eval(c.assignment(package + '.catalog', 'LEGACY'))
            if legacy.get(op) != ('benchmarks.ascend_nn_direct', 'opsnn_inplace_add', 'norm/inplace_add_rms_norm'):
                raise ValueError('unknown NN legacy route')
            # staging imports these modules even for a single selected factory.
            c.follow_imports(package + '.staging')
    if case_id is not None and op == 'AddFwdOp' and 'hold' not in route:
        directories = (['.tools/baselines/opp/ops-math/vendors/baseline_math_math',
                        'ascend_baselines/ops-math/math/add'] if token == 'ops_math' else
                       ['.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn',
                        'ascend_baselines/ops-nn/norm/inplace_add_rms_norm',
                        'ascend_baselines/ops-nn/norm/add_rms_norm'])
        route['native_directories'] = directories
        for name in directories:
            directory = c.root.parent / name
            if not directory.is_dir(): raise ValueError('missing original Add native directory: ' + name)
            for path in sorted(directory.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    c.add_file(path, 'original selected Add source/vendor complete inventory')
    if 'selected' not in route:
        route['selected'] = {'module': package, 'qualname': '_provider.<locals>.factory' if token == 'ops_nn' else '_public',
                             'source': c.key(c.module_path(package))}
    return route


def dependency_closure(root, op, library, case_id=None):
    """Selected symbol closure + actual registries/spec data + execution fence.

    Registry files themselves are dependencies (including absent selected
    package sentinels), but no recursive all-benchmarks or all-workloads scan.
    The manifest loader merges every YAML, so these data files remain fenced.
    """
    c = SourceClosure(root)
    roots = [('benchmarks.api', 'Case'), ('benchmarks.api', 'cases'),
             ('workloads.numerics', 'verify'), ('workloads.numerics', 'compare_outputs'),
             ('workloads.numerics', '_require_unchanged'), ('workloads.numerics', '_unique'),
             ('workloads.numerics', '_tensors'), ('workloads.device', 'set_run_device'),
             ('tools.baseline_protocol', 'seed_case'),
             # Merged case registries import benchmarks.baselines, which installs
             # this finder. It executes during cold Torch serialization imports.
             ('benchmarks.baselines', '_install_flaggems_import_order')]
    family = {'AddFwdOp': 'elementwise', 'AmaxFwdOp': 'reduction',
              'AdaptiveMaxPool2dFwdOp': 'pool', 'AdaptiveMaxPool2dIndicesFwdOp': 'pool'}[op]
    klass = {'elementwise': 'ElementwiseCall', 'reduction': 'ReductionCall', 'pool': 'AdaptiveMaxPool2dCall'}[family]
    roots += [('workloads.' + family, klass)]
    # Actual merged registry definitions are data dependencies. Traversing every
    # non-selected workload they register would recreate the original bug.
    for p in sorted((c.root / 'benchmarks/_cases').glob('*.py')):
        c.add_file(p, 'merged case registry declaration')
    for p in sorted((c.root / 'src/tileops/manifest/spec').glob('*.yaml')):
        c.add_file(p, 'manifest loader reads merged specs')
    registry = 'benchmarks.ascend_all_pairs.registry'
    c.add_file(c.module_path(registry), 'original candidate_factory dispatch')
    token = c.ast.literal_eval(c.assignment(registry, 'LIBRARY_MODULES'))[library]
    package = 'benchmarks.ascend_all_pairs.' + token
    package_file = c.root / ('benchmarks/ascend_all_pairs/' + token + '/__init__.py')
    c.add_file(package_file, 'selected provider package existence sentinel')
    selected, dynamic = None, None
    if package_file.is_file():
        declaration = c.assignment(package, 'PROVIDERS')
        if isinstance(declaration, c.ast.DictComp):
            dynamic = dynamic_provider(c, package, op, case_id)
            if dynamic['registered']:
                selected = dynamic['selected']
                roots += dynamic['roots']
        else:
            selected = c.dict_item(declaration, op)
        if selected is not None and dynamic is None:
            if not isinstance(selected, c.ast.Name): raise ValueError('unreviewed provider expression')
            roots.append((package, selected.id))
    if selected is None:
        classic = 'benchmarks.ops.bench_ascend_classics'
        c.add_file(c.module_path(classic), 'original fallback provider dispatch')
        by_op = c.dict_item(c.assignment(classic, 'PROVIDERS'), op)
        if by_op is None:
            raise ValueError('no registered provider or original classic fallback for ' + op + '/' + library)
        selected = c.dict_item(by_op, library)
        if not isinstance(selected, c.ast.Name): raise ValueError('unreviewed fallback provider expression')
        roots.append((classic, selected.id))
    selected_factory = dynamic['selected'] if dynamic and dynamic['registered'] else c.resolve_symbol(*roots[-1])
    if dynamic:
        selected_factory = dict(selected_factory, dynamic_route=dynamic)
    for module, symbol in roots:
        c.follow(module, symbol)
    for path in (c.root / 'scripts/ascend_pair_check.py',
                 *(Path(__file__).with_name(name) for name in
                   ('ascend_pair_phases.py', 'ascend_phase_artifacts.py', 'ascend_pair_cpu.py'))):
        c.add_file(path, 'checker/codec/bootstrap whole source')
    return {'version': 2, 'route': [op, library], 'roots': roots,
            'selected_factory': selected_factory,
            'sha256': dict(sorted(c.files.items())), 'edges': c.edges,
            'scope': 'selected static symbol closure; whole-file hashes; unexpected executed project files fail closed',
            'runtime_packages': 'matching Torch build required; external runtime/native evidence audited separately'}


def source_path(root, key):
    if key.startswith('@checker/'):
        name = key.removeprefix('@checker/')
        if name not in ('ascend_pair_phases.py', 'ascend_phase_artifacts.py', 'ascend_pair_cpu.py'):
            raise ValueError('unknown checker source key')
        return Path(__file__).with_name(name)
    path = (Path(root).parent / key).resolve()
    if not path.is_relative_to(Path(root).parent.resolve()):
        raise ValueError('source escaped workspace: ' + key)
    return path


from contextlib import contextmanager


@contextmanager
def execution_fence(root, expected, observed, *, enabled=True):
    """Record actual project Python call files; reject any missing static dep.

    This is Python call provenance, NOT driver-level native launch evidence.
    Profiling is scoped to generation/build/run/cleanup or CPU reference/verify.
    Candidate resolution imports are covered by registry/file fences separately.
    """
    import sys
    root = Path(root).resolve()
    checker = Path(__file__).resolve().parent
    cache, checked_codes, source_codes = {}, set(), {}
    prior = sys.getprofile()
    if enabled and prior is not None:
        raise ValueError('existing Python profiler; original checker required')
    def profile(frame, event, arg):
        if event != 'call': return
        name = frame.f_code.co_filename
        if frame.f_code in checked_codes: return
        if name in cache and cache[name] is None: return
        path = Path(name)
        # Most calls are Torch/stdlib; no filesystem resolution per repeated call.
        if not path.is_absolute():
            cache[name] = None
            return
        path = path.resolve()
        if path.parent == checker and path.name in ('ascend_pair_phases.py', 'ascend_phase_artifacts.py', 'ascend_pair_cpu.py'):
            key = '@checker/' + path.name
        elif path.is_relative_to(root) or path.is_relative_to(root.parent / 'tools'):
            key = str(path.relative_to(root.parent))
        else:
            cache[name] = None
            return
        digest = sha_file(path)
        if expected.get(key) != digest:
            raise ValueError('executed dependency outside source fence or changed: ' + key)
        if key not in source_codes:
            from types import CodeType
            def codes(node):
                yield node
                for value in node.co_consts:
                    if isinstance(value, CodeType): yield from codes(value)
            source_codes[key] = set(codes(compile(path.read_bytes(), str(path), 'exec')))
        if frame.f_code not in source_codes[key]:
            raise ValueError('executed project code object differs from source: ' + key)
        checked_codes.add(frame.f_code)
        observed[key] = digest
        cache[name] = key
    if enabled: sys.setprofile(profile)
    try:
        yield
    finally:
        if enabled: sys.setprofile(prior)
