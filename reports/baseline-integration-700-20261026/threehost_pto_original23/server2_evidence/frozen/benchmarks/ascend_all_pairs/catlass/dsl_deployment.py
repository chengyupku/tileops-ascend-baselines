"""Validate a same-source CPU stage before guarded device use.

Registration uses only metadata and hashes. Native imports and launches remain
inside the separately authorized npu-guard runtime harness.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sysconfig


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _child(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise RuntimeError('Invalid path in CATLASS deployment manifest')
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()):
        raise RuntimeError('CATLASS deployment manifest escapes its root')
    return result


@lru_cache(maxsize=1)
def resolve_dsl_package(vendor):
    """Return package parent directory or None. Reject mismatched explicit stage."""
    vendor = Path(vendor).resolve()
    setting = os.environ.get('CATLASS_DSL_STAGE')
    if not setting:
        dsl = vendor / 'python/tla_dsl'
        if (dsl / 'catlass/_mlir').is_dir() and any((dsl / 'catlass/_mlir').rglob('*.so')):
            return dsl
        return None
    stage = Path(setting).expanduser().resolve()
    if not stage.is_relative_to(vendor.parent.parent):
        raise RuntimeError('CATLASS stage must reside within this workspace for receipt hashing')
    manifest = stage / 'deployment_manifest.json'
    info = json.loads(manifest.read_text())
    if info.get('schema') != 'catlass-dsl-stage/v1':
        raise RuntimeError('Unsupported CATLASS stage manifest schema')
    if info.get('status') != 'cpu_deployment_ready_unverified':
        raise RuntimeError('CATLASS stage has not passed CPU deployment checks')
    if info.get('machine') != platform.machine():
        raise RuntimeError('CATLASS stage CPU architecture does not match this host')
    if info.get('python_soabi') != sysconfig.get_config_var('SOABI'):
        raise RuntimeError('CATLASS stage Python ABI does not match this interpreter')
    sources = info.get('vendor_source_sha256', {})
    artifacts = info.get('stage_sha256', {})
    if not sources or not artifacts:
        raise RuntimeError('CATLASS stage has no source/artifact provenance')
    for rel, expected in sources.items():
        if _sha(_child(vendor, rel)) != expected:
            raise RuntimeError('CATLASS vendor source differs from stage: ' + rel)
    for rel, expected in artifacts.items():
        if _sha(_child(stage, rel)) != expected:
            raise RuntimeError('CATLASS stage artifact changed: ' + rel)
    if not any(rel.startswith('catlass/_mlir/') and '.so' in rel for rel in artifacts):
        raise RuntimeError('CATLASS stage has no MLIR native bindings')
    if not any(rel.startswith('catlass/_tla_type_bridge_native.') and rel.endswith('.so') for rel in artifacts):
        raise RuntimeError('CATLASS stage has no typed compiler bridge')
    return stage


def dsl_artifact_paths(vendor):
    package = resolve_dsl_package(vendor)
    if package is None:
        return []
    paths = [Path(__file__)] + [p for p in (package / 'catlass').rglob('*') if p.is_file()]
    manifest = package / 'deployment_manifest.json'
    if manifest.exists():
        paths.append(manifest)
    return paths


def dsl_cache_dir(default, vendor):
    """Keep BC cache separate: an unpackaged source tree reports version 'unknown'."""
    package = resolve_dsl_package(vendor)
    manifest = package / 'deployment_manifest.json' if package else None
    if manifest is not None and manifest.exists():
        return Path(default) / ('stage-' + _sha(manifest)[:16])
    return Path(default)
