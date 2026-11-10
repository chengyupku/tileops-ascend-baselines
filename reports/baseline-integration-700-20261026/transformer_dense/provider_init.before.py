"""Isolated ops-transformer providers; importing this module is device-free."""
from .adapters import bmm

PROVIDERS = {'BmmFwdOp': bmm}
