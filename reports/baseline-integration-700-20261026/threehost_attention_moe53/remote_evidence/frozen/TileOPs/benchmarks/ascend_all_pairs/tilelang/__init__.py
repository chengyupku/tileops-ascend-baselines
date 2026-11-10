"""Isolated TileLang providers. Importing this module does not import device runtimes."""
from .adapters import bmm

PROVIDERS = {"BmmFwdOp": bmm}
