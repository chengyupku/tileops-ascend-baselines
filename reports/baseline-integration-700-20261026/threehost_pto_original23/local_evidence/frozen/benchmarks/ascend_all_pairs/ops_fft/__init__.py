"""Native ops-fft on AArch64; imports and plan creation are lazy."""

from .provider import fft

PROVIDERS = {"FFTC2CFwdOp": fft}
