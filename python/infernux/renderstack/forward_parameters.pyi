"""Typed Default Forward parameters shared by native render backends."""
from enum import IntEnum

class MSAASamples(IntEnum):
    OFF = 1
    X2 = 2
    X4 = 4
    X8 = 8

class DefaultForwardParameters:
    name: str
    shadow_resolution: int
    msaa_samples: MSAASamples
