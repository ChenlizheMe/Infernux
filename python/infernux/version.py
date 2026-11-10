"""Single runtime-visible Infernux release identity."""

ENGINE_VERSION = "0.4.8"
ENGINE_BUILD_NUMBER = 1
# Project pins and compiled Player payloads identify a release, including its
# wheel revision. PEP 440 post releases let plugin requirements include hotfixes.
ENGINE_RELEASE = (
    ENGINE_VERSION if ENGINE_BUILD_NUMBER == 1
    else f"{ENGINE_VERSION}-v{ENGINE_BUILD_NUMBER}"
)
ENGINE_COMPATIBILITY_VERSION = (
    ENGINE_VERSION if ENGINE_BUILD_NUMBER == 1
    else f"{ENGINE_VERSION}.post{ENGINE_BUILD_NUMBER}"
)

__all__ = ["ENGINE_VERSION", "ENGINE_BUILD_NUMBER", "ENGINE_RELEASE", "ENGINE_COMPATIBILITY_VERSION"]
