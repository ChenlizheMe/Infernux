"""Single runtime-visible Infernux release identity."""

ENGINE_VERSION = "0.4.1"
ENGINE_BUILD_NUMBER = 3
# Project pins and compiled Player payloads identify a release, including its
# wheel revision. Package dependency specifiers continue to use ENGINE_VERSION.
ENGINE_RELEASE = (
    ENGINE_VERSION if ENGINE_BUILD_NUMBER == 1
    else f"{ENGINE_VERSION}-v{ENGINE_BUILD_NUMBER}"
)

__all__ = ["ENGINE_VERSION", "ENGINE_BUILD_NUMBER", "ENGINE_RELEASE"]
