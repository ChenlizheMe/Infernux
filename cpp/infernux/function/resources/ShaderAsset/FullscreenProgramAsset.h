#pragma once

#include <function/renderer/FullscreenRenderer.h>

namespace infernux
{
/// Resolve one author asset by GUID, source path or shader ID and retain its
/// selected target with the immutable asset publication that owns it.
[[nodiscard]] std::shared_ptr<const FullscreenProgram> LoadFullscreenProgram(const std::string &name,
                                                                             ShaderCodeTarget targetKind);
} // namespace infernux
