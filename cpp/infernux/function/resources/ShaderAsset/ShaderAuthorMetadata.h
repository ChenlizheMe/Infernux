#pragma once

#include "SlangCompiler.h"
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetSource.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <nlohmann/json.hpp>

namespace infernux
{
inline void CreateShaderAuthorMetadata(AssetSource &source, InxResourceMeta &metadata)
{
    metadata.InitFromSource(source, ResourceType::Shader);
    const auto *kind = asset_formats::FindShaderSourceKind(source.Path());
    const auto prepared = SlangCompiler::Get().PrepareAuthorSource(
        {"author_metadata", source.Path(), std::string(source.Bytes())}, {}, ShaderAuthorPhase::Metadata);
    const auto info = nlohmann::json::parse(prepared.schema);
    metadata.AddMetadata("type", std::string(kind->stage));
    metadata.AddMetadata("file_extension", std::string(kind->suffix));
    metadata.AddMetadata("shader_source_kind", std::string(kind->stage));
    metadata.AddMetadata("shader_domain", info.at("domain").get<std::string>());
    metadata.AddMetadata("shader_id", info.at("shader_id").get<std::string>());
    metadata.AddMetadata("shader_hidden", info.at("hidden").get<bool>());
    metadata.AddMetadata("shader_surface_type", info.at("surface_type").get<std::string>());
    metadata.AddMetadata("shader_cull_mode", info.at("cull_mode").get<std::string>());
    metadata.AddMetadata("shader_depth_write", info.at("depth_write").is_null()     ? std::string{}
                                               : info.at("depth_write").get<bool>() ? std::string("on")
                                                                                    : std::string("off"));
    metadata.AddMetadata("shader_depth_test", info.at("depth_test").get<std::string>());
    metadata.AddMetadata("shader_blend", info.at("blend").get<std::string>());
    metadata.AddMetadata("shader_queue", info.at("render_queue").get<int>());
    metadata.AddMetadata("shader_pass_tag", info.at("pass_tag").get<std::string>());
    metadata.AddMetadata("shader_stencil", info.at("stencil").get<std::string>());
    metadata.AddMetadata("shader_alpha_test",
                         info.at("alpha_clip").is_null() ? std::string{} : info.at("alpha_clip").dump());
    metadata.AddMetadata("shader_cast_shadows", info.at("cast_shadows").get<bool>());
    metadata.AddMetadata("shader_receive_shadows", info.at("receive_shadows").get<bool>());
}
} // namespace infernux
