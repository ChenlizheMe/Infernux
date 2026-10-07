#pragma once

#include <function/resources/ShaderAsset/ShaderInfoSchema.h>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace shader_parser_tests
{
struct Case
{
    const char *body;
    bool valid;
    const char *resource;
    const char *constant;
    const char *property;
};

inline constexpr Case Cases[] = {
    {"Resources { Texture2D keep; }", true, "keep", "", ""},
    {"Resources { Texture3D keep; }", true, "keep", "", ""},
    {"Resources { Texture2DUInt keep; }", true, "keep", "", ""},
    {"Resources { Texture2DMS keep; }", true, "keep", "", ""},
    {"Resources { Texture2DMSUInt keep; }", true, "keep", "", ""},
    {"Resources { BufferUInt keep; }", true, "keep", "", ""},
    {"Resources { Float bad; Texture2D keep; }", false, "keep", "", ""},
    {"Resources { Float bad Texture3D keep; }", false, "keep", "", ""},
    {"Resources { Bogus bad BufferUInt keep; }", false, "keep", "", ""},
    {"Resources { Texture2D ; Texture3D keep; }", false, "keep", "", ""},
    {"Resources { Texture2D keep;", false, "keep", "", ""},
    {"PushConstants pc { Float keep; }", true, "", "keep", ""},
    {"PushConstants pc { Float3 normal; Mat4 keep; }", true, "", "keep", ""},
    {"PushConstants pc { Texture2D bad; Float keep; }", false, "", "keep", ""},
    {"PushConstants pc { FloatArray bad Float3 keep; }", false, "", "keep", ""},
    {"PushConstants pc { Bogus bad Float keep; }", false, "", "keep", ""},
    {"PushConstants pc { Float ; Float3 keep; }", false, "", "keep", ""},
    {"PushConstants pc { Float keep;", false, "", "keep", ""},
    {"Properties { Float bad Float keep = 2; }", false, "", "", "keep"},
    {"Properties { Float ; Float keep = 2; }", false, "", "", "keep"},
    {"Properties { Float keep = 2;", false, "", "", "keep"},
    {"Resources { }", true, "", "", ""},
    {"PushConstants pc { }", false, "", "", ""},
};

inline void CheckCase(size_t index)
{
    if (index >= std::size(Cases))
        throw std::invalid_argument("unknown parser test case");
    const auto &test = Cases[index];
    const std::string source = std::string("ShaderInfo { Name \"Boundary\" ") + test.body + " }";
    const auto document = infernux::ParseShaderInfo(source);
    if (document.IsValid() != test.valid)
        throw std::runtime_error("parser validity disagrees with the authored declaration");
    if (!test.valid) {
        bool hasError = false;
        for (const auto &diagnostic : document.diagnostics) {
            hasError |= diagnostic.severity == infernux::ShaderInfoDiagnosticSeverity::Error;
            if (diagnostic.message.empty() || diagnostic.location.offset > source.size())
                throw std::runtime_error("malformed parser diagnostic");
        }
        if (!hasError || document.diagnostics.size() > 8)
            throw std::runtime_error("invalid source needs bounded explicit diagnostics");
    }
    const auto contains = [](const auto &fields, const char *name) {
        for (const auto &field : fields)
            if (field.name == name)
                return true;
        return false;
    };
    if (*test.resource && !contains(document.resources, test.resource))
        throw std::runtime_error("resource recovery skipped the next legal declaration");
    if (*test.constant && (!document.pushConstants || !contains(document.pushConstants->fields, test.constant)))
        throw std::runtime_error("constant recovery skipped the next legal declaration");
    if (*test.property && !contains(document.properties, test.property))
        throw std::runtime_error("property recovery skipped the next legal declaration");
    std::cout << "PARSER_CASE_OK " << index << '\n';
}
} // namespace shader_parser_tests
