#include "BindingRegistration.h"

#include <core/text/TextSegments.h>
#include <function/renderer/gui/InxTextGeometry.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infernux
{
namespace
{
py::array CharacterView(const std::shared_ptr<textlayout::TextCharacterBuffer> &buffer, size_t offset,
                        std::vector<py::ssize_t> shape)
{
    shape.insert(shape.begin(), static_cast<py::ssize_t>(buffer->GetCount()));
    std::vector<py::ssize_t> strides(shape.size(), sizeof(float));
    for (size_t index = shape.size() - 1; index > 1; --index)
        strides[index - 1] = strides[index] * shape[index];
    strides[0] = sizeof(textlayout::TextCharacterBuffer::Row);
    float *data = buffer->GetData();
    return py::array(py::dtype::of<float>(), shape, strides, data ? data + offset : nullptr, py::cast(buffer));
}
} // namespace

void RegisterTextBindings(py::module_ &module)
{
    using namespace textlayout;
    py::class_<TextLine>(module, "TextLineInfo")
        .def_readonly("source_start", &TextLine::startOffset)
        .def_readonly("source_end", &TextLine::endOffset)
        .def_readonly("width", &TextLine::width)
        .def_readonly("y", &TextLine::y)
        .def_readonly("height", &TextLine::height)
        .def_readonly("glyph_start", &TextLine::glyphStart)
        .def_readonly("glyph_end", &TextLine::glyphEnd);
    py::class_<TextGlyph>(module, "TextGlyphInfo")
        .def_readonly("source_start", &TextGlyph::startOffset)
        .def_readonly("source_end", &TextGlyph::endOffset)
        .def_readonly("codepoint", &TextGlyph::codepoint)
        .def_readonly("character", &TextGlyph::character)
        .def_property_readonly("line",
                               [](const TextGlyph &glyph) -> py::object {
                                   return glyph.line == text::NoWord ? py::none() : py::cast(glyph.line);
                               })
        .def_readonly("advance", &TextGlyph::advance)
        .def_readonly("visible", &TextGlyph::visible)
        .def_readonly("font_size", &TextGlyph::logicalFontSize)
        .def_readonly("color", &TextGlyph::color)
        .def_readonly("bounds", &TextGlyph::bounds);
    py::class_<text::StyleRun>(module, "TextStyleSpan")
        .def_readonly("source_start", &text::StyleRun::start)
        .def_readonly("source_end", &text::StyleRun::end)
        .def_readonly("font_size", &text::StyleRun::fontSize)
        .def_readonly("color", &text::StyleRun::color);
    py::class_<TextCharacter>(module, "TextCharacterInfo")
        .def_readonly("source_start", &TextCharacter::startOffset)
        .def_readonly("source_end", &TextCharacter::endOffset)
        .def_readonly("glyph_start", &TextCharacter::glyphStart)
        .def_readonly("glyph_end", &TextCharacter::glyphEnd)
        .def_readonly("line", &TextCharacter::line)
        .def_property_readonly("word",
                               [](const TextCharacter &character) -> py::object {
                                   return character.word == text::NoWord ? py::none() : py::cast(character.word);
                               })
        .def_readonly("whitespace", &TextCharacter::whitespace)
        .def_readonly("bounds", &TextCharacter::bounds);
    py::class_<text::Word>(module, "TextWordInfo")
        .def_readonly("character_start", &text::Word::characterStart)
        .def_readonly("character_end", &text::Word::characterEnd);
    py::class_<TextLayoutResult, std::shared_ptr<TextLayoutResult>>(
        module, "TextLayout",
        "Immutable UI text layout. Source offsets are UTF-8 bytes; characters are extended graphemes.")
        .def(py::init([](const std::string &text, float fontSize, float wrapWidth, const std::string &fontPath,
                         float lineHeight, float letterSpacing, const std::vector<std::string> &fallbackFonts,
                         bool richText) {
                 return CreateTextLayout(
                     {text, fontPath, fontSize, wrapWidth, lineHeight, letterSpacing, fallbackFonts, richText});
             }),
             py::arg("text"), py::arg("font_size") = 18.0f, py::arg("wrap_width") = 0.0f, py::arg("font_path") = "",
             py::arg("line_height") = 1.2f, py::arg("letter_spacing") = 0.0f,
             py::arg("fallback_font_paths") = std::vector<std::string>(), py::arg("rich_text") = false)
        .def_static(
            "split_graphemes",
            [](const std::string &source) {
                const auto segments = text::SegmentText(source);
                std::vector<std::string> characters;
                characters.reserve(segments.characters.size());
                for (const auto &character : segments.characters)
                    characters.emplace_back(source.substr(character.start, character.end - character.start));
                return characters;
            },
            py::arg("text"), "Split original text into extended graphemes without a renderer or font.")
        .def_readonly("text", &TextLayoutResult::text)
        .def_readonly("display_text", &TextLayoutResult::displayText)
        .def_readonly("width", &TextLayoutResult::totalWidth)
        .def_readonly("height", &TextLayoutResult::totalHeight)
        .def_readonly("font_size", &TextLayoutResult::logicalFontSize)
        .def_readonly("line_advance", &TextLayoutResult::lineAdvance)
        .def_property_readonly("lines", [](const TextLayoutResult &layout) { return layout.lines; })
        .def_property_readonly("glyphs", [](const TextLayoutResult &layout) { return layout.glyphs; })
        .def_property_readonly("characters", [](const TextLayoutResult &layout) { return layout.characters; })
        .def_property_readonly("styles", [](const TextLayoutResult &layout) { return layout.styles; })
        .def_property_readonly("words", [](const TextLayoutResult &layout) { return layout.segments.words; });

    py::enum_<TextGeometryLayer>(module, "TextGeometryLayer")
        .value("Manual", TextGeometryLayer::Manual)
        .value("Effects", TextGeometryLayer::Effects)
        .value("Appearance", TextGeometryLayer::Appearance);
    py::class_<TextCharacterBuffer, std::shared_ptr<TextCharacterBuffer>>(
        module, "TextCharacterBuffer", py::buffer_protocol(),
        "Writable float32 character views. Colors and vertices use TL, TR, BR, BL order. Commit with "
        "UIText.set_character_data().")
        .def_readonly_static("columns", &TextCharacterBuffer::Columns)
        .def_property_readonly("character_count", &TextCharacterBuffer::GetCount)
        .def("reset", &TextCharacterBuffer::Reset)
        .def_buffer([](TextCharacterBuffer &buffer) {
            return py::buffer_info(
                buffer.GetData(), sizeof(float), py::format_descriptor<float>::format(), 2,
                {static_cast<py::ssize_t>(buffer.GetCount()), static_cast<py::ssize_t>(TextCharacterBuffer::Columns)},
                {static_cast<py::ssize_t>(sizeof(TextCharacterBuffer::Row)), static_cast<py::ssize_t>(sizeof(float))});
        })
        .def_property_readonly(
            "offsets", [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 0, {2}); })
        .def_property_readonly(
            "rotations",
            [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 2, {}); })
        .def_property_readonly(
            "scales", [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 3, {2}); })
        .def_property_readonly(
            "skews", [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 5, {}); })
        .def_property_readonly(
            "vertex_offsets",
            [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 6, {4, 2}); })
        .def_property_readonly(
            "colors",
            [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 14, {4, 4}); })
        .def_property_readonly(
            "affine_transforms",
            [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 30, {3, 4}); })
        .def_property_readonly(
            "vertex_depth_offsets",
            [](const std::shared_ptr<TextCharacterBuffer> &buffer) { return CharacterView(buffer, 42, {4}); })
        .def_property_readonly("color_overrides", [](const std::shared_ptr<TextCharacterBuffer> &buffer) {
            return CharacterView(buffer, 46, {});
        });
    py::class_<TextGeometry, std::shared_ptr<TextGeometry>>(module, "TextGeometry")
        .def(py::init([](const std::string &text, float fontSize, float wrapWidth, const std::string &fontPath,
                         float lineHeight, float letterSpacing, const std::vector<std::string> &fallbackFonts,
                         bool richText) {
                 return std::make_shared<TextGeometry>(TextLayoutParams{text, fontPath, fontSize, wrapWidth, lineHeight,
                                                                        letterSpacing, fallbackFonts, richText});
             }),
             py::arg("text"), py::arg("font_size") = 18.0f, py::arg("wrap_width") = 0.0f, py::arg("font_path") = "",
             py::arg("line_height") = 1.2f, py::arg("letter_spacing") = 0.0f,
             py::arg("fallback_font_paths") = std::vector<std::string>(), py::arg("rich_text") = false)
        .def(
            "set_layout",
            [](TextGeometry &geometry, const std::string &text, float fontSize, float wrapWidth,
               const std::string &fontPath, float lineHeight, float letterSpacing,
               const std::vector<std::string> &fallbackFonts, bool richText) {
                geometry.SetLayout(
                    {text, fontPath, fontSize, wrapWidth, lineHeight, letterSpacing, fallbackFonts, richText});
            },
            py::arg("text"), py::arg("font_size") = 18.0f, py::arg("wrap_width") = 0.0f, py::arg("font_path") = "",
            py::arg("line_height") = 1.2f, py::arg("letter_spacing") = 0.0f,
            py::arg("fallback_font_paths") = std::vector<std::string>(), py::arg("rich_text") = false)
        .def("prepare", &TextGeometry::Prepare, py::arg("scale") = 1.0f)
        .def_property_readonly("layout", &TextGeometry::GetLayout)
        .def_property_readonly("revision", &TextGeometry::GetRevision)
        .def_property_readonly("layout_revision", &TextGeometry::GetLayoutRevision)
        .def_property("visible_characters", &TextGeometry::GetVisibleCharacters, &TextGeometry::SetVisibleCharacters)
        .def("create_character_buffer", &TextGeometry::CreateCharacterBuffer)
        .def("set_character_data", &TextGeometry::SetCharacterData, py::arg("buffer"),
             py::arg("layer") = TextGeometryLayer::Manual)
        .def("reset_character_data", &TextGeometry::ResetCharacterData, py::arg("layer") = -1);
}

} // namespace infernux
