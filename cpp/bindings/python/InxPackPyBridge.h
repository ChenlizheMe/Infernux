#pragma once
#include "BufferPyView.h"
#include <platform/filesystem/InxPack.h>
#include <platform/filesystem/InxPath.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <utility>

namespace infernux::binding
{
namespace py = pybind11;

inline py::dict InxPackManifestToPython(const inxpack::Manifest &manifest)
{
    py::dict result;
    result["format"] = "infernux-native-inxpack";
    result["codec"] = "zstd-or-store";
    result["file_count"] = manifest.entries.size();
    result["raw_bytes"] = manifest.rawBytes;
    result["stored_bytes"] = manifest.storedBytes;
    result["payload_bytes"] = manifest.payloadBytes;
    result["archive_bytes"] = manifest.archiveBytes;
    const auto checksum = inxpack::HashToHex(manifest.archiveHash);
    if (manifest.textHeader)
        result["archive_xxh64"] = checksum.substr(0, 16);
    else
        result["archive_sha256"] = checksum;
    result["text_header"] = manifest.textHeader ? py::object(py::bytes(*manifest.textHeader)) : py::object(py::none());
    py::list files;
    for (const auto &entry : manifest.entries) {
        py::dict item;
        item["path"] = entry.path;
        item["offset"] = entry.offset;
        item["stored_bytes"] = entry.storedBytes;
        item["raw_bytes"] = entry.rawBytes;
        item["codec"] = inxpack::CodecName(entry.codec);
        files.append(std::move(item));
    }
    result["files"] = std::move(files);
    return result;
}

inline std::vector<inxpack::Source> InxPackSourcesFromPython(py::handle value, std::vector<py::buffer_info> &pins)
{
    if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value))
        throw std::invalid_argument("InxPack sources require (logical_path, file_path or contiguous buffer) pairs");
    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    std::vector<inxpack::Source> sources;
    sources.reserve(sequence.size());
    for (const py::handle item : sequence) {
        if (!py::isinstance<py::sequence>(item) || py::isinstance<py::str>(item))
            throw std::invalid_argument("InxPack source entry must be a two-value sequence");
        const py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
        if (pair.size() != 2)
            throw std::invalid_argument("InxPack source entry requires a logical path and file path or memory buffer");
        const auto path = py::cast<std::string>(pair[0]);
        if (py::isinstance<py::str>(pair[1])) {
            sources.push_back({path, ToFsPath(py::cast<std::string>(pair[1]))});
        } else {
            auto info = bindings::ContiguousBuffer(pair[1]);
            sources.push_back({path, inxpack::MemorySource{static_cast<const uint8_t *>(info.ptr),
                                                           static_cast<size_t>(info.size * info.itemsize)}});
            pins.push_back(std::move(info));
        }
    }
    return sources;
}

inline std::vector<std::string> InxPackAllowedRootsFromPython(py::handle value)
{
    if (value.is_none())
        return {};
    if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value))
        throw std::invalid_argument("InxPack allowed_roots must be a sequence or None");
    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    std::vector<std::string> roots;
    roots.reserve(sequence.size());
    for (const py::handle item : sequence)
        roots.push_back(py::cast<std::string>(item));
    return roots;
}

inline inxpack::WriteOptions InxPackWriteOptionsFromPython(py::handle compressionLevel, const std::string &profile)
{
    inxpack::WriteOptions options;
    if (profile == "development")
        options.profile = inxpack::CompressionProfile::Development;
    else if (profile == "release")
        options.profile = inxpack::CompressionProfile::Release;
    else
        throw std::invalid_argument("InxPack compression profile must be 'development' or 'release'");

    if (!compressionLevel.is_none()) {
        options.compressionLevel = py::cast<int>(compressionLevel);
        inxpack::ValidateCompressionLevel(options.compressionLevel);
    }
    return options;
}

inline void BindPackFunctions(py::module_ &m)
{
    m.def("_inxpack_header_reservation", &inxpack::TextHeaderReservationBytes, py::arg("header_bytes"));
    m.def(
        "_inxpack_write",
        [](py::handle sources, const std::string &destination, py::handle compressionLevel, const std::string &profile,
           std::string textHeader) {
            std::vector<py::buffer_info> pins;
            auto sourceFiles = InxPackSourcesFromPython(sources, pins);
            const auto destinationPath = ToFsPath(destination);
            auto options = InxPackWriteOptionsFromPython(compressionLevel, profile);
            if (!textHeader.empty())
                options.textHeader = std::make_shared<const std::string>(std::move(textHeader));
            inxpack::Manifest manifest;
            {
                // The caller pins borrowed buffers until this synchronous IO
                // finishes; file/hash/compression work does not hold the GIL.
                py::gil_scoped_release release;
                manifest = inxpack::Write(destinationPath, std::move(sourceFiles), options);
            }
            return InxPackManifestToPython(manifest);
        },
        py::arg("sources"), py::arg("destination"), py::arg("compression_level") = py::none(),
        py::arg("profile") = "development", py::arg("text_header") = "",
        "Write native InxPack from file paths or borrowed contiguous buffers, optionally with an INM text header.");
    m.def(
        "_inxpack_read_manifest",
        [](const std::string &path) {
            inxpack::Manifest manifest;
            {
                py::gil_scoped_release release;
                manifest = inxpack::ReadManifest(ToFsPath(path));
            }
            return InxPackManifestToPython(manifest);
        },
        py::arg("path"), "Read the native InxPack header and directory.");
    m.def(
        "_inxpack_extract",
        [](const std::string &path, const std::string &destination, py::handle allowedRoots) {
            const auto roots = InxPackAllowedRootsFromPython(allowedRoots);
            inxpack::Manifest manifest;
            {
                py::gil_scoped_release release;
                manifest = inxpack::Extract(ToFsPath(path), ToFsPath(destination), roots);
            }
            return InxPackManifestToPython(manifest);
        },
        py::arg("path"), py::arg("destination"), py::arg("allowed_roots") = py::none(),
        "Validate and extract a native InxPack with an optional allowed root filter.");
    m.def(
        "_inxpack_read_entry",
        [](const std::string &path, const std::string &entryPath) {
            std::vector<uint8_t> bytes;
            {
                py::gil_scoped_release release;
                bytes = inxpack::ReadEntry(ToFsPath(path), entryPath);
            }
            return py::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        },
        py::arg("path"), py::arg("entry_path"), "Read and decode one native InxPack entry.");
}
} // namespace infernux::binding
