#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelImporter.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <platform/filesystem/InxPath.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
using namespace infernux;
using Json = nlohmann::json;

void Check(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}

glm::mat4 Bind(glm::vec3 scale, bool rotated)
{
    const auto rotation = rotated ? glm::angleAxis(0.7f, glm::normalize(glm::vec3(1, 2, 3)))
                                  : glm::quat(1, 0, 0, 0);
    return glm::translate(glm::mat4(1), glm::vec3(2, -3, 4)) * glm::mat4_cast(rotation) *
           glm::scale(glm::mat4(1), scale);
}

InxSkinnedMesh MakeModel(glm::vec3 scale, bool rotated)
{
    InxSkinnedMesh model;
    model.guid = "signed-bind";
    model.scaleFactor = 1;
    SkinnedRuntimeNode node;
    node.name = "Joint";
    node.bindLocal = node.bindGlobal = Bind(scale, rotated);
    model.skeleton.nodes.push_back(node);
    model.skeleton.nodeByName.emplace("Joint", 0);
    SkinnedRuntimeBone bone;
    bone.name = "Joint";
    bone.nodeIndex = 0;
    bone.inverseBind = glm::inverse(node.bindGlobal);
    model.skeleton.bones.push_back(bone);
    model.skeleton.boneByName.emplace("Joint", 0);
    for (auto position : {glm::vec3(1, 0, 0), glm::vec3(2, 0, 0), glm::vec3(1, 1, 0)}) {
        Vertex vertex{};
        vertex.pos = position;
        vertex.normal = {0, 0, 1};
        vertex.tangent = {1, 0, 0, 1};
        model.baseVertices.push_back(vertex);
        SkinInfluence influence;
        influence.weight[0] = 1;
        model.influences.push_back(influence);
    }
    model.indices = {0, 1, 2};
    SubMesh part;
    part.vertexCount = part.indexCount = 3;
    part.boundsMin = {1, 0, 0};
    part.boundsMax = {2, 1, 0};
    model.subMeshes.push_back(part);
    SkinnedRuntimeAnimation take;
    take.name = take.id = "Move";
    take.durationTicks = take.ticksPerSecond = 1;
    take.trackByNodeIndex = {0};
    SkinnedRuntimeTrack track;
    track.nodeIndex = 0;
    track.positions = {{0, {2, -3, 4}}, {1, {3, -3, 4}}};
    take.tracks.push_back(track);
    model.animations.push_back(take);
    model.NormalizeInfluences();
    return model;
}

std::shared_ptr<InxSkinnedMesh> ImportGltf(const std::filesystem::path &folder, glm::vec3 scale, bool rotated)
{
    Json document = {{"asset", {{"version", "2.0"}}}, {"scene", 0},
                     {"scenes", Json::array({{{"nodes", {0}}}})}};
    Json views = Json::array(), accessors = Json::array();
    std::vector<unsigned char> bytes;
    const auto append = [&](const auto &values, int component, const char *type, int count) {
        while (bytes.size() % 4)
            bytes.push_back(0);
        const auto offset = bytes.size();
        const auto *begin = reinterpret_cast<const unsigned char *>(values.data());
        bytes.insert(bytes.end(), begin, begin + values.size() * sizeof(values[0]));
        views.push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bytes.size() - offset}});
        accessors.push_back({{"bufferView", views.size() - 1}, {"componentType", component}, {"type", type}, {"count", count}});
        return accessors.size() - 1;
    };
    const auto positions = append(std::vector<float>{1, 0, 0, 2, 0, 0, 1, 1, 0}, 5126, "VEC3", 3);
    accessors[positions]["min"] = {1, 0, 0};
    accessors[positions]["max"] = {2, 1, 0};
    const auto indices = append(std::vector<uint16_t>{0, 1, 2}, 5123, "SCALAR", 3);
    const auto joints = append(std::vector<uint16_t>(12, 0), 5123, "VEC4", 3);
    const auto weights = append(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, 5126, "VEC4", 3);
    std::vector<float> inverse;
    const auto bind = Bind(scale, rotated);
    const auto inverseBind = glm::inverse(bind);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row) {
            inverse.push_back(inverseBind[column][row]);
        }
    const auto inverseIndex = append(inverse, 5126, "MAT4", 1);
    const auto times = append(std::vector<float>{0, 1}, 5126, "SCALAR", 2);
    accessors[times]["min"] = {0};
    accessors[times]["max"] = {1};
    const auto translations = append(std::vector<float>{2, -3, 4, 3, -3, 4}, 5126, "VEC3", 2);
    document["buffers"] = Json::array({{{"uri", "skin.bin"}, {"byteLength", bytes.size()}}});
    document["bufferViews"] = views;
    document["accessors"] = accessors;
    const auto rotation = rotated ? glm::angleAxis(0.7f, glm::normalize(glm::vec3(1, 2, 3)))
                                  : glm::quat(1, 0, 0, 0);
    document["nodes"] = Json::array({{{"name", "Scene"}, {"children", {1, 2}}},
        {{"name", "Joint"}, {"translation", {2, -3, 4}}, {"scale", {scale.x, scale.y, scale.z}},
         {"rotation", {rotation.x, rotation.y, rotation.z, rotation.w}}},
        {{"name", "Mesh"}, {"mesh", 0}, {"skin", 0}}});
    document["skins"] = Json::array({{{"joints", {1}}, {"skeleton", 1}, {"inverseBindMatrices", inverseIndex}}});
    document["meshes"] = Json::array({{{"primitives", Json::array({
        {{"attributes", {{"POSITION", positions}, {"JOINTS_0", joints}, {"WEIGHTS_0", weights}}}, {"indices", indices}}})}}});
    document["animations"] = Json::array({{{"name", "Move"},
        {"samplers", Json::array({{{"input", times}, {"output", translations}, {"interpolation", "LINEAR"}}})},
        {"channels", Json::array({{{"sampler", 0}, {"target", {{"node", 1}, {"path", "translation"}}}}})}}});
    {
        std::ofstream binary(folder / "skin.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        Check(binary.good(), "gltf buffer write failed");
        std::ofstream source(folder / "skin.gltf");
        source << document;
        Check(source.good(), "gltf source write failed");
    }
    return SkinnedModelImporter::ImportSource("signed-gltf", FromFsPath(folder / "skin.gltf"), 1);
}

InxSkinnedMesh WithParent(InxSkinnedMesh model)
{
    SkinnedRuntimeNode parent;
    parent.name = "Parent";
    parent.bindLocal = parent.bindGlobal = Bind({2, 1, 3}, true);
    model.skeleton.nodes.insert(model.skeleton.nodes.begin(), parent);
    model.skeleton.nodes[1].parent = 0;
    model.skeleton.nodes[1].bindGlobal = parent.bindGlobal * model.skeleton.nodes[1].bindLocal;
    model.skeleton.nodeByName = {{"Parent", 0}, {"Joint", 1}};
    model.skeleton.bones[0].nodeIndex = 1;
    model.skeleton.bones[0].inverseBind = glm::inverse(model.skeleton.nodes[1].bindGlobal);
    model.animations[0].tracks[0].nodeIndex = 1;
    model.animations[0].trackByNodeIndex = {-1, 0};
    return model;
}

void CheckPose(InxSkinnedMesh &model, const char *mode, bool checkExternal = true)
{
    Check(model.IsAssetPayloadValid() && model.skeleton.IsValid(), "invalid fixture model");
    SkinnedSampleRequest request;
    glm::vec3 delta(0);
    if (std::string_view(mode) == "animated") {
        request.takeName = "Move";
        request.timeSeconds = 0.5f;
        delta.x = 0.5f;
        const auto &joint = model.skeleton.nodes[model.skeleton.nodeByName.at("Joint")];
        if (joint.parent >= 0)
            delta = glm::mat3(model.skeleton.nodes[joint.parent].bindGlobal) * delta;
    } else if (std::string_view(mode) == "root_motion") {
        MeshImportSettings settings;
        settings.animationApplyRootMotion = true;
        settings.animationReferencePose = "bind_pose";
        SkinnedModelImporter::ApplyAnimationSettings(model, settings);
        request.takeName = "Move";
        request.timeSeconds = 0.5f;
        const auto motion = model.SampleRootMotionDelta("Move", 0, 0.5f, false);
        Check(std::abs(glm::length(motion.translation) - 0.5f) < 2.e-5f &&
              std::abs(std::abs(motion.rotation.w) - 1.0f) < 2.e-5f, "root motion delta changed");
    }
    const auto palette = std::string_view(mode) == "pose_stack"
        ? model.BuildGpuBonePaletteFromPoseStack({}) : model.BuildGpuBonePalette(request);
    const auto expected = glm::translate(glm::mat4(1), delta);
    Check(!palette.empty(), "missing bone palette");
    for (const auto &bone : palette)
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                Check(std::abs(bone[column][row] - expected[column][row]) < 2.e-5f, "bone palette changed bind geometry");
    if (checkExternal && !request.takeName.empty()) {
        const auto source = model;
        const auto external = model.BuildGpuBonePalette(request, &source);
        Check(external.size() == palette.size(), "external animation palette size changed");
        for (const auto &bone : external)
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    Check(std::abs(bone[column][row] - expected[column][row]) < 2.e-5f,
                          "external animation lost reflected scale");
    }
    glm::vec3 minimum(std::numeric_limits<float>::max()), maximum(std::numeric_limits<float>::lowest());
    const auto vertices = model.SampleVertices(request);
    Check(vertices.size() == model.baseVertices.size(), "sample stream length changed");
    for (size_t index = 0; index < vertices.size(); ++index) {
        const auto expectedPosition = model.baseVertices[index].pos + delta;
        Check(glm::length(vertices[index].pos - expectedPosition) < 2.e-5f, "sampled bind vertex moved");
        minimum = glm::min(minimum, expectedPosition);
        maximum = glm::max(maximum, expectedPosition);
    }
    glm::vec3 actualMin, actualMax;
    Check(model.ComputeSkinnedBounds(palette, actualMin, actualMax), "missing skinned bounds");
    Check(glm::length(minimum - actualMin) < 2.e-5f && glm::length(maximum - actualMax) < 2.e-5f,
          "skinned bounds disagree with geometry");
}
} // namespace

int main()
{
    const auto folder = std::filesystem::temp_directory_path() /
        ("infernux-bind-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(folder);
    int passed = 0, failed = 0;
    for (int signs = 0; signs < 8; ++signs)
        for (const bool rotated : {false, true})
            for (const bool imported : {false, true})
                for (const bool cooked : {false, true})
                    for (const char *mode : {"bind", "pose_stack", "animated", "root_motion"}) {
                        const std::string label = std::to_string(signs) + "/" + std::to_string(rotated) + "/" +
                            std::to_string(imported) + "/" + std::to_string(cooked) + "/" + mode;
                        try {
                            const glm::vec3 scale(signs & 1 ? -1 : 1, signs & 2 ? -2 : 2, signs & 4 ? -3 : 3);
                            auto model = imported ? *ImportGltf(folder, scale, rotated) : MakeModel(scale, rotated);
                            if (cooked)
                                model = *SkinnedMeshArtifact::Deserialize(SkinnedMeshArtifact::Serialize(model, "source"), "source");
                            CheckPose(model, mode);
                            ++passed;
                        } catch (const std::exception &error) {
                            std::cerr << "FAIL " << label << ": " << error.what() << '\n';
                            ++failed;
                        }
                    }
    std::filesystem::remove_all(folder);
    for (const bool reflected : {false, true})
        for (const bool cooked : {false, true})
            for (const char *mode : {"bind", "pose_stack", "animated"}) {
                const std::string label = "hierarchy/" + std::to_string(reflected) + "/" + std::to_string(cooked) + "/" + mode;
                try {
                    auto model = WithParent(MakeModel({reflected ? -1 : 1, 2, 3}, true));
                    if (cooked)
                        model = *SkinnedMeshArtifact::Deserialize(SkinnedMeshArtifact::Serialize(model, "source"), "source");
                    CheckPose(model, mode, false);
                    ++passed;
                } catch (const std::exception &error) {
                    std::cerr << "FAIL " << label << ": " << error.what() << '\n';
                    ++failed;
                }
            }
    std::cout << "SKIN_BIND_RESULT passed=" << passed << " failed=" << failed << '\n';
    return failed ? 1 : 0;
}
