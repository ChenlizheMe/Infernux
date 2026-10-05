#include <function/scene/SceneAuthoringIdentity.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
using Json = nlohmann::json;
using namespace infernux;

void Require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

std::string Guid(char prefix, char suffix)
{
    return std::string(1, prefix) + std::string(30, '0') + suffix;
}

SceneAuthoringIdentity Identities()
{
    return {
        {{19, Guid('a', '1')}, {26, Guid('a', '2')}, {91, Guid('a', '3')}},
        {{38, Guid('b', '1')}, {45, Guid('b', '2')}, {47, Guid('b', '3')}, {52, Guid('b', '4')}, {99, Guid('b', '5')}}};
}

Json RuntimeDocument()
{
    auto component = [](uint64_t id, std::string type, Json data) {
        return Json{{"component_id", id},
                    {"type_id", std::move(type)},
                    {"enabled", true},
                    {"execution_order", 0},
                    {"data", std::move(data)}};
    };
    Json first = {
        {"name", "First"},
        {"id", uint64_t{19}},
        {"transform", {{"component_id", uint64_t{38}}}},
        {"children", Json::array()},
        {"prefab_source_id", 42},
        {"prefab_source", {{"root_object", {{"local_id", 19}, {"component_id", 47}}}}},
        {"components",
         Json::array(
             {component(47, "native:infernux.HingeJoint", {{"connected_body_component_id", uint64_t{45}}}),
              component(52, "python:script:type:module:Component",
                        {{"nested", Json::array({Json{{"$type", "game_object_ref"}, {"object_id", uint64_t{91}}},
                                                 Json{{"$type", "component_ref"},
                                                      {"game_object_id", uint64_t{26}},
                                                      {"component_id", uint64_t{99}},
                                                      {"component_type", "Light"}}})},
                         {"user_dict", {{"component_id", 777}, {"object_id", 888}}},
                         {"empty", {{"$type", "game_object_ref"}, {"object_id", uint64_t{0}}}}})})}};
    Json second = {{"name", "Second"},
                   {"id", uint64_t{26}},
                   {"transform", {{"component_id", uint64_t{45}}}},
                   {"children", Json::array()},
                   {"components", Json::array()}};
    return {{"name", "Authored"},
            {"isPlaying", false},
            {"objects", Json::array({first, second})},
            {"nextObjectId", uint64_t{120}},
            {"nextComponentId", uint64_t{140}},
            {"mainCameraComponentId", uint64_t{52}}};
}

template <class Callback> void MustReject(Callback operation, const char *message)
{
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, message);
}

void TestRoundTripAndMissingReferences()
{
    const auto original = RuntimeDocument();
    const auto encoded = EncodeSceneAuthoringDocument(original, Identities());
    Require(!encoded.contains("nextObjectId") && !encoded.contains("nextComponentId"), "file contains watermarks");
    Require(encoded["objects"][0]["id"] == Guid('a', '1'), "object GUID was not encoded");
    const auto &fields = encoded["objects"][0]["components"][1]["data"];
    Require(fields["nested"][0]["object_id"] == Guid('a', '3'), "missing object identity was discarded");
    Require(fields["nested"][1]["component_id"] == Guid('b', '5'), "missing component identity was discarded");
    Require(fields["empty"]["object_id"] == "", "null object reference was not encoded");
    Require(fields["user_dict"] == original["objects"][0]["components"][1]["data"]["user_dict"],
            "untyped user data was rewritten");
    Require(encoded["objects"][0]["prefab_source"] == original["objects"][0]["prefab_source"],
            "other asset's prefab namespace was rewritten");
    SceneAuthoringIdentity decodedIds;
    const auto decoded = DecodeSceneAuthoringDocument(encoded, decodedIds);
    Require(decoded["objects"][0]["id"] == 1 && decoded["objects"][1]["id"] == 2,
            "runtime IDs were not assigned by declaration order");
    Require(decoded["objects"][0]["components"][0]["data"]["connected_body_component_id"] ==
                decoded["objects"][1]["transform"]["component_id"],
            "forward native reference changed target");
    Require(decodedIds.objects.size() == 3 && decodedIds.components.size() == 5,
            "missing reference tables were not preserved");
    Require(EncodeSceneAuthoringDocument(decoded, decodedIds) == encoded, "authoring round trip changed identity");
    SceneAuthoringIdentity repeated;
    Require(DecodeSceneAuthoringDocument(encoded, repeated) == decoded, "same file decoded differently");
    Require(original == RuntimeDocument(), "encoder modified caller's runtime document");
}

void TestRejectInvalidIdentityBeforePublishing()
{
    const auto document = RuntimeDocument();
    const auto good = EncodeSceneAuthoringDocument(document, Identities());
    for (const std::string &value :
         std::array<std::string, 4>{"", "invalid", std::string(32, '0'), std::string(32, 'A')}) {
        auto invalid = good;
        invalid["objects"][0]["id"] = value;
        auto output = Identities();
        MustReject([&] { (void)DecodeSceneAuthoringDocument(invalid, output); }, "invalid GUID accepted");
        Require(output.objects == Identities().objects && output.components == Identities().components,
                "failed decode partially replaced owner identity tables");
    }
    auto duplicate = good;
    duplicate["objects"][1]["id"] = duplicate["objects"][0]["id"];
    SceneAuthoringIdentity output;
    MustReject([&] { (void)DecodeSceneAuthoringDocument(duplicate, output); }, "duplicate object GUID accepted");
    duplicate = good;
    duplicate["objects"][1]["transform"]["component_id"] = duplicate["objects"][0]["components"][0]["component_id"];
    MustReject([&] { (void)DecodeSceneAuthoringDocument(duplicate, output); }, "duplicate component GUID accepted");
    MustReject([&] { (void)DecodeSceneAuthoringDocument(document, output); }, "numeric legacy asset accepted");
    auto watermark = good;
    watermark["nextObjectId"] = 100;
    MustReject([&] { (void)DecodeSceneAuthoringDocument(watermark, output); }, "file watermark accepted");
    auto missing = Identities();
    missing.objects.erase(91);
    MustReject([&] { (void)EncodeSceneAuthoringDocument(document, missing); }, "missing identity was invented");
    auto alias = Identities();
    alias.components[99] = alias.components[45];
    MustReject([&] { (void)EncodeSceneAuthoringDocument(document, alias); }, "identity alias accepted");
}

void TestIndependentAuthorsDoNotPersistRuntimeCollisions()
{
    const auto snapshot = RuntimeDocument();
    auto a = Identities();
    auto b = Identities();
    for (auto &[id, guid] : b.objects)
        guid[0] = 'c';
    for (auto &[id, guid] : b.components)
        guid[0] = 'd';
    auto combined = EncodeSceneAuthoringDocument(snapshot, a);
    auto other = EncodeSceneAuthoringDocument(snapshot, b);
    for (const auto &object : other["objects"])
        combined["objects"].push_back(object);
    SceneAuthoringIdentity identities;
    const auto decoded = DecodeSceneAuthoringDocument(combined, identities);
    Require(decoded["objects"].size() == 4, "independent author graph was lost");
    Require(identities.objects.size() == 6 && identities.components.size() == 10,
            "independent authors' same runtime numbers collided");
    Require(EncodeSceneAuthoringDocument(decoded, identities) == combined, "combined references changed");
}
} // namespace

int main()
{
    try {
        TestRoundTripAndMissingReferences();
        TestRejectInvalidIdentityBeforePublishing();
        TestIndependentAuthorsDoNotPersistRuntimeCollisions();
        std::cout << "Scene authoring identity codec contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
