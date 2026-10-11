#include <core/reflection/SemanticTypeRegistry.h>

#include <atomic>
#include <cassert>
#include <stdexcept>
#include <thread>

using namespace infernux;

template <typename Function> void Reject(Function operation)
{
    bool failed = false;
    try {
        operation();
    } catch (const std::exception &) {
        failed = true;
    }
    assert(failed);
}

SemanticTypeDescriptor Type(std::string guid, std::string owner, std::string origin = "native")
{
    SemanticTypeDescriptor type;
    type.typeGuid = guid;
    type.readableId = "test." + guid;
    type.owner = std::move(owner);
    type.origin = std::move(origin);
    type.fields.push_back(
        {"Probe.value", "FieldType.FLOAT", false, {{"field_id", "value"}, {"default", 2.0}, {"range", {0.0, 5.0}}}});
    return type;
}

int main()
{
    SemanticTypeRegistry registry;
    const auto empty = registry.Read();
    auto native = Type("native-guid", "engine");
    auto python = Type("python-guid", "package:probe", "python");
    python.baseTypeGuid = native.typeGuid;
    auto prepared = registry.Prepare({{"engine", {native}}, {"package:probe", {python}}});
    assert(registry.Read() == empty && empty->revision == 0);
    native.fields[0].attributes["default"] = 99;
    assert(prepared.Candidate()->types.at("native-guid")->fields[0].attributes.at("default") == 2.0);
    assert(registry.Publish(prepared) == 1);
    const auto first = registry.Read();
    assert(first->types.size() == 2);
    assert(first->types.at("python-guid")->revision == 1);
    Reject([&] { registry.Publish(prepared); });
    SemanticTypeRegistry other;
    Reject([&] { other.Publish(prepared); });

    auto stolen = Type("native-guid", "package:probe", "python");
    Reject([&] { (void)registry.Prepare({{"package:probe", {stolen}}}); });
    Reject([&] { (void)registry.Prepare({{"engine", {}}, {"package:probe", {stolen}}}); });
    auto alias = Type("different-guid", "other");
    alias.readableId = "test.native-guid";
    Reject([&] { (void)registry.Prepare({{"other", {alias}}}); });
    auto invalid = Type("invalid", "other");
    invalid.fields.push_back(invalid.fields.front());
    Reject([&] { (void)registry.Prepare({{"other", {invalid}}}); });
    Reject([&] { (void)registry.Prepare({{"engine", {}}}); }); // Live derived type needs its base.
    auto cyclic = Type("native-guid", "engine");
    cyclic.baseTypeGuid = python.typeGuid;
    Reject([&] { (void)registry.Prepare({{"engine", {cyclic}}}); });
    auto unresolved = python;
    unresolved.baseTypeGuid = "missing-guid";
    Reject([&] { (void)registry.Prepare({{"package:probe", {unresolved}}}); });
    auto mismatchedOwner = python;
    mismatchedOwner.owner = "other";
    Reject([&] { (void)registry.Prepare({{"package:probe", {mismatchedOwner}}}); });
    assert(registry.Read() == first);

    auto stale = registry.Prepare({{"package:probe", {}}});
    auto update = python;
    update.fields[0].attributes["default"] = 3.0;
    registry.Publish(registry.Prepare({{"package:probe", {update}}}));
    Reject([&] { registry.Publish(stale); });
    assert(first->revision == 1 && first->types.at("python-guid")->fields[0].attributes.at("default") == 2.0);

    std::atomic<bool> stop{false};
    std::atomic<unsigned> reads{0};
    std::thread reader([&] {
        uint64_t previous = 0;
        while (!stop.load()) {
            const auto snapshot = registry.Read();
            assert(snapshot->revision >= previous);
            previous = snapshot->revision;
            assert(snapshot->types.size() == 2);
            assert(snapshot->readableIds.at("test.python-guid") == "python-guid");
            assert(snapshot->types.at("python-guid")->revision == snapshot->revision);
            ++reads;
        }
    });
    while (reads.load() == 0)
        std::this_thread::yield();
    for (unsigned i = 0; i < 100; ++i)
        registry.Publish(registry.Prepare({{"package:probe", {update}}}));
    stop = true;
    reader.join();
    assert(reads > 0);
    registry.Publish(registry.Prepare({{"package:probe", {}}, {"engine", {}}}));
    assert(registry.Read()->types.empty());
    assert(registry.Read()->revision == 103);
    assert(first->types.size() == 2); // Retained snapshots own their descriptors.
}
