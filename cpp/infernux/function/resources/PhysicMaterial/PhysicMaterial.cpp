#include "PhysicMaterial.h"
#include <platform/filesystem/AssetDocument.h>

#include <atomic>
#include <cmath>
#include <platform/filesystem/DocumentStore.h>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{

namespace
{
std::atomic<uint64_t> surfaceRevision{0};
}

uint64_t PhysicMaterial::GetSurfaceRevision() noexcept
{
    return surfaceRevision.load(std::memory_order_relaxed);
}

void PhysicMaterial::NotifySurfaceChanged() const noexcept
{
    // Staged decoding and unused asset edits cannot affect a collision scene.
    if (m_colliderBindings != 0)
        surfaceRevision.fetch_add(1, std::memory_order_relaxed);
}

void PhysicMaterial::SetFriction(float value)
{
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
        throw std::invalid_argument("PhysicMaterial friction must be finite and in [0, 1]");
    if (m_friction != value) {
        m_friction = value;
        NotifySurfaceChanged();
    }
}

void PhysicMaterial::SetBounciness(float value)
{
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
        throw std::invalid_argument("PhysicMaterial bounciness must be finite and in [0, 1]");
    if (m_bounciness != value) {
        m_bounciness = value;
        NotifySurfaceChanged();
    }
}

void PhysicMaterial::ValidateCombine(PhysicsMaterialCombine value, const char *fieldName)
{
    const int raw = static_cast<int>(value);
    if (raw < static_cast<int>(PhysicsMaterialCombine::Average) ||
        raw > static_cast<int>(PhysicsMaterialCombine::Maximum)) {
        throw std::invalid_argument(std::string("PhysicMaterial ") + fieldName + " is invalid");
    }
}

void PhysicMaterial::SetFrictionCombine(PhysicsMaterialCombine value)
{
    ValidateCombine(value, "friction_combine");
    if (m_frictionCombine != value) {
        m_frictionCombine = value;
        NotifySurfaceChanged();
    }
}

void PhysicMaterial::SetBounceCombine(PhysicsMaterialCombine value)
{
    ValidateCombine(value, "bounce_combine");
    if (m_bounceCombine != value) {
        m_bounceCombine = value;
        NotifySurfaceChanged();
    }
}

nlohmann::json PhysicMaterial::SerializeDocument() const
{
    return {{"friction", m_friction},
            {"bounciness", m_bounciness},
            {"friction_combine", static_cast<int>(m_frictionCombine)},
            {"bounce_combine", static_cast<int>(m_bounceCombine)}};
}

void PhysicMaterial::DeserializeDocument(const nlohmann::json &document)
{
    static const std::unordered_set<std::string> fields = {"friction", "bounciness", "friction_combine",
                                                           "bounce_combine"};
    if (!document.is_object() || document.size() != fields.size())
        throw std::invalid_argument("PhysicMaterial document must contain exactly the current schema fields");
    for (const auto &[key, value] : document.items()) {
        (void)value;
        if (fields.find(key) == fields.end())
            throw std::invalid_argument("unknown PhysicMaterial field: " + key);
    }
    if (!document.at("friction").is_number() || !document.at("bounciness").is_number() ||
        !document.at("friction_combine").is_number_integer() || !document.at("bounce_combine").is_number_integer())
        throw std::invalid_argument("PhysicMaterial fields have invalid types");

    // Check the original JSON integer (including uint64) before get<int>() or
    // the byte-sized enum can discard its high bits.
    for (const char *field : {"friction_combine", "bounce_combine"}) {
        const auto &value = document.at(field);
        if (value < 0 || value > static_cast<int>(PhysicsMaterialCombine::Maximum))
            throw std::invalid_argument(std::string("PhysicMaterial ") + field + " is invalid");
    }
    PhysicMaterial staged;
    staged.SetFriction(document.at("friction").get<float>());
    staged.SetBounciness(document.at("bounciness").get<float>());
    staged.SetFrictionCombine(static_cast<PhysicsMaterialCombine>(document.at("friction_combine").get<int>()));
    staged.SetBounceCombine(static_cast<PhysicsMaterialCombine>(document.at("bounce_combine").get<int>()));
    CopyValuesFrom(staged);
}

void PhysicMaterial::CopyValuesFrom(const PhysicMaterial &other)
{
    if (m_friction == other.m_friction && m_bounciness == other.m_bounciness &&
        m_frictionCombine == other.m_frictionCombine && m_bounceCombine == other.m_bounceCombine)
        return;
    m_friction = other.m_friction;
    m_bounciness = other.m_bounciness;
    m_frictionCombine = other.m_frictionCombine;
    m_bounceCombine = other.m_bounceCombine;
    NotifySurfaceChanged();
}

void PhysicMaterial::SaveToFile() const
{
    if (IsCookedAssetDocument(m_filePath))
        throw std::logic_error("Cooked physics material is read-only");
    if (m_filePath.empty())
        throw std::logic_error("PhysicMaterial has no file path");
    DocumentStore::Instance().WriteAndWait(m_filePath, SerializeDocument().dump(2));
}

void PhysicMaterial::SaveToFile(const std::string &path)
{
    if (path.empty())
        throw std::invalid_argument("PhysicMaterial save path cannot be empty");
    m_filePath = path;
    SaveToFile();
}

} // namespace infernux
