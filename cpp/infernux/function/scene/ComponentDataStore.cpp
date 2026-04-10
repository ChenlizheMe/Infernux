#include "ComponentDataStore.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infernux
{

// ── singleton ────────────────────────────────────────────────────────────

ComponentDataStore &ComponentDataStore::Instance()
{
    static ComponentDataStore instance;
    return instance;
}

// ── helpers ──────────────────────────────────────────────────────────────

size_t ComponentDataStore::ElementSize(DataType type)
{
    switch (type) {
    case DataType::Float64:
        return sizeof(double);
    case DataType::Int64:
        return sizeof(int64_t);
    case DataType::Bool:
        return sizeof(uint8_t);
    case DataType::Vec2:
        return sizeof(float) * 2;
    case DataType::Vec3:
        return sizeof(float) * 3;
    case DataType::Vec4:
        return sizeof(float) * 4;
    }
    return 0;
}

void ComponentDataStore::FieldStorage::Grow(size_t newCapacity)
{
    data.resize(newCapacity * elementSize, 0);
}

void ComponentDataStore::FieldStorage::ResetSlot(size_t slot)
{
    std::memset(data.data() + slot * elementSize, 0, elementSize);
}

void ComponentDataStore::ClassStorage::GrowTo(size_t newCapacity)
{
    if (newCapacity <= capacity)
        return;
    for (auto &f : fields) {
        f.Grow(newCapacity);
    }
    alive.resize(newCapacity, 0);
    nextFree.resize(newCapacity, UINT32_MAX);
    capacity = newCapacity;
}

// ── registration ─────────────────────────────────────────────────────────

uint32_t ComponentDataStore::RegisterClass(const std::string &className)
{
    auto it = m_classNameToId.find(className);
    if (it != m_classNameToId.end())
        return it->second;

    uint32_t id = static_cast<uint32_t>(m_classes.size());
    m_classes.emplace_back();
    m_classNameToId[className] = id;
    return id;
}

uint32_t ComponentDataStore::RegisterField(uint32_t classId, const std::string &fieldName, DataType type)
{
    auto &cls = m_classes.at(classId);
    auto it = cls.fieldNameToId.find(fieldName);
    if (it != cls.fieldNameToId.end())
        return it->second;

    uint32_t fid = static_cast<uint32_t>(cls.fields.size());
    FieldStorage fs;
    fs.type = type;
    fs.elementSize = ElementSize(type);
    fs.Grow(cls.capacity); // match existing capacity
    cls.fields.push_back(std::move(fs));
    cls.fieldNameToId[fieldName] = fid;
    return fid;
}

uint32_t ComponentDataStore::GetClassId(const std::string &className) const
{
    auto it = m_classNameToId.find(className);
    if (it == m_classNameToId.end())
        return UINT32_MAX;
    return it->second;
}

uint32_t ComponentDataStore::GetFieldId(uint32_t classId, const std::string &fieldName) const
{
    if (classId >= m_classes.size())
        return UINT32_MAX;
    const auto &cls = m_classes[classId];
    auto it = cls.fieldNameToId.find(fieldName);
    if (it == cls.fieldNameToId.end())
        return UINT32_MAX;
    return it->second;
}

// ── slot lifecycle ───────────────────────────────────────────────────────

uint32_t ComponentDataStore::AllocateSlot(uint32_t classId)
{
    auto &cls = m_classes.at(classId);
    uint32_t slot;

    if (cls.freeHead != UINT32_MAX) {
        slot = cls.freeHead;
        cls.freeHead = cls.nextFree[slot];
    } else {
        slot = static_cast<uint32_t>(cls.capacity);
        cls.GrowTo(cls.capacity + 1);
    }

    cls.alive[slot] = 1;
    // Reset all field values for this slot.
    for (auto &f : cls.fields) {
        f.ResetSlot(slot);
    }
    ++cls.aliveCount;
    return slot;
}

void ComponentDataStore::ReleaseSlot(uint32_t classId, uint32_t slot)
{
    auto &cls = m_classes.at(classId);
    if (slot >= cls.capacity || !cls.alive[slot])
        return;
    cls.alive[slot] = 0;
    cls.nextFree[slot] = cls.freeHead;
    cls.freeHead = slot;
    --cls.aliveCount;
}

// ── per-element scalar ───────────────────────────────────────────────────

double ComponentDataStore::GetFloat(uint32_t classId, uint32_t fieldId, uint32_t slot) const
{
    return m_classes[classId].fields[fieldId].At<double>(slot);
}

void ComponentDataStore::SetFloat(uint32_t classId, uint32_t fieldId, uint32_t slot, double value)
{
    m_classes[classId].fields[fieldId].At<double>(slot) = value;
}

int64_t ComponentDataStore::GetInt(uint32_t classId, uint32_t fieldId, uint32_t slot) const
{
    return m_classes[classId].fields[fieldId].At<int64_t>(slot);
}

void ComponentDataStore::SetInt(uint32_t classId, uint32_t fieldId, uint32_t slot, int64_t value)
{
    m_classes[classId].fields[fieldId].At<int64_t>(slot) = value;
}

bool ComponentDataStore::GetBool(uint32_t classId, uint32_t fieldId, uint32_t slot) const
{
    return m_classes[classId].fields[fieldId].At<uint8_t>(slot) != 0;
}

void ComponentDataStore::SetBool(uint32_t classId, uint32_t fieldId, uint32_t slot, bool value)
{
    m_classes[classId].fields[fieldId].At<uint8_t>(slot) = value ? 1 : 0;
}

// ── per-element vector ───────────────────────────────────────────────────

void ComponentDataStore::GetVec2(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[2]) const
{
    const float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    out[0] = p[0];
    out[1] = p[1];
}

void ComponentDataStore::SetVec2(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[2])
{
    float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    p[0] = in[0];
    p[1] = in[1];
}

void ComponentDataStore::GetVec3(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[3]) const
{
    const float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
}

void ComponentDataStore::SetVec3(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[3])
{
    float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    p[0] = in[0];
    p[1] = in[1];
    p[2] = in[2];
}

void ComponentDataStore::GetVec4(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[4]) const
{
    const float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    out[3] = p[3];
}

void ComponentDataStore::SetVec4(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[4])
{
    float *p = m_classes[classId].fields[fieldId].FloatsAt(slot);
    p[0] = in[0];
    p[1] = in[1];
    p[2] = in[2];
    p[3] = in[3];
}

// ── batch gather/scatter ─────────────────────────────────────────────────

void ComponentDataStore::GatherFloat(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, double *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        out[i] = f.At<double>(slots[i]);
}

void ComponentDataStore::ScatterFloat(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const double *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        f.At<double>(slots[i]) = in[i];
}

void ComponentDataStore::GatherInt(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, int64_t *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        out[i] = f.At<int64_t>(slots[i]);
}

void ComponentDataStore::ScatterInt(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const int64_t *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        f.At<int64_t>(slots[i]) = in[i];
}

void ComponentDataStore::GatherBool(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, uint8_t *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        out[i] = f.At<uint8_t>(slots[i]);
}

void ComponentDataStore::ScatterBool(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const uint8_t *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i)
        f.At<uint8_t>(slots[i]) = in[i];
}

void ComponentDataStore::GatherVec2(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, float *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        const float *p = f.FloatsAt(slots[i]);
        out[i * 2 + 0] = p[0];
        out[i * 2 + 1] = p[1];
    }
}

void ComponentDataStore::ScatterVec2(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const float *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        float *p = f.FloatsAt(slots[i]);
        p[0] = in[i * 2 + 0];
        p[1] = in[i * 2 + 1];
    }
}

void ComponentDataStore::GatherVec3(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, float *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        const float *p = f.FloatsAt(slots[i]);
        out[i * 3 + 0] = p[0];
        out[i * 3 + 1] = p[1];
        out[i * 3 + 2] = p[2];
    }
}

void ComponentDataStore::ScatterVec3(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const float *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        float *p = f.FloatsAt(slots[i]);
        p[0] = in[i * 3 + 0];
        p[1] = in[i * 3 + 1];
        p[2] = in[i * 3 + 2];
    }
}

void ComponentDataStore::GatherVec4(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, float *out) const
{
    const auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        const float *p = f.FloatsAt(slots[i]);
        out[i * 4 + 0] = p[0];
        out[i * 4 + 1] = p[1];
        out[i * 4 + 2] = p[2];
        out[i * 4 + 3] = p[3];
    }
}

void ComponentDataStore::ScatterVec4(uint32_t cid, uint32_t fid, const uint32_t *slots, size_t count, const float *in)
{
    auto &f = m_classes[cid].fields[fid];
    for (size_t i = 0; i < count; ++i) {
        float *p = f.FloatsAt(slots[i]);
        p[0] = in[i * 4 + 0];
        p[1] = in[i * 4 + 1];
        p[2] = in[i * 4 + 2];
        p[3] = in[i * 4 + 3];
    }
}

void ComponentDataStore::Clear()
{
    m_classes.clear();
    m_classNameToId.clear();
}

} // namespace infernux
