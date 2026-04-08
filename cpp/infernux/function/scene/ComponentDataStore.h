#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

/// SoA data store for Python InxComponent numeric fields.
///
/// Each registered component class gets its own set of parallel arrays
/// (one per numeric field).  Per-element access is O(1) via slot index.
/// Batch gather/scatter enables efficient numpy ↔ engine data transfer.
class ComponentDataStore
{
  public:
    enum class DataType : uint8_t
    {
        Float64, // Python float → double
        Int64,   // Python int → int64_t
        Bool,    // Python bool → uint8_t
        Vec2,    // Vector2 → 2 × float
        Vec3,    // Vector3 → 3 × float
        Vec4,    // vec4f   → 4 × float
    };

    static ComponentDataStore &Instance();

    // ── class / field registration ──

    uint32_t RegisterClass(const std::string &className);
    uint32_t RegisterField(uint32_t classId, const std::string &fieldName, DataType type);
    uint32_t GetClassId(const std::string &className) const;
    uint32_t GetFieldId(uint32_t classId, const std::string &fieldName) const;

    // ── slot lifecycle ──

    uint32_t AllocateSlot(uint32_t classId);
    void ReleaseSlot(uint32_t classId, uint32_t slot);

    // ── per-element scalar access ──

    double GetFloat(uint32_t classId, uint32_t fieldId, uint32_t slot) const;
    void SetFloat(uint32_t classId, uint32_t fieldId, uint32_t slot, double value);

    int64_t GetInt(uint32_t classId, uint32_t fieldId, uint32_t slot) const;
    void SetInt(uint32_t classId, uint32_t fieldId, uint32_t slot, int64_t value);

    bool GetBool(uint32_t classId, uint32_t fieldId, uint32_t slot) const;
    void SetBool(uint32_t classId, uint32_t fieldId, uint32_t slot, bool value);

    // ── per-element vector access ──

    void GetVec2(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[2]) const;
    void SetVec2(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[2]);

    void GetVec3(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[3]) const;
    void SetVec3(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[3]);

    void GetVec4(uint32_t classId, uint32_t fieldId, uint32_t slot, float out[4]) const;
    void SetVec4(uint32_t classId, uint32_t fieldId, uint32_t slot, const float in[4]);

    // ── batch gather/scatter ──

    void GatherFloat(uint32_t classId, uint32_t fieldId,
                     const uint32_t *slots, size_t count, double *out) const;
    void ScatterFloat(uint32_t classId, uint32_t fieldId,
                      const uint32_t *slots, size_t count, const double *in);

    void GatherInt(uint32_t classId, uint32_t fieldId,
                   const uint32_t *slots, size_t count, int64_t *out) const;
    void ScatterInt(uint32_t classId, uint32_t fieldId,
                    const uint32_t *slots, size_t count, const int64_t *in);

    void GatherBool(uint32_t classId, uint32_t fieldId,
                    const uint32_t *slots, size_t count, uint8_t *out) const;
    void ScatterBool(uint32_t classId, uint32_t fieldId,
                     const uint32_t *slots, size_t count, const uint8_t *in);

    void GatherVec3(uint32_t classId, uint32_t fieldId,
                    const uint32_t *slots, size_t count, float *out) const;
    void ScatterVec3(uint32_t classId, uint32_t fieldId,
                     const uint32_t *slots, size_t count, const float *in);

    void GatherVec2(uint32_t classId, uint32_t fieldId,
                    const uint32_t *slots, size_t count, float *out) const;
    void ScatterVec2(uint32_t classId, uint32_t fieldId,
                     const uint32_t *slots, size_t count, const float *in);

    void GatherVec4(uint32_t classId, uint32_t fieldId,
                    const uint32_t *slots, size_t count, float *out) const;
    void ScatterVec4(uint32_t classId, uint32_t fieldId,
                     const uint32_t *slots, size_t count, const float *in);

    /// Reset everything (e.g. scene unload).
    void Clear();

  private:
    ComponentDataStore() = default;

    static size_t ElementSize(DataType type);

    struct FieldStorage
    {
        DataType type{};
        std::vector<uint8_t> data; // raw byte buffer
        size_t elementSize = 0;

        void Grow(size_t newCapacity);
        void ResetSlot(size_t slot);

        template <typename T>
        T &At(size_t slot)
        {
            return *reinterpret_cast<T *>(data.data() + slot * elementSize);
        }
        template <typename T>
        const T &At(size_t slot) const
        {
            return *reinterpret_cast<const T *>(data.data() + slot * elementSize);
        }
        float *FloatsAt(size_t slot) { return reinterpret_cast<float *>(data.data() + slot * elementSize); }
        const float *FloatsAt(size_t slot) const
        {
            return reinterpret_cast<const float *>(data.data() + slot * elementSize);
        }
    };

    struct ClassStorage
    {
        std::vector<FieldStorage> fields;
        std::unordered_map<std::string, uint32_t> fieldNameToId;
        std::vector<uint8_t> alive;
        std::vector<uint32_t> nextFree;
        uint32_t freeHead = UINT32_MAX;
        size_t capacity = 0;
        size_t aliveCount = 0;

        void GrowTo(size_t newCapacity);
    };

    std::vector<ClassStorage> m_classes;
    std::unordered_map<std::string, uint32_t> m_classNameToId;
};

} // namespace infernux
