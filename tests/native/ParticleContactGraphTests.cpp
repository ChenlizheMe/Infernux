// CPU regression of production graph declarations and emitted buffer barriers.
// No Vulkan instance, VkDevice, allocation, shader compilation, or GPU submission.
#include <function/renderer/particle/ParticleRenderGraph.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/renderer/vk/RenderGraph.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#ifdef NDEBUG
#error Test assertions must remain enabled
#endif
using namespace infernux;
using json = nlohmann::json;

// Explicit-instantiation access is restricted to this test TU. Production files
// and class definitions remain unchanged. It connects a CPU handle registry and
// intercepts the final Vulkan barrier call; all declarations/barrier decisions
// are made by the linked production methods, not a reimplementation.
template<class Tag, typename Tag::type Member> struct AuditAccess {
    friend typename Tag::type audit_member(Tag) { return Member; }
};
#define AUDIT_MEMBER(tag, member, ...) \
    struct tag { using type = __VA_ARGS__; friend type audit_member(tag); }; \
    template struct AuditAccess<tag, &vk::RenderGraph::member>
AUDIT_MEMBER(DeviceTag, m_rhiDevice, vk::VulkanRhiDevice* vk::RenderGraph::*);
AUDIT_MEMBER(PassesTag, m_passes, std::vector<vk::RenderPassData> vk::RenderGraph::*);
AUDIT_MEMBER(BarrierTag, m_cmdPipelineBarrier2, PFN_vkCmdPipelineBarrier2 vk::RenderGraph::*);
AUDIT_MEMBER(InsertTag, InsertBarriers, void (vk::RenderGraph::*)(VkCommandBuffer, uint32_t));
AUDIT_MEMBER(PrepareTag, PrepareExecutionResourceStates, void (vk::RenderGraph::*)());
AUDIT_MEMBER(CullTag, CullPasses, void (vk::RenderGraph::*)());
AUDIT_MEMBER(SortTag, TopologicalSort, bool (vk::RenderGraph::*)());
#undef AUDIT_MEMBER

template<class T> T Sentinel(uintptr_t n) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(n);
    else return static_cast<T>(n);
}

// Same recording-device boundary as ParticleGpuRuntimeTests, with buffers
// additionally registered in the real VulkanRhiDevice CPU handle registry.
struct RecordingDevice final : rhi::Device {
    vk::VulkanRhiDevice registry;
    uint32_t next = 1, buffersCreated = 0, buffersReleased = 0;
    uint32_t objectsCreated = 0, objectsReleased = 0;
    rhi::BufferHandle CreateBuffer(const rhi::BufferDesc& d) override {
        ++buffersCreated;
        return registry.RegisterBuffer(Sentinel<VkBuffer>(0x1000 + next++), d.byteSize);
    }
#define MAKE_OBJECT(handle, desc, name) \
    rhi::handle name(const rhi::desc&) override { ++objectsCreated; return {next++, 1}; }
    MAKE_OBJECT(TextureHandle, TextureDesc, CreateTexture)
    MAKE_OBJECT(TextureViewHandle, TextureViewDesc, CreateTextureView)
    MAKE_OBJECT(SamplerHandle, SamplerDesc, CreateSampler)
    MAKE_OBJECT(ShaderModuleHandle, ShaderModuleDesc, CreateShaderModule)
    MAKE_OBJECT(BindingLayoutHandle, BindingLayoutDesc, CreateBindingLayout)
    MAKE_OBJECT(BindGroupHandle, BindGroupDesc, CreateBindGroup)
    MAKE_OBJECT(GraphicsPipelineHandle, GraphicsPipelineDesc, CreateGraphicsPipeline)
    MAKE_OBJECT(ComputePipelineHandle, ComputePipelineDesc, CreateComputePipeline)
#undef MAKE_OBJECT
    bool WriteBuffer(rhi::BufferHandle h, uint64_t, const void* p, uint64_t n) override {
        return h.IsValid() && p && n;
    }
    void Release(rhi::BufferHandle h) noexcept override {
        if (h.IsValid()) { ++buffersReleased; registry.Release(h); }
    }
#define RELEASE_OBJECT(handle) \
    void Release(rhi::handle h) noexcept override { objectsReleased += h.IsValid(); }
    RELEASE_OBJECT(TextureHandle)
    RELEASE_OBJECT(TextureViewHandle)
    RELEASE_OBJECT(SamplerHandle)
    RELEASE_OBJECT(ShaderModuleHandle)
    RELEASE_OBJECT(BindingLayoutHandle)
    RELEASE_OBJECT(BindGroupHandle)
    RELEASE_OBJECT(GraphicsPipelineHandle)
    RELEASE_OBJECT(ComputePipelineHandle)
#undef RELEASE_OBJECT
};

static std::vector<VkBufferMemoryBarrier2> captured;
static uint32_t globalMemoryBarriers = 0;
static VKAPI_ATTR void VKAPI_CALL CaptureBarriers(VkCommandBuffer, const VkDependencyInfo* info) {
    globalMemoryBarriers += info->memoryBarrierCount;
    if (info->bufferMemoryBarrierCount)
        captured.insert(captured.end(), info->pBufferMemoryBarriers,
                        info->pBufferMemoryBarriers + info->bufferMemoryBarrierCount);
}
static bool HasVisibility(VkBuffer buffer) {
    return std::any_of(captured.begin(), captured.end(), [&](const auto& b) {
        return b.buffer == buffer && (b.srcAccessMask & VK_ACCESS_2_SHADER_WRITE_BIT) &&
               (b.dstAccessMask & VK_ACCESS_2_SHADER_READ_BIT);
    });
}
static vk::RenderPassData& FindPass(vk::RenderGraph& graph, const std::string& suffix) {
    auto& passes = graph.*audit_member(PassesTag{});
    auto it = std::find_if(passes.begin(), passes.end(), [&](const auto& p) {
        return p.name.size() >= suffix.size() &&
               p.name.compare(p.name.size() - suffix.size(), suffix.size(), suffix) == 0;
    });
    if (it == passes.end()) throw std::runtime_error("missing production pass " + suffix);
    return *it;
}
static bool Declares(vk::RenderGraph& graph, const std::vector<vk::ResourceAccess>& list,
                     VkBuffer buffer, rhi::Access access) {
    return std::any_of(list.begin(), list.end(), [&](const auto& a) {
        return graph.ResolveBuffer(a.handle) == buffer && rhi::HasAny(a.access, access);
    });
}

int main() {
    json result = {{"boundary", "CPU production ParticleRenderGraph::Attach and RenderGraph barrier methods; no GPU execution"},
                   {"cases", json::array()}, {"gpu_submissions", 0}};
    int failures = 0;
    auto record = [&](std::string name, bool ok, json detail = json::object()) {
        detail["case"] = name; detail["passed"] = ok;
        result["cases"].push_back(std::move(detail)); failures += !ok;
    };
    try {
        RecordingDevice device;
        {
            particle::ParticleGpuRuntime runtime;
            particle::ParticleGpuBounds bounds;
            particle::ParticleGpuGraphSpawnDomain spawn;
            particle::ParticleRenderGraph scheduler;
            vk::RenderGraph graph;
            graph.*audit_member(DeviceTag{}) = &device.registry;
            graph.*audit_member(BarrierTag{}) = &CaptureBarriers;
            std::array<vk::NativeQueueBinding, static_cast<size_t>(rhi::QueueRole::Count)> topology;
            topology.fill({0, 0}); // All stages intentionally run on one queue/family.
            graph.SetQueueTopology(topology);
            std::array<uint32_t, 5> words{0x07230203u, 0, 0, 0, 0};
            particle::ShaderBytecode shader{words.data(), words.size()};
            particle::GpuEmitterDesc desc;
            desc.capacity = 32; desc.stateStride = 32; desc.collisionEnabled = true;
            desc.collisionSceneHeader = {0xffd0u, 1}; desc.collisionSceneColliders = {0xffd1u, 1};
            desc.collisionSceneGridOffsets = {0xffd2u, 1}; desc.collisionSceneGridColliderIndices = {0xffd3u, 1};
            desc.collisionSceneMeshVertices = {0xffd4u, 1}; desc.collisionSceneMeshIndices = {0xffd5u, 1};
            desc.collisionSceneMeshBvhNodes = {0xffd6u, 1};
            for (auto& kernel : desc.kernels) kernel = shader;
            desc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
            if (!runtime.Create(device, desc)) throw std::runtime_error("runtime Create");
            particle::GpuParticleBoundsDesc bd;
            bd.capacity = desc.capacity; bd.visibility = runtime.VisibilityBuffer();
            bd.sourceIndices = runtime.RenderIndexBuffer(); bd.sourceIndirectArguments = runtime.IndirectBuffer();
            bd.simulationControl = runtime.SimulationControlBuffer(); bd.program = {shader, shader, shader};
            particle::ParticleGpuGraphSpawnDomain::GraphResources spawnResources;
            if (!bounds.Create(device, bd) || !spawn.Create(device, 1, 1, {shader, shader}, {}) ||
                !spawn.RegisterEmitter(0, runtime) || !spawn.Attach(graph, "ContactSpawn", spawnResources) ||
                !scheduler.Attach(graph, runtime, bounds, spawn, spawnResources, 0, "ContactParticle"))
                throw std::runtime_error("production graph Attach");
            auto& solve = FindPass(graph, "ContactSolve");
            auto& dispatch = FindPass(graph, "ContactDispatch");
            const auto resources = runtime.ContactResources();
            VkBuffer records = device.registry.Resolve(resources.contactRecords);
            VkBuffer states = device.registry.Resolve(resources.particleStates);
            VkBuffer particleStates = device.registry.Resolve(runtime.StateBuffer());
            VkBuffer transforms = device.registry.Resolve(runtime.TransformBuffer());
            VkBuffer work = device.registry.Resolve(resources.workItems);
            record("solve_declares_record_write", Declares(graph, solve.writes, records, rhi::Access::ShaderWrite));
            record("solve_declares_particle_state_write", Declares(graph, solve.writes, states, rhi::Access::ShaderWrite));
            record("dispatch_declares_particle_state_read", Declares(graph, dispatch.reads, states, rhi::Access::ShaderRead));
            record("solve_reads_particle_generation", Declares(graph, solve.reads, particleStates, rhi::Access::ShaderRead));
            record("dispatch_reads_transforms", Declares(graph, dispatch.reads, transforms, rhi::Access::UniformRead));
            (graph.*audit_member(CullTag{}))();
            if (!(graph.*audit_member(SortTag{}))()) throw std::runtime_error("graph TopologicalSort");
            const auto order = graph.GetExecutionPassNames();
            result["production_execution_order"] = order;
            (graph.*audit_member(PrepareTag{}))();
            bool sawSolve = false, sawDispatch = false;
            for (const auto& name : order) {
                auto& pass = FindPass(graph, name);
                captured.clear();
                (graph.*audit_member(InsertTag{}))(VK_NULL_HANDLE, pass.id);
                if (pass.name == solve.name) {
                    sawSolve = true;
                    record("update_to_solve_particle_generation_visibility", HasVisibility(particleStates));
                }
                if (pass.name == dispatch.name) {
                    if (!sawSolve) throw std::runtime_error("contact dispatch before solve");
                    sawDispatch = true;
                    record("solve_to_dispatch_record_visibility", HasVisibility(records));
                    record("solve_to_dispatch_particle_state_visibility", HasVisibility(states));
                    record("solve_to_dispatch_work_item_visibility_control", HasVisibility(work));
                    result["dispatch_barrier_count"] = captured.size();
                    break;
                }
            }
            if (!sawDispatch) throw std::runtime_error("contact stage was culled");
            result["production_debug_graph"] = graph.GetDebugString();
            // Normal control uses public PassBuilder declarations. It differs
            // only in accurately declaring the two actual writes and reads.
            vk::RenderGraph corrected;
            corrected.*audit_member(BarrierTag{}) = &CaptureBarriers;
            corrected.SetQueueTopology(topology);
            vk::ResourceHandle controlRecords, controlStates;
            auto writer = corrected.AddComputePass("CorrectSolve", [&](vk::PassBuilder& b) {
                controlRecords = b.ReadWrite(b.ImportBuffer("Records", records, 96), rhi::PipelineStage::ComputeShader);
                controlStates = b.ReadWrite(b.ImportBuffer("States", states, 16), rhi::PipelineStage::ComputeShader);
                return [](vk::RenderContext&) {};
            });
            auto reader = corrected.AddComputePass("CorrectDispatch", [&](vk::PassBuilder& b) {
                b.ReadStorageBuffer(controlRecords); b.ReadStorageBuffer(controlStates);
                return [](vk::RenderContext&) {};
            });
            (corrected.*audit_member(PrepareTag{}))();
            (corrected.*audit_member(InsertTag{}))(VK_NULL_HANDLE, writer.id);
            captured.clear();
            (corrected.*audit_member(InsertTag{}))(VK_NULL_HANDLE, reader.id);
            record("correct_declarations_record_visibility_control", HasVisibility(records));
            record("correct_declarations_particle_state_visibility_control", HasVisibility(states));
            result["global_memory_barriers"] = globalMemoryBarriers;
            corrected.Destroy(); graph.Destroy();
            spawn.Destroy(); bounds.Destroy(); runtime.Destroy();
        }
        result["buffers_created"] = device.buffersCreated;
        result["buffers_released"] = device.buffersReleased;
        result["objects_created"] = device.objectsCreated;
        result["objects_released"] = device.objectsReleased;
        result["cleanup_complete"] = device.buffersCreated == device.buffersReleased &&
                                       device.objectsCreated == device.objectsReleased;
        if (!result["cleanup_complete"].get<bool>()) throw std::runtime_error("resource cleanup");
        result["case_count"] = result["cases"].size(); result["failures"] = failures;
        std::cout << result.dump(2) << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        result["fixture_error"] = e.what(); std::cout << result.dump(2) << '\n'; return 2;
    }
}
