#include <function/renderer/particle/ParticleGpuSurfaceBinding.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>
#ifdef NDEBUG
#error Audit assertions must remain enabled
#endif
using namespace infernux;
using json = nlohmann::json;

struct RecordingDevice final : rhi::Device {
    uint32_t next = 1, created = 0, released = 0;
    std::unordered_map<uint32_t, rhi::BindGroupDesc> groups;
#define MAKE_OBJECT(handle, desc, name) \
    rhi::handle name(const rhi::desc&) override { ++created; return {next++, 1}; }
    MAKE_OBJECT(BufferHandle, BufferDesc, CreateBuffer)
    MAKE_OBJECT(TextureHandle, TextureDesc, CreateTexture)
    MAKE_OBJECT(TextureViewHandle, TextureViewDesc, CreateTextureView)
    MAKE_OBJECT(SamplerHandle, SamplerDesc, CreateSampler)
    MAKE_OBJECT(ShaderModuleHandle, ShaderModuleDesc, CreateShaderModule)
    MAKE_OBJECT(BindingLayoutHandle, BindingLayoutDesc, CreateBindingLayout)
    MAKE_OBJECT(GraphicsPipelineHandle, GraphicsPipelineDesc, CreateGraphicsPipeline)
    MAKE_OBJECT(ComputePipelineHandle, ComputePipelineDesc, CreateComputePipeline)
#undef MAKE_OBJECT
    rhi::BindGroupHandle CreateBindGroup(const rhi::BindGroupDesc& desc) override {
        ++created; auto id=next++; groups.emplace(id,desc); return {id,1};
    }
    bool WriteBuffer(rhi::BufferHandle h,uint64_t,const void* p,uint64_t n) override {
        return h.IsValid() && p && n;
    }
#define RELEASE_OBJECT(handle) \
    void Release(rhi::handle h) noexcept override { released += h.IsValid(); }
    RELEASE_OBJECT(BufferHandle)
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

static std::shared_ptr<rhi::TextureGpuViewSlot> MakeSlot(RecordingDevice& device, const std::string& source) {
    const auto view=device.CreateTextureView({}); const auto sampler=device.CreateSampler({});
    auto owner=std::shared_ptr<const void>(new int(1),[&device,view,sampler](const void* p){
        device.Release(view); device.Release(sampler); delete static_cast<const int*>(p);
    });
    auto slot=std::make_shared<rhi::TextureGpuViewSlot>(source);
    assert(slot->TryPublish(std::make_shared<const rhi::TextureGpuView>(
        source,1,rhi::TextureHandle{},view,sampler,1,std::move(owner))));
    return slot;
}

int main() {
    json result={{"cases",json::array()},{"boundary","CPU production SurfaceBinding/Material with controlled resolver and recording device; no GPU"},
                 {"gpu_submissions",0}};
    unsigned failed=0;
    for (const std::string mode : {"already_ready","pending_then_ready","failed_then_ready"}) {
        RecordingDevice device;
        json observation;
        {
            const std::string guid="11111111-2222-4333-8444-555555555555";
            auto material=std::make_shared<InxMaterial>("AuditParticle");
            material->SetTextureGuid("texSampler","white");
            // Normal durable-document load: missing references are expressly
            // preserved by RestoreTextureGuidReference for Undo/reimport.
            auto document=json::parse(material->Serialize());
            // A bare InxMaterial has no shader pair; the persisted document
            // requires one. Use the same authored pair as the linked artifact.
            document["shaders"]["vertex"]["shader_id"]="Tests/ParticleSprite";
            document["shaders"]["fragment"]["shader_id"]="Tests/ParticleUnlit";
            document["properties"]["texSampler"]["guid"]=guid;
            assert(material->DeserializeDocument(document));
            auto artifact=std::make_shared<ShaderProgramArtifact>();
            artifact->key={{"Tests/ParticleSprite","Tests/ParticleUnlit"},1};
            artifact->domain=ShaderProgramDomain::ParticleSprite;
            artifact->compatibilitySignature=1;
            artifact->properties={{"texSampler","Texture2D","","white",ShaderProgramStageMask::Fragment,
                                   false,std::nullopt,std::nullopt,0,0,0}};
            ShaderProgramArtifact::PassVariant pass;
            pass.compatibilitySignature=1;
            pass.vertexSpirv.resize(5*sizeof(uint32_t)); pass.fragmentSpirv.resize(5*sizeof(uint32_t));
            const uint32_t magic=0x07230203u;
            std::memcpy(pass.vertexSpirv.data(),&magic,sizeof(magic));
            std::memcpy(pass.fragmentSpirv.data(),&magic,sizeof(magic));
            artifact->variants.push_back(std::move(pass)); assert(artifact->IsValid());
            auto white=MakeSlot(device,"white"); auto real=MakeSlot(device,guid);
            auto status=mode=="already_ready" ? particle::GpuBillboardTextureStatus::Ready :
                        mode=="pending_then_ready" ? particle::GpuBillboardTextureStatus::Pending :
                        particle::GpuBillboardTextureStatus::Failed;
            unsigned realCalls=0;
            auto resolver=[&](const std::string& id,const std::string& binding,particle::GpuParticleTextureRequest request){
                assert(binding=="texSampler" && request==particle::GpuParticleTextureRequest::Poll);
                assert(id==guid || id=="white" || id.empty());
                if(id==guid) { ++realCalls; if(status!=particle::GpuBillboardTextureStatus::Ready)
                    return particle::GpuBillboardTextureLease{status}; }
                auto slot=id==guid ? real : white; auto publication=slot->Acquire();
                return particle::GpuBillboardTextureLease{particle::GpuBillboardTextureStatus::Ready,
                    publication->GetView(),publication->GetSampler(),slot,publication};
            };
            particle::ParticleGpuSurfaceBinding surface;
            assert(surface.Create(device,artifact,material,{}, {},resolver,nullptr));
            auto bound=[&](){
                const auto& d=device.groups.at(surface.ResolveBindGroup().index);
                assert(d.textureCount==1 && d.textures[0].binding==2);
                return d.textures[0].texture;
            };
            const auto before=bound(); const auto callsBefore=realCalls;
            assert(before==(mode=="already_ready" ? real : white)->Acquire()->GetView());
            bool unchangedFailureIsCached = true, invalidationRetriesOnce = true;
            if (mode == "failed_then_ready") {
                for (unsigned frame = 0; frame < 3; ++frame) {
                    material->SetFloat("animatedIntensity", static_cast<float>(frame));
                    assert(surface.RefreshTextureBindings(false));
                }
                unchangedFailureIsCached = realCalls == callsBefore;
                material->InvalidateTextureAssets(guid, false);
                for (unsigned frame = 0; frame < 3; ++frame)
                    assert(surface.RefreshTextureBindings(false));
                invalidationRetriesOnce = realCalls == callsBefore + 1;
            }
            const auto recoveryCallsBefore = realCalls;
            const auto versionBefore=material->GetVersion();
            status=particle::GpuBillboardTextureStatus::Ready;
            material->InvalidateTextureAssets(guid,false); // Same formal asset-notification path.
            assert(material->GetVersion()>versionBefore);
            for(unsigned frame=0;frame<3;++frame) {
                assert(surface.RefreshMaterialBuffer(false));
                assert(surface.RefreshTextureBindings(false));
            }
            const auto after=bound();
            const bool passed=after==real->Acquire()->GetView() && unchangedFailureIsCached && invalidationRetriesOnce;
            observation={{"case",mode},{"passed",passed},{"before_view",before.index},{"after_view",after.index},
                         {"real_view",real->Acquire()->GetView().index},{"real_resolver_calls_after_ready",realCalls-recoveryCallsBefore},
                         {"unchanged_failure_is_cached", unchangedFailureIsCached},
                         {"invalidation_retries_once", invalidationRetriesOnce},
                         {"material_version_changed",true},{"refresh_frames",3}};
            if(!passed) ++failed;
            surface.Destroy(); real.reset(); white.reset();
        }
        observation["objects_created"]=device.created; observation["objects_released"]=device.released;
        observation["cleanup_complete"]=device.created==device.released;
        assert(device.created==device.released);
        result["cases"].push_back(std::move(observation));
    }
    result["case_count"]=result["cases"].size(); result["failed"]=failed;
    result["passed"]=result["cases"].size()-failed; result["cleanup_complete"]=true;
    std::cout<<result.dump(2)<<'\n'; return failed ? 1 : 0;
}
