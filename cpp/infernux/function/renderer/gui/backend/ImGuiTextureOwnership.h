#pragma once

#include "../ImGuiTextureAllocationLease.h"
#include <vulkan/vulkan.h>

// Existing maintained backend extension, also used by atlas destruction.
void ImGui_ImplVulkan_SetTextureLinearColor(VkDescriptorSet descriptorSet, bool linearColor);
void ImGui_ImplVulkan_SetTextureRetirement(ImGuiTextureAllocationLease::Retire retire);
std::shared_ptr<ImGuiTextureAllocationLease> ImGui_ImplVulkan_AcquireAtlasAllocation(VkDescriptorSet descriptor);
// Called after the engine has stopped recording views and drained the device,
// before the backend's pools/device objects are destroyed.
void ImGui_ImplVulkan_ReleaseAtlasAllocations();
