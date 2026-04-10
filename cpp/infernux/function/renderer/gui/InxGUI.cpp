#include "InxGUI.h"
#include "InxGUIContext.h"
#include <function/editor/EditorTheme.h>
#include <function/renderer/vk/VkRenderUtils.h>

#include <SDL3/SDL.h>
#include <algorithm>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono>
#include <cmath>
#include <core/log/InxLog.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <memory>
#include <platform/input/InputManager.h>

namespace infernux
{

InxGUI::InxGUI(InxVkCoreModular *vkCore) : m_vkCore_ptr(vkCore)
{
}

InxGUI::~InxGUI()
{
    Shutdown();

    ImGui::DestroyContext(m_imguiContext_ptr);
    m_imguiContext_ptr = nullptr;
}

void InxGUI::Init(SDL_Window *window)
{
    m_window_ptr = window;

    // Detect display DPI scale (e.g. 2.0 for 200% Windows scaling)
    m_dpiScale = SDL_GetWindowDisplayScale(window);
    if (m_dpiScale <= 0.0f)
        m_dpiScale = 1.0f;
    InxGUIContext::s_dpiScale = m_dpiScale;
    INXLOG_DEBUG("Display scale: ", m_dpiScale);

    IMGUI_CHECKVERSION();
    m_imguiContext_ptr = ImGui::CreateContext();
    ImGui::SetCurrentContext(m_imguiContext_ptr);
    ImGui::StyleColorsDark();

    // =========================================================================
    // Notion-style dark theme — matches launcher palette (style.py)
    // bg_base=#191919  bg_surface=#202020  bg_hover=#2a2a2a
    // bg_selected=#333333  border=#2f2f2f  text=#cfcfcf
    // text_secondary=#707070  text_muted=#555555  accent=white
    //
    // NOTE: Swapchain is VK_FORMAT_B8G8R8A8_UNORM — no hardware sRGB
    // encoding. ImGui colours are already in display (sRGB) space and
    // written directly to the framebuffer.
    // =========================================================================
    {
        ImGuiStyle &style = ImGui::GetStyle();
        ImVec4 *c = style.Colors;

        // No conversion needed: UNORM swapchain writes values as-is.
        auto L = [](float r, float g, float b, float a) -> ImVec4 { return ImVec4(r, g, b, a); };

        // Accent color shorthand (Infernux red #EB5757)
        constexpr float AR = EditorTheme::ACCENT_R;
        constexpr float AG = EditorTheme::ACCENT_G;
        constexpr float AB = EditorTheme::ACCENT_B;

        // --- Text ---
        c[ImGuiCol_Text] = L(0.812f, 0.812f, 0.812f, 1.00f);           // #CFCFCF
        c[ImGuiCol_TextDisabled] = L(0.333f, 0.333f, 0.333f, 1.00f);   // #555555
        c[ImGuiCol_TextSelectedBg] = L(0.200f, 0.200f, 0.200f, 0.60f); // #333333

        // --- Backgrounds ---
        c[ImGuiCol_WindowBg] = L(0.098f, 0.098f, 0.098f, 1.00f);       // #191919
        c[ImGuiCol_ChildBg] = L(0.125f, 0.125f, 0.125f, 1.00f);        // #202020
        c[ImGuiCol_PopupBg] = L(0.125f, 0.125f, 0.125f, 0.98f);        // #202020
        c[ImGuiCol_FrameBg] = L(0.125f, 0.125f, 0.125f, 1.00f);        // #202020
        c[ImGuiCol_FrameBgHovered] = L(0.165f, 0.165f, 0.165f, 1.00f); // #2A2A2A
        c[ImGuiCol_FrameBgActive] = L(0.240f, 0.170f, 0.170f, 1.00f);  // reddish tint when active

        // --- Title bar ---
        c[ImGuiCol_TitleBg] = L(0.098f, 0.098f, 0.098f, 1.00f);
        c[ImGuiCol_TitleBgActive] = L(0.098f, 0.098f, 0.098f, 1.00f);
        c[ImGuiCol_TitleBgCollapsed] = L(0.098f, 0.098f, 0.098f, 0.75f);

        // --- MenuBar ---
        c[ImGuiCol_MenuBarBg] = L(0.098f, 0.098f, 0.098f, 1.00f);

        // --- Scrollbar ---
        c[ImGuiCol_ScrollbarBg] = L(0.098f, 0.098f, 0.098f, 0.00f);
        c[ImGuiCol_ScrollbarGrab] = L(0.184f, 0.184f, 0.184f, 1.00f);
        c[ImGuiCol_ScrollbarGrabHovered] = L(0.333f, 0.333f, 0.333f, 1.00f);
        c[ImGuiCol_ScrollbarGrabActive] = L(0.439f, 0.439f, 0.439f, 1.00f);

        // --- Interactive accent ---
        c[ImGuiCol_CheckMark] = L(AR, AG, AB, 1.00f);              // #EB5757
        c[ImGuiCol_SliderGrab] = L(AR, AG, AB, 0.88f);             // #EB5757
        c[ImGuiCol_SliderGrabActive] = L(AR, AG, AB, 1.00f);       // #EB5757
        c[ImGuiCol_NavHighlight] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f); // no outline on active fields

        // --- Buttons --- subtle dark surface with red accent on interaction
        c[ImGuiCol_Button] = L(0.165f, 0.165f, 0.165f, 1.00f); // #2A2A2A
        c[ImGuiCol_ButtonHovered] = L(0.220f, 0.165f, 0.165f, 1.00f);
        c[ImGuiCol_ButtonActive] = L(0.270f, 0.180f, 0.180f, 1.00f);

        // --- Header ---
        c[ImGuiCol_Header] = L(0.200f, 0.200f, 0.200f, 1.00f);
        c[ImGuiCol_HeaderHovered] = L(0.200f, 0.160f, 0.160f, 1.00f);
        c[ImGuiCol_HeaderActive] = L(0.240f, 0.170f, 0.170f, 1.00f);

        // --- Border / Separator ---
        c[ImGuiCol_Border] = L(0.184f, 0.184f, 0.184f, 1.00f);
        c[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        c[ImGuiCol_Separator] = L(0.184f, 0.184f, 0.184f, 1.00f);
        c[ImGuiCol_SeparatorHovered] = L(AR, AG, AB, 0.60f); // #EB5757
        c[ImGuiCol_SeparatorActive] = L(AR, AG, AB, 0.80f);  // #EB5757

        // --- Resize grip ---
        c[ImGuiCol_ResizeGrip] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        c[ImGuiCol_ResizeGripHovered] = L(AR, AG, AB, 0.30f);
        c[ImGuiCol_ResizeGripActive] = L(AR, AG, AB, 0.50f);

        // --- Tabs ---
        c[ImGuiCol_Tab] = L(0.098f, 0.098f, 0.098f, 1.00f);               // #191919
        c[ImGuiCol_TabHovered] = L(0.165f, 0.165f, 0.165f, 1.00f);        // #2A2A2A
        c[ImGuiCol_TabSelected] = L(0.125f, 0.125f, 0.125f, 1.00f);       // #202020
        c[ImGuiCol_TabSelectedOverline] = L(AR, AG, AB, 1.00f);           // #EB5757 red overline
        c[ImGuiCol_TabDimmed] = L(0.098f, 0.098f, 0.098f, 1.00f);         // #191919
        c[ImGuiCol_TabDimmedSelected] = L(0.125f, 0.125f, 0.125f, 1.00f); // #202020
        c[ImGuiCol_TabDimmedSelectedOverline] = L(AR, AG, AB, 0.60f);     // dimmer red overline

        // --- Docking ---
        c[ImGuiCol_DockingPreview] = L(AR, AG, AB, 0.25f); // #EB5757
        c[ImGuiCol_DockingEmptyBg] = L(0.060f, 0.060f, 0.060f, 1.00f);

        // --- Plots ---
        c[ImGuiCol_PlotLines] = L(0.439f, 0.439f, 0.439f, 1.00f);
        c[ImGuiCol_PlotHistogram] = L(0.812f, 0.812f, 0.812f, 1.00f);

        // --- Drag-drop target highlight --- pure white border
        c[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        c[ImGuiCol_DragDropTargetBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

        // --- Modal dim ---
        c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.56f);

        // --- Table ---
        c[ImGuiCol_TableHeaderBg] = L(0.125f, 0.125f, 0.125f, 1.00f);
        c[ImGuiCol_TableBorderStrong] = L(0.184f, 0.184f, 0.184f, 1.00f);
        c[ImGuiCol_TableBorderLight] = L(0.149f, 0.149f, 0.149f, 1.00f);
        c[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        c[ImGuiCol_TableRowBgAlt] = L(1.000f, 1.000f, 1.000f, 0.02f);

        // =====================================================================
        // Style dimensions — Notion-style clean, modern spacing
        // =====================================================================
        style.WindowPadding = ImVec2(10.0f, 10.0f);
        style.FramePadding = ImVec2(8.0f, 3.0f);
        style.CellPadding = ImVec2(4.0f, 4.0f);
        style.ItemSpacing = ImVec2(8.0f, 6.0f);
        style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
        style.IndentSpacing = 18.0f;
        style.ScrollbarSize = 8.0f; // thin Notion scrollbar
        style.GrabMinSize = 6.0f;

        // Borders — minimal, but keep inputs readable
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.FrameBorderSize = 1.0f; // visible border around input fields
        style.TabBorderSize = 0.0f;
        style.TabBarBorderSize = 1.0f;

        // Rounding — project-wide square language
        style.WindowRounding = 0.0f; // main window stays square
        style.ChildRounding = 0.0f;
        style.FrameRounding = 0.0f;
        style.PopupRounding = 0.0f;
        style.ScrollbarRounding = 0.0f;
        style.GrabRounding = 0.0f;
        style.TabRounding = 0.0f;

        // Anti-aliasing
        style.AntiAliasedLines = true;
        style.AntiAliasedFill = true;

        // Scale all style dimensions for high-DPI displays
        if (m_dpiScale > 1.0f) {
            style.ScaleAllSizes(m_dpiScale);
        }
    }

    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport (optional, can cause issues)

    // Docking configuration
    io.ConfigDockingWithShift = false;    // Dock without holding shift
    io.ConfigDockingAlwaysTabBar = true;  // Always show tab bar for docked windows
    io.ConfigDragClickToInputText = true; // Single click-release on DragFloat → text input

    ImGui_ImplSDL3_InitForVulkan(window);

    VkDevice device = m_vkCore_ptr->GetDevice();
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000 * IM_ARRAYSIZE(poolSizes);
    poolInfo.poolSizeCount = static_cast<uint32_t>(IM_ARRAYSIZE(poolSizes));
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool_vk) != VK_SUCCESS) {
        INXLOG_FATAL("Failed to create descriptor pool for ImGui.");
        return;
    }

    // Create a minimal compatible render pass for ImGui (swapchain format, no depth)
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = m_vkCore_ptr->GetSwapchainFormat();
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve previous content
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        const VkSubpassDependency dependency = vkrender::MakePipelineCompatibleSubpassDependency();

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAttachment;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_imguiRenderPass) != VK_SUCCESS) {
            INXLOG_FATAL("Failed to create ImGui render pass.");
            return;
        }
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = m_vkCore_ptr->GetInstance();
    initInfo.PhysicalDevice = m_vkCore_ptr->GetPhysicalDevice();
    initInfo.Device = device;
    initInfo.QueueFamily = m_vkCore_ptr->GetDeviceContext().GetQueueIndices().graphicsFamily.value();
    initInfo.Queue = m_vkCore_ptr->GetGraphicsQueue();
    initInfo.DescriptorPool = m_descriptorPool_vk;
    initInfo.MinImageCount = m_vkCore_ptr->GetSwapchainImageCount();
    initInfo.ImageCount = m_vkCore_ptr->GetSwapchainImageCount();
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;

    // Use InxGUI's own render pass instead of pulling from InxVkCoreModular
    initInfo.PipelineInfoMain.RenderPass = m_imguiRenderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        INXLOG_FATAL("Failed to initialize ImGui Vulkan implementation.");
        return;
    }

    // Font texture is now created automatically by the backend

    // Initialize resource preview manager
    m_resourcePreviewManager.SetGUI(this);
}

void InxGUI::SetGUIFont(const char *fontPath, float fontSize)
{
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->Clear();

    // Scale font size by display DPI (e.g. 14px * 2.0 = 28px on 200% display)
    float scaledSize = fontSize * m_dpiScale;
    INXLOG_DEBUG("Loading font at ", scaledSize, "px (base ", fontSize, " x scale ", m_dpiScale, ")");

    ImFontConfig fontConfig;
    fontConfig.FontDataOwnedByAtlas = false;

    // Since ImGui 1.92+ with RendererHasTextures, glyph ranges are no longer
    // needed. Glyphs are loaded on-demand at any requested size, so the atlas
    // grows incrementally instead of pre-baking all CJK glyphs up-front.
    ImFont *font = io.Fonts->AddFontFromFileTTF(fontPath, scaledSize, &fontConfig);
    if (font == nullptr) {
        INXLOG_WARN("InxGUI::SetGUIFont(): Failed to load font from ", fontPath);
        return;
    }

    // Font texture is now created automatically by the backend
    // No need to manually call ImGui_ImplVulkan_CreateFontsTexture()
}

void InxGUI::BuildFrame()
{
    static auto ctx = std::make_unique<InxGUIContext>();

    ImGui_ImplSDL3_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    // When the cursor is locked (game mode), suppress all mouse input from
    // reaching ImGui so editor panels (Inspector, Hierarchy, etc.) don't
    // react to invisible cursor movement — matching Unity behaviour.
    if (InputManager::Instance().IsCursorLocked()) {
        ImGuiIO &io = ImGui::GetIO();
        io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); ++i)
            io.MouseDown[i] = false;
        io.MouseWheel = 0.0f;
        io.MouseWheelH = 0.0f;
    }

    // In player mode, skip DockSpace/DockBuilder entirely — they are only
    // needed for the editor's multi-panel layout.  The player registers a
    // single full-screen renderable (PlayerGUI), so docking is wasted work.
    if (!m_playerMode) {
        // Create a full-screen DockSpace (reserve bottom strip for the Python status bar)
        const float kStatusBarHeight = 24.0f * m_dpiScale; // must match _HEIGHT in status_bar.py
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y - kStatusBarHeight));
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGuiWindowFlags dockSpaceFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                          ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                          ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpaceWindow", nullptr, dockSpaceFlags);
        ImGui::PopStyleVar(3);

        // Check whether a saved layout already exists BEFORE DockSpace()
        // creates the node.  If the node doesn't exist yet (first launch or
        // imgui.ini was deleted by the Python layout-version mechanism), we
        // need to build the default Unity-style layout.
        ImGuiID dockspaceId = ImGui::GetID("MainDockSpace");
        bool needsDefaultLayout = (ImGui::DockBuilderGetNode(dockspaceId) == nullptr);

        ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        // Setup default Unity-style layout only when no saved layout exists.
        // This preserves user customizations across restarts while still
        // providing the correct initial tab arrangement on first launch
        // (or after a layout-version bump that deletes imgui.ini).
        if (needsDefaultLayout) {

            ImGui::DockBuilderRemoveNode(dockspaceId);
            ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspaceId,
                                          ImVec2(viewport->WorkSize.x, viewport->WorkSize.y - kStatusBarHeight));

            // Split: Main area | Right panel (Inspector)
            ImGuiID dockMain;
            ImGuiID dockRight;
            ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Right, 0.25f, &dockRight, &dockMain);

            // Split main: Top area (Hierarchy+Scene) | Bottom (Console/Project)
            ImGuiID dockTop;
            ImGuiID dockBottom;
            ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down, 0.30f, &dockBottom, &dockTop);

            // Split top: Left (Hierarchy) | Center-top (Toolbar+Scene)
            ImGuiID dockLeft;
            ImGuiID dockCenterTop;
            ImGui::DockBuilderSplitNode(dockTop, ImGuiDir_Left, 0.20f, &dockLeft, &dockCenterTop);

            // Split center-top: Toolbar (thin strip) | Scene/Game
            ImGuiID dockToolbar;
            ImGuiID dockScene;
            ImGui::DockBuilderSplitNode(dockCenterTop, ImGuiDir_Up, 0.04f, &dockToolbar, &dockScene);

            // Set a fixed size for the toolbar node so it doesn't stretch
            ImGui::DockBuilderSetNodeSize(dockToolbar, ImVec2(viewport->WorkSize.x, 36));

            // Hide tab bar on toolbar node — it should be locked in place
            ImGuiDockNode *toolbarNode = ImGui::DockBuilderGetNode(dockToolbar);
            if (toolbarNode) {
                toolbarNode->SetLocalFlags(toolbarNode->LocalFlags | ImGuiDockNodeFlags_NoTabBar |
                                           ImGuiDockNodeFlags_NoDockingSplit | ImGuiDockNodeFlags_NoResize |
                                           ImGuiDockNodeFlags_NoUndocking);
            }

            // Dock windows to their positions.
            // Window IDs use the ### separator so the docking layout is
            // independent of the displayed (localised) title.  The text
            // before ### is ignored for ID purposes; only the part after
            // ### must match what the Python panel passes to ImGui::Begin.
            ImGui::DockBuilderDockWindow("###hierarchy", dockLeft);
            ImGui::DockBuilderDockWindow("###inspector", dockRight);
            ImGui::DockBuilderDockWindow("###toolbar", dockToolbar);
            ImGui::DockBuilderDockWindow("###scene_view", dockScene);
            ImGui::DockBuilderDockWindow("###game_view", dockScene);
            ImGui::DockBuilderDockWindow("###ui_editor", dockScene);
            ImGui::DockBuilderDockWindow("###console", dockBottom);
            ImGui::DockBuilderDockWindow("###project", dockBottom);

            ImGui::DockBuilderFinish(dockspaceId);

            // Ensure Scene tab is the active/selected tab after initial layout
            ImGui::SetWindowFocus("###scene_view");
        }

        ImGui::End();
    } // !m_playerMode

    using hrc = std::chrono::high_resolution_clock;
    m_lastPanelTimesMs.clear();

    // Render against a stable snapshot so Register/Unregister calls that
    // happen during panel rendering do not invalidate the active iteration.
    const auto renderableOrderSnapshot = m_renderableOrder;
    for (const auto &name : renderableOrderSnapshot) {
        auto it = m_renderables_umap.find(name);
        if (it == m_renderables_umap.end() || !it->second) {
            continue;
        }

        auto renderable = it->second;
        auto t0 = hrc::now();
        renderable->OnRender(ctx.get());
        auto t1 = hrc::now();
        m_lastPanelTimesMs[name] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    ApplyPendingDockTabSelections();
}

void InxGUI::QueueDockTabSelection(const std::string &windowId)
{
    if (windowId.empty()) {
        return;
    }
    if (std::find(m_pendingDockTabSelections.begin(), m_pendingDockTabSelections.end(), windowId) ==
        m_pendingDockTabSelections.end()) {
        m_pendingDockTabSelections.push_back(windowId);
    }
}

void InxGUI::ApplyPendingDockTabSelections()
{
    if (m_pendingDockTabSelections.empty()) {
        return;
    }

    std::vector<std::string> pending;
    pending.swap(m_pendingDockTabSelections);

    for (const auto &windowId : pending) {
        const std::string imguiName = "###" + windowId;
        ImGuiWindow *window = ImGui::FindWindowByName(imguiName.c_str());
        if (window == nullptr) {
            m_pendingDockTabSelections.push_back(windowId);
            continue;
        }

        ImGuiDockNode *dockNode = window->DockNode;
        if (dockNode != nullptr) {
            dockNode->SelectedTabId = window->TabId;
            dockNode->VisibleWindow = window;
            if (dockNode->TabBar != nullptr) {
                dockNode->TabBar->SelectedTabId = window->TabId;
                dockNode->TabBar->NextSelectedTabId = window->TabId;
                dockNode->TabBar->VisibleTabId = window->TabId;
            }
            ImGui::MarkIniSettingsDirty(window);
        }

        ImGui::FocusWindow(window);
    }
}

void InxGUI::RecordCommand(VkCommandBuffer cmdBuf)
{
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
}

void InxGUI::Shutdown()
{
    // Clean up InxGUI-owned textures (image, imageView, sampler, memory).
    // Descriptor sets allocated from m_descriptorPool_vk are freed implicitly
    // when the pool is destroyed below, so we don't free them individually.
    for (auto &[name, tex] : m_textures_umap) {
        if (tex.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_vkCore_ptr->GetDevice(), tex.sampler, nullptr);
        }
        if (tex.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_vkCore_ptr->GetDevice(), tex.imageView, nullptr);
        }
        if (tex.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vkCore_ptr->GetDeviceContext().GetVmaAllocator(), tex.image, tex.allocation);
        }
    }
    m_textures_umap.clear();

    // Shut down ImGui backends BEFORE destroying the descriptor pool —
    // ImGui_ImplVulkan_Shutdown() internally frees descriptor sets and
    // other resources that were allocated from m_descriptorPool_vk.
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();

    // Now safe to destroy the descriptor pool (all sets already freed).
    if (m_descriptorPool_vk != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_vkCore_ptr->GetDevice(), m_descriptorPool_vk, nullptr);
        m_descriptorPool_vk = VK_NULL_HANDLE;
    }

    if (m_imguiRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(m_vkCore_ptr->GetDevice(), m_imguiRenderPass, nullptr);
        m_imguiRenderPass = VK_NULL_HANDLE;
    }
}

void InxGUI::Register(const std::string &name, std::shared_ptr<InxGUIRenderable> renderable)
{
    auto existing = m_renderables_umap.find(name);
    if (existing != m_renderables_umap.end()) {
        INXLOG_WARN("InxGUI::Register(): Renderable with name '", name, "' already exists. Overwriting.");
    } else {
        // Preserve deterministic submission order for ImGui windows.
        // Dock/tab focus can become unstable when panels are submitted via
        // unordered_map iteration.
        m_renderableOrder.push_back(name);
    }
    m_renderables_umap[name] = renderable;
}

void InxGUI::Unregister(const std::string &name)
{
    auto it = m_renderables_umap.find(name);
    if (it != m_renderables_umap.end()) {
        m_renderables_umap.erase(it);
        m_renderableOrder.erase(std::remove(m_renderableOrder.begin(), m_renderableOrder.end(), name),
                                m_renderableOrder.end());
    } else {
        INXLOG_WARN("InxGUI::Unregister(): Renderable with name '", name, "' does not exist.");
    }
}

uint64_t InxGUI::UploadTextureForImGui(const std::string &name, const unsigned char *pixels, int width, int height)
{
    // Check if texture already exists
    auto it = m_textures_umap.find(name);
    if (it != m_textures_umap.end()) {
        // Remove existing texture first
        RemoveImGuiTexture(name);
    }

    VkDevice device = m_vkCore_ptr->GetDevice();
    VkPhysicalDevice physDevice = m_vkCore_ptr->GetPhysicalDevice();
    VkQueue queue = m_vkCore_ptr->GetGraphicsQueue();
    VkCommandPool cmdPool = m_vkCore_ptr->GetCommandPool();

    VkDeviceSize imageSize = width * height * 4; // RGBA

    // Create staging buffer via VMA
    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    m_vkCore_ptr->CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               stagingBuffer, stagingAllocation);

    // Copy pixel data to staging buffer
    VmaAllocator allocator = m_vkCore_ptr->GetDeviceContext().GetVmaAllocator();
    void *data;
    vmaMapMemory(allocator, stagingAllocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vmaUnmapMemory(allocator, stagingAllocation);

    // Create image
    ImGuiTextureResource tex{};

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    // Create image + allocate memory via VMA (combined create+alloc+bind)
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult result = vmaCreateImage(allocator, &imageInfo, &allocCreateInfo, &tex.image, &tex.allocation, nullptr);
    if (result != VK_SUCCESS) {
        INXLOG_ERROR("InxGUI::UploadTextureForImGui(): Failed to create image for '", name, "'");
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        return 0;
    }

    // Transition image layout and copy buffer to image
    VkCommandBuffer cmdBuf = m_vkCore_ptr->BeginSingleTimeCommands();

    // Transition to TRANSFER_DST
    VkImageMemoryBarrier barrier =
        vkrender::MakeImageBarrier(tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

    vkCmdCopyBufferToImage(cmdBuf, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ_ONLY
    barrier = vkrender::MakeImageBarrier(tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    m_vkCore_ptr->EndSingleTimeCommands(cmdBuf);

    // Clean up staging buffer
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &tex.imageView) != VK_SUCCESS) {
        INXLOG_ERROR("InxGUI::UploadTextureForImGui(): Failed to create image view for '", name, "'");
        vmaDestroyImage(allocator, tex.image, tex.allocation);
        return 0;
    }

    // Create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &tex.sampler) != VK_SUCCESS) {
        INXLOG_ERROR("InxGUI::UploadTextureForImGui(): Failed to create sampler for '", name, "'");
        vkDestroyImageView(device, tex.imageView, nullptr);
        vmaDestroyImage(allocator, tex.image, tex.allocation);
        return 0;
    }

    // Create descriptor set for ImGui
    tex.descriptorSet =
        ImGui_ImplVulkan_AddTexture(tex.sampler, tex.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (tex.descriptorSet == VK_NULL_HANDLE) {
        INXLOG_ERROR("InxGUI::UploadTextureForImGui(): Failed to create ImGui texture descriptor for '", name, "'");
        vkDestroySampler(device, tex.sampler, nullptr);
        vkDestroyImageView(device, tex.imageView, nullptr);
        vmaDestroyImage(allocator, tex.image, tex.allocation);
        return 0;
    }

    m_textures_umap[name] = tex;

    return reinterpret_cast<uint64_t>(tex.descriptorSet);
}

void InxGUI::RemoveImGuiTexture(const std::string &name)
{
    auto it = m_textures_umap.find(name);
    if (it == m_textures_umap.end()) {
        return;
    }

    VkDevice device = m_vkCore_ptr->GetDevice();
    auto &tex = it->second;

    // Wait for GPU to finish using the texture
    vkDeviceWaitIdle(device);

    if (tex.descriptorSet != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(tex.descriptorSet);
    }
    if (tex.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, tex.sampler, nullptr);
    }
    if (tex.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, tex.imageView, nullptr);
    }
    if (tex.image != VK_NULL_HANDLE) {
        VmaAllocator allocator = m_vkCore_ptr->GetDeviceContext().GetVmaAllocator();
        vmaDestroyImage(allocator, tex.image, tex.allocation);
    }

    m_textures_umap.erase(it);
}

bool InxGUI::HasImGuiTexture(const std::string &name) const
{
    return m_textures_umap.find(name) != m_textures_umap.end();
}

uint64_t InxGUI::GetImGuiTextureId(const std::string &name) const
{
    auto it = m_textures_umap.find(name);
    if (it != m_textures_umap.end()) {
        return reinterpret_cast<uint64_t>(it->second.descriptorSet);
    }
    return 0;
}

} // namespace infernux
