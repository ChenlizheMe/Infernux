#pragma once

/**
 * @file ProfileConfig.h
 * @brief Compile-time switch for frame profiling instrumentation.
 *
 * Set INFERNUX_FRAME_PROFILE to 1 to enable per-frame timing measurements
 * across the renderer (DrawFrame, DrawSceneFiltered, DrawShadowCasters,
 * RenderGraph::Execute, SceneRenderer prepare/build, ScriptableRenderContext,
 * executor sub-timing, and selected scene-side helpers.
 *
 * Set to 0 (default) to compile out all profiling overhead.
 */

#ifndef INFERNUX_FRAME_PROFILE
#define INFERNUX_FRAME_PROFILE 0
#endif

#ifndef INFERNUX_FRAME_PROFILE_TERMINAL
#define INFERNUX_FRAME_PROFILE_TERMINAL 0
#endif

#ifndef INFERNUX_FRAME_PROFILE_WINDOW
#define INFERNUX_FRAME_PROFILE_WINDOW 120
#endif

#ifndef INFERNUX_FRAME_PROFILE_DETAIL
#define INFERNUX_FRAME_PROFILE_DETAIL 1
#endif
