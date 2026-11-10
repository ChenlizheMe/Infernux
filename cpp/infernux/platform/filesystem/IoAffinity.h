#pragma once

#include <core/threading/JobSystem.h>

namespace infernux::detail
{
// Single-thread Web reads its cooked virtual filesystem through owner jobs.
// Native and threaded Web builds keep IO outside owner-thread help budgets.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
inline constexpr JobAffinity IoJobAffinity = JobAffinity::AnyThread;
#else
inline constexpr JobAffinity IoJobAffinity = JobAffinity::WorkerOnly;
#endif
} // namespace infernux::detail
