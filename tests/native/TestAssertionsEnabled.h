#pragma once

// C assert checks and their setup expressions must execute in optimized tests.
#ifdef NDEBUG
#error Native regression tests require assertions in every build configuration.
#endif
