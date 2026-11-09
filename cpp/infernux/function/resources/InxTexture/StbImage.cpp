#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdlib>

namespace
{
struct ResizeAllocation
{
    size_t maximum;
    size_t requested = 0;
};

thread_local ResizeAllocation *s_resizeAllocation = nullptr;

void *AllocateResize(size_t size)
{
    if (s_resizeAllocation) {
        s_resizeAllocation->requested = size;
        if (size > s_resizeAllocation->maximum)
            return nullptr;
    }
    return std::malloc(size);
}
} // namespace

#define STBIR_MALLOC(size, user) ((void)(user), AllocateResize(size))
#define STBIR_FREE(pointer, user) ((void)(user), std::free(pointer))
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace infernux::detail
{
size_t MeasureTextureResize(STBIR_RESIZE &resize)
{
    ResizeAllocation allocation{0};
    s_resizeAllocation = &allocation;
    (void)stbir_build_samplers(&resize);
    s_resizeAllocation = nullptr;
    return allocation.requested;
}

bool RunTextureResize(STBIR_RESIZE &resize, size_t maximum)
{
    ResizeAllocation allocation{maximum};
    s_resizeAllocation = &allocation;
    const bool success = stbir_resize_extended(&resize) != 0;
    s_resizeAllocation = nullptr;
    return success;
}
} // namespace infernux::detail
