#include "ShaderPropertyId.h"

namespace infernux
{
ShaderPropertyIds::Names &ShaderPropertyIds::Registry()
{
    static Names names;
    return names;
}
} // namespace infernux
