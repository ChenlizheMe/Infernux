#pragma once

#include <vector>

namespace infernux
{
// Fixed-size storage shared by native resources and exported buffer views.
struct Float32Array
{
    std::vector<float> values;
};
} // namespace infernux
