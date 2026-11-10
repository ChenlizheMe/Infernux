#include "Random.h"

namespace infernux
{
Random::Storage &Random::Shared()
{
    static Storage storage;
    return storage;
}
} // namespace infernux
