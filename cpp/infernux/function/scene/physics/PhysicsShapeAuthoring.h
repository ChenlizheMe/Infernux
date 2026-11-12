#pragma once

#include <cstddef>
#include <vector>

namespace JPH
{
class Shape;
template <class T> class RefConst;
} // namespace JPH

namespace infernux
{
class Collider;
struct PhysicsActorData;

/// Owner-thread authoring: capture all active collider geometry as one shape.
/// Null means the whole candidate is not ready; pending children still begin
/// cooking. Returned children already contain world scale, so relative actor
/// transforms contribute only rigid translation and rotation.
JPH::RefConst<JPH::Shape> BuildActorShape(const PhysicsActorData &actor, const Collider *exclude = nullptr,
                                          size_t *outShapeCount = nullptr,
                                          std::vector<Collider *> *outMembers = nullptr);
} // namespace infernux
