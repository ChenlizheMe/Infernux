#pragma once

namespace infernux
{

/// A joint unregisters its constraint before its own destruction. The world
/// also retires constraints when either body dies or the physics world shuts
/// down; the owning joint must observe that retirement at the same boundary.
class PhysicsConstraintOwner
{
  public:
    virtual void OnPhysicsConstraintDestroyed() noexcept = 0;

  protected:
    ~PhysicsConstraintOwner() = default;
};

} // namespace infernux
