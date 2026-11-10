"""Editable skeleton hierarchy shared by animation, scripts and physics."""
from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.core.asset_ref import GenericAssetRef


class SkeletonInstance(BuiltinComponent):
    _cpp_type_name = "SkeletonInstance"
    _component_category_ = "Animation"

    model = CppProperty.from_native("SkeletonInstance", "model",
        get_converter=lambda guid: GenericAssetRef("Mesh", guid=guid) if guid else None,
        set_converter=lambda value: value.guid if value else "")

    @property
    def root(self):
        return self._require_cpp_component().root

    @property
    def bones(self):
        """All hierarchy nodes, including helpers; renderer.bones is the skin subset."""
        return self._require_cpp_component().bones

    @property
    def bone_names(self):
        return self._require_cpp_component().bone_names

    @property
    def parent_indices(self):
        return self._require_cpp_component().parent_indices

    def get_bone_transform(self, name):
        return self._require_cpp_component().get_bone_transform(name)

    def get_human_bone_transform(self, slot):
        return self._require_cpp_component().get_human_bone_transform(slot)

    def get_bind_local_matrices(self):
        return self._require_cpp_component().get_bind_local_matrices()

    def get_world_matrices(self):
        return self._require_cpp_component().get_world_matrices()

    @property
    def physics_controls_pose(self):
        return self._require_cpp_component().physics_controls_pose

    @property
    def has_animation_pose(self):
        return self._require_cpp_component().has_animation_pose

    def capture_animation_pose(self):
        """Use the current bone locals as a target, without moving physics."""
        self._require_cpp_component().capture_animation_pose()

    def apply_animation_pose(self):
        self._require_cpp_component().apply_animation_pose()

    def get_animation_world_matrices(self):
        """Complete animation target; physical bone presentation stays independent."""
        return self._require_cpp_component().get_animation_world_matrices()

    def get_local_matrices(self):
        return self._require_cpp_component().get_local_matrices()

    def get_root_space_matrices(self):
        return self._require_cpp_component().get_root_space_matrices()

    def rebuild_hierarchy(self):
        self._require_cpp_component().rebuild_hierarchy()

    def reset_to_bind_pose(self):
        self._require_cpp_component().reset_to_bind_pose()
