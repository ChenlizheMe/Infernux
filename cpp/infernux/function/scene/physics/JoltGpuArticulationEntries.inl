// Actual precise SPIR-V/WGSL resource closure; regenerated only offline.
struct NumericEntryDefinition
{
    const char *name;
    uint8_t source;
    const char *resources;
};
inline constexpr NumericEntryDefinition NumericEntries[] = {
    {"normal_capture_initialize", 1,
     "normal_resident_coordinates|normal_resident_roots|sf_coordinates|sf_roots|sf_work"},
    {"normal_capture_begin", 1, "normal_capture_control|sf_coordinates|sf_links|sf_roots|sf_trees|sf_work"},
    {"normal_capture_advance", 1,
     "normal_capture_control|sf_accepted|sf_coordinates|sf_joints|sf_links|sf_models|sf_roots|sf_trees|sf_work"},
    {"normal_seed_advance", 1,
     "normal_active_coordinates|normal_active_roots|normal_capture_control|sf_coordinates|sf_roots|sf_trees|sf_work"},
    {"normal_promote_begin", 1,
     "normal_capture_control|normal_dynamics_trees|sf_coordinates|sf_links|sf_roots|sf_trees|sf_work"},
    {"normal_promote_advance", 1,
     "normal_capture_control|normal_resident_coordinates|normal_resident_roots|sf_coordinates|sf_roots|sf_trees|sf_"
     "work"},
    {"normal_dynamics_begin", 2, "art_links|art_scratch|art_trees|art_work|dyn_coordinates|dyn_roots"},
    {"normal_dynamics_preserve", 2,
     "art_joints|art_links|art_models|art_scratch|art_states|art_trees|art_work|dyn_coordinates|dyn_roots"},
    {"normal_dynamics_solve_begin", 2, "art_scratch|art_trees|art_work"},
    {"normal_dynamics_solve", 2,
     "art_joints|art_links|art_models|art_scratch|art_states|art_trees|art_work|dyn_coordinates|dyn_roots"},
    {"normal_reaction_begin", 2, "art_scratch|art_trees|art_work"},
    {"normal_reaction_advance", 2, "art_joints|art_links|art_models|art_scratch|art_states|art_trees|art_work"},
    {"normal_history_begin", 2, "art_scratch|art_trees|art_work"},
    {"normal_history_advance", 2, "art_joints|art_links|art_scratch|art_trees|art_work"},
    {"normal_velocity_begin", 2, "art_scratch|art_trees|art_work"},
    {"normal_velocity_advance", 2,
     "art_joints|art_links|art_models|art_scratch|art_states|art_trees|art_work|dyn_coordinates|dyn_roots"},
    {"normal_pose_begin", 2, "art_scratch|art_trees|art_work"},
    {"normal_pose_advance", 2,
     "art_joints|art_links|art_models|art_scratch|art_states|art_trees|art_work|dyn_coordinates|dyn_roots"},
    {"articulation_dirty", 0, "articulation_params|articulation_words"},
    {"articulation_initialize_numeric", 0, "articulation_params|articulation_words"},
    {"articulation_mark_candidate", 0, "articulation_params|articulation_trees|articulation_words"},
    {"articulation_admit", 0, "articulation_params|articulation_trees|articulation_words"},
    {"articulation_capture_transition", 0,
     "articulation_params|articulation_previous_control|articulation_transition_witness|articulation_transitions|joint_"
     "stream"},
    {"articulation_reset_transition", 0,
     "articulation_params|articulation_summary|articulation_transition_witness|articulation_transitions|articulation_"
     "words|joint_stream"},
    {"articulation_clear_summary", 0, "articulation_summary"},
    {"articulation_numeric_summary", 0,
     "articulation_capture_trees|articulation_params|articulation_summary|articulation_trees|articulation_words"},
    {"articulation_retire_declined", 0, "articulation_retirement|joint_stream"},
    {"articulation_retire_captured", 0, "articulation_retirement|articulation_transition_witness|joint_stream"},
    {"articulation_clear_retirement_summary", 0, "articulation_retirement"},
    {"articulation_advance_retirement", 0, "articulation_retirement"},
    {"articulation_retirement_summary", 0, "articulation_words"},
};
