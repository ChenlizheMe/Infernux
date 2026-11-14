include("${CMAKE_CURRENT_LIST_DIR}/stable_copy.cmake")

# The private Runtime archive must not appear in a staged native payload.
file(REMOVE "${TARGET_DIR}/libInfernuxRuntime.so")
# Standard-module IR now belongs to the compiler, not the SDK directory.
file(GLOB _retired_standard_modules "${TARGET_DIR}/slang-standard-module-*")
foreach(_retired IN LISTS _retired_standard_modules)
    file(REMOVE_RECURSE "${_retired}")
endforeach()
list(REMOVE_DUPLICATES DEPENDENCY_FILES)
foreach(_source IN LISTS DEPENDENCY_FILES)
    get_filename_component(_name "${_source}" NAME)
    infernux_copy_file_stable("${_source}" "${TARGET_DIR}/${_name}")
endforeach()
