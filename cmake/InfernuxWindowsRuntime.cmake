# A private DLL linked through an object library need not appear in
# TARGET_RUNTIME_DLLS. Read the finished PE's imports and source each DLL from
# the exact producer for this configuration, including nested external projects.
function(infernux_collect_dll_outputs directory result)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    set(outputs)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(type STREQUAL "SHARED_LIBRARY")
            list(APPEND outputs "$<TARGET_FILE:${target}>")
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        infernux_collect_dll_outputs("${child}" nested)
        list(APPEND outputs ${nested})
    endforeach()
    set(${result} "${outputs}" PARENT_SCOPE)
endfunction()

get_filename_component(_infernux_linker_dir "${CMAKE_LINKER}" DIRECTORY)
find_program(INFERNUX_PE_INSPECTOR dumpbin HINTS "${_infernux_linker_dir}" REQUIRED)
infernux_collect_dll_outputs("${CMAKE_SOURCE_DIR}" _infernux_dll_outputs)
list(APPEND _infernux_dll_outputs ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
set(_infernux_dll_catalog "${CMAKE_BINARY_DIR}/windows-runtime/$<CONFIG>/targets.txt")
file(GENERATE OUTPUT "${_infernux_dll_catalog}" CONTENT "$<JOIN:${_infernux_dll_outputs},\n>\n")
set(_infernux_dll_resolver "${Python3_EXECUTABLE}" -I -X utf8
    "${CMAKE_CURRENT_LIST_DIR}/windows_runtime_dlls.py" "${INFERNUX_PE_INSPECTOR}"
    --targets "${_infernux_dll_catalog}" "$<$<CONFIG:Debug>:--developer-crt>")
set(_infernux_wheel_pe_roots "$<TARGET_FILE:_Infernux>" "$<TARGET_FILE:_InfernuxBootstrap>")
if(NOT INFERNUX_USE_TARGET_PYTHON)
    list(APPEND _infernux_wheel_pe_roots "$<TARGET_FILE:InfernuxEditorProgress>")
endif()
