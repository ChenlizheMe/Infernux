if(INFERNUX_BUILD_TESTS)
    add_executable(infernux_scene_authoring_identity_tests
        tests/native/SceneAuthoringIdentityTests.cpp
        cpp/infernux/function/scene/SceneAuthoringIdentity.cpp
        cpp/infernux/core/types/Guid.cpp
    )
    target_include_directories(infernux_scene_authoring_identity_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/infernux
        ${CMAKE_SOURCE_DIR}/external
    )
    target_compile_features(infernux_scene_authoring_identity_tests PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(infernux_scene_authoring_identity_tests PRIVATE /utf-8)
    endif()
    add_test(NAME infernux.scene_authoring_identity COMMAND infernux_scene_authoring_identity_tests)
    set_tests_properties(infernux.scene_authoring_identity PROPERTIES TIMEOUT 30)
endif()
