cmake_minimum_required(VERSION 3.20)
project(statecentric_gdn_state_codec LANGUAGES CXX)

# The legacy AscendC merge helper requires a non-empty build type. Without a
# default it is invoked with a missing --build-type value and fresh builds fail
# after device compilation even though the source is valid.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.1.0" CACHE PATH "CANN root")
set(SOC_VERSION "ascend910b2" CACHE STRING "AscendC target")
set(CANN_ARCH "aarch64-linux" CACHE STRING "CANN architecture")
if(NOT SOC_VERSION STREQUAL "ascend910b2")
  message(FATAL_ERROR "GDN State Codec requires ascend910b2")
endif()

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(_ascendc_candidates
    "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_HOME_PATH}/${CANN_ARCH}/tikcpp/ascendc_kernel_cmake")
foreach(_candidate IN LISTS _ascendc_candidates)
  if(EXISTS "${_candidate}/ascendc.cmake")
    set(ASCENDC_CMAKE_DIR "${_candidate}")
    break()
  endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
  message(FATAL_ERROR "ascendc.cmake not found")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

ascendc_library(statecentric_gdn_state_codec_smoothing_kernel SHARED
                kernel/gdn_state_codec_smoothing.cpp)
ascendc_library(statecentric_gdn_state_codec_quantize_kernel SHARED
                kernel/gdn_state_codec_quantize.cpp)
ascendc_library(statecentric_gdn_state_codec_dequantize_kernel SHARED
                kernel/gdn_state_codec_dequantize.cpp)
ascendc_library(statecentric_gdn_state_codec_windowed_gdn_step_kernel SHARED
                kernel/gdn_state_codec_windowed_gdn_step.cpp)
ascendc_library(statecentric_gdn_state_codec_fp32_gdn_step_kernel SHARED
                kernel/gdn_state_codec_fp32_gdn_step.cpp)
ascendc_library(statecentric_gdn_state_codec_window_boundary_kernel SHARED
                kernel/gdn_state_codec_window_boundary.cpp)
ascendc_library(statecentric_gdn_state_codec_cube_matmul_probe_kernel SHARED
                kernel/gdn_state_codec_cube_matmul_probe.cpp)
ascendc_library(statecentric_gdn_state_codec_cube_normalize_kernel SHARED
                kernel/gdn_state_codec_cube_normalize.cpp)
ascendc_library(statecentric_gdn_state_codec_cube_deflation_kernel SHARED
                kernel/gdn_state_codec_cube_deflation.cpp)
ascendc_library(statecentric_gdn_state_codec_int8_cube_matmul_kernel SHARED
                kernel/gdn_state_codec_int8_cube_matmul.cpp)
ascendc_library(statecentric_gdn_state_codec_bf16_cube_matmul_kernel SHARED
                kernel/gdn_state_codec_bf16_cube_matmul.cpp)
ascendc_library(statecentric_gdn_state_codec_encoded_projection_kernel SHARED
                kernel/gdn_state_codec_encoded_projection.cpp)
ascendc_library(statecentric_gdn_state_codec_encoded_requantize_kernel SHARED
                kernel/gdn_state_codec_encoded_requantize.cpp)
ascendc_compile_definitions(
  statecentric_gdn_state_codec_cube_matmul_probe_kernel PRIVATE
  HAVE_WORKSPACE HAVE_TILING)
ascendc_compile_definitions(
  statecentric_gdn_state_codec_int8_cube_matmul_kernel PRIVATE
  HAVE_WORKSPACE HAVE_TILING)
ascendc_compile_definitions(
  statecentric_gdn_state_codec_bf16_cube_matmul_kernel PRIVATE
  HAVE_WORKSPACE HAVE_TILING)
foreach(_target IN ITEMS statecentric_gdn_state_codec_smoothing_kernel
                         statecentric_gdn_state_codec_quantize_kernel
                         statecentric_gdn_state_codec_dequantize_kernel
                         statecentric_gdn_state_codec_windowed_gdn_step_kernel
                         statecentric_gdn_state_codec_fp32_gdn_step_kernel
                         statecentric_gdn_state_codec_window_boundary_kernel
                         statecentric_gdn_state_codec_cube_matmul_probe_kernel
                         statecentric_gdn_state_codec_cube_normalize_kernel
                         statecentric_gdn_state_codec_cube_deflation_kernel
                         statecentric_gdn_state_codec_int8_cube_matmul_kernel
                         statecentric_gdn_state_codec_bf16_cube_matmul_kernel
                         statecentric_gdn_state_codec_encoded_projection_kernel
                         statecentric_gdn_state_codec_encoded_requantize_kernel)
  ascendc_include_directories(
    ${_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
endforeach()

find_path(ASCENDCL_INCLUDE_DIR NAMES acl/acl.h
          PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/include"
          NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDCL_LIBRARY NAMES ascendcl
             PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
             NO_DEFAULT_PATH REQUIRED)

add_library(statecentric_gdn_state_codec_provider_runtime SHARED
            host/gdn_state_codec_provider_runtime.cpp)
target_include_directories(
  statecentric_gdn_state_codec_provider_runtime
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(
  statecentric_gdn_state_codec_provider_runtime
  PRIVATE statecentric_gdn_state_codec_smoothing_kernel
          statecentric_gdn_state_codec_quantize_kernel)
target_compile_options(
  statecentric_gdn_state_codec_provider_runtime
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_provider_runtime
  PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

add_executable(statecentric_gdn_state_codec_runtime_probe
               host/gdn_state_codec_runtime_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_runtime_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_runtime_probe
  SYSTEM PRIVATE "${ASCENDCL_INCLUDE_DIR}")
target_link_libraries(
  statecentric_gdn_state_codec_runtime_probe
  PRIVATE statecentric_gdn_state_codec_smoothing_kernel
          statecentric_gdn_state_codec_quantize_kernel
          statecentric_gdn_state_codec_dequantize_kernel
          "${ASCENDCL_LIBRARY}")
target_compile_options(
  statecentric_gdn_state_codec_runtime_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_runtime_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_provider_quantize_probe
               host/gdn_state_codec_provider_quantize_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_provider_quantize_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_provider_quantize_probe
  SYSTEM PRIVATE "${ASCENDCL_INCLUDE_DIR}")
target_link_libraries(
  statecentric_gdn_state_codec_provider_quantize_probe
  PRIVATE statecentric_gdn_state_codec_smoothing_kernel
          statecentric_gdn_state_codec_quantize_kernel
          statecentric_gdn_state_codec_provider_runtime
          "${ASCENDCL_LIBRARY}")
target_compile_options(
  statecentric_gdn_state_codec_provider_quantize_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_provider_quantize_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_windowed_gdn_runtime_probe
               host/gdn_state_codec_windowed_gdn_runtime_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_windowed_gdn_runtime_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_windowed_gdn_runtime_probe
  SYSTEM PRIVATE "${ASCENDCL_INCLUDE_DIR}")
target_link_libraries(
  statecentric_gdn_state_codec_windowed_gdn_runtime_probe
  PRIVATE statecentric_gdn_state_codec_windowed_gdn_step_kernel
          statecentric_gdn_state_codec_fp32_gdn_step_kernel
          "${ASCENDCL_LIBRARY}")
target_compile_options(
  statecentric_gdn_state_codec_windowed_gdn_runtime_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_windowed_gdn_runtime_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_window_boundary_runtime_probe
               host/gdn_state_codec_window_boundary_runtime_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_window_boundary_runtime_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_window_boundary_runtime_probe
  SYSTEM PRIVATE "${ASCENDCL_INCLUDE_DIR}")
target_link_libraries(
  statecentric_gdn_state_codec_window_boundary_runtime_probe
  PRIVATE statecentric_gdn_state_codec_window_boundary_kernel "${ASCENDCL_LIBRARY}")
target_compile_options(
  statecentric_gdn_state_codec_window_boundary_runtime_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_window_boundary_runtime_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

find_library(TILING_API_LIBRARY NAMES tiling_api
             PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(REGISTER_LIBRARY NAMES register
             PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(PLATFORM_LIBRARY NAMES platform
             PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDALOG_LIBRARY NAMES ascendalog
             PATHS "${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
             NO_DEFAULT_PATH REQUIRED)

add_executable(statecentric_gdn_state_codec_cube_matmul_probe
               host/gdn_state_codec_cube_matmul_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_cube_matmul_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_cube_matmul_probe SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_cube_matmul_probe
  PRIVATE statecentric_gdn_state_codec_cube_matmul_probe_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_cube_matmul_probe
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_cube_matmul_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_cube_matmul_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_cube_power_iteration_probe
               host/gdn_state_codec_cube_power_iteration_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_cube_power_iteration_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_cube_power_iteration_probe SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_cube_power_iteration_probe
  PRIVATE statecentric_gdn_state_codec_cube_matmul_probe_kernel
          statecentric_gdn_state_codec_cube_normalize_kernel
          statecentric_gdn_state_codec_cube_deflation_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_cube_power_iteration_probe
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_cube_power_iteration_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_cube_power_iteration_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_int8_cube_matmul_probe
               host/gdn_state_codec_int8_cube_matmul_probe.cpp)
target_include_directories(
  statecentric_gdn_state_codec_int8_cube_matmul_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_int8_cube_matmul_probe SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_int8_cube_matmul_probe
  PRIVATE statecentric_gdn_state_codec_int8_cube_matmul_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_int8_cube_matmul_probe
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_int8_cube_matmul_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_int8_cube_matmul_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")

add_executable(statecentric_gdn_state_codec_encoded_projection_probe
               host/gdn_state_codec_encoded_projection_probe.cpp)
add_library(statecentric_gdn_state_codec_decode_runtime SHARED
            host/gdn_state_codec_decode_runtime.cpp)
add_library(statecentric_gdn_state_codec_boundary_runtime SHARED
            host/gdn_state_codec_boundary_runtime.cpp)
target_include_directories(
  statecentric_gdn_state_codec_decode_runtime
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_decode_runtime SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_decode_runtime
  PRIVATE statecentric_gdn_state_codec_int8_cube_matmul_kernel
          statecentric_gdn_state_codec_encoded_projection_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_decode_runtime
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_decode_runtime
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_decode_runtime
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
             BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")
target_include_directories(
  statecentric_gdn_state_codec_boundary_runtime
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_boundary_runtime SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_boundary_runtime
  PRIVATE statecentric_gdn_state_codec_window_boundary_kernel
          statecentric_gdn_state_codec_bf16_cube_matmul_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_boundary_runtime
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_boundary_runtime
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_boundary_runtime
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
             BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")
target_include_directories(
  statecentric_gdn_state_codec_encoded_projection_probe
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(
  statecentric_gdn_state_codec_encoded_projection_probe SYSTEM PRIVATE
  "${ASCENDCL_INCLUDE_DIR}"
  "${ASCEND_HOME_PATH}/${CANN_ARCH}/asc/include")
target_link_libraries(
  statecentric_gdn_state_codec_encoded_projection_probe
  PRIVATE statecentric_gdn_state_codec_int8_cube_matmul_kernel
          statecentric_gdn_state_codec_encoded_projection_kernel
          statecentric_gdn_state_codec_encoded_requantize_kernel
          statecentric_gdn_state_codec_windowed_gdn_step_kernel
          statecentric_gdn_state_codec_fp32_gdn_step_kernel
          "${ASCENDCL_LIBRARY}" "${TILING_API_LIBRARY}"
          "${REGISTER_LIBRARY}" "${PLATFORM_LIBRARY}"
          "${ASCENDALOG_LIBRARY}" "${CMAKE_DL_LIBS}")
target_compile_definitions(
  statecentric_gdn_state_codec_encoded_projection_probe
  PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(
  statecentric_gdn_state_codec_encoded_projection_probe
  PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(
  statecentric_gdn_state_codec_encoded_projection_probe
  PROPERTIES BUILD_RPATH
             "${CMAKE_CURRENT_BINARY_DIR}/lib;${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64")
