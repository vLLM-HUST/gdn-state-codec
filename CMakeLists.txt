cmake_minimum_required(VERSION 3.20)
project(gdn_state_codec LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gdn_state_codec_reference STATIC
  native/src/sha256.cpp
  native/src/gdn_state_codec_state_capsule.cpp
  native/src/gdn_state_codec_qwen38_gdn_adapter.cpp)
target_include_directories(gdn_state_codec_reference
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/native/include")
target_compile_options(gdn_state_codec_reference
  PRIVATE -Wall -Wextra -Wpedantic -Werror)

foreach(test_name IN ITEMS sha256 gdn_state_codec_state_capsule
                           gdn_state_codec_qwen38_gdn_adapter)
  add_executable(${test_name}_test native/tests/${test_name}_test.cpp)
  target_link_libraries(${test_name}_test PRIVATE gdn_state_codec_reference)
  target_compile_options(${test_name}_test
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
endforeach()
