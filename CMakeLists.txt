cmake_minimum_required(VERSION 3.20)
project(stateaxis_leapquant LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(stateaxis_leapquant_reference STATIC
  native/src/sha256.cpp
  native/src/leapquant_state_capsule.cpp
  native/src/leapquant_qwen38_gdn_adapter.cpp)
target_include_directories(stateaxis_leapquant_reference
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/native/include")
target_compile_options(stateaxis_leapquant_reference
  PRIVATE -Wall -Wextra -Wpedantic -Werror)

foreach(test_name IN ITEMS sha256 leapquant_state_capsule
                           leapquant_qwen38_gdn_adapter)
  add_executable(${test_name}_test native/tests/${test_name}_test.cpp)
  target_link_libraries(${test_name}_test PRIVATE stateaxis_leapquant_reference)
  target_compile_options(${test_name}_test
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
endforeach()
