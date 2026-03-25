# test/fuzz/CMakeLists.txt
# Fuzz targets — included only when VOLATILE_ENABLE_FUZZING=ON.

cmake_minimum_required(VERSION 3.25)

# Determine the extra compile/link flags required by the chosen fuzzer.
if(VOLATILE_FUZZER STREQUAL "libfuzzer")
  set(_fuzz_flags -fsanitize=fuzzer,address)
elseif(VOLATILE_FUZZER STREQUAL "afl++")
  # AFL++ via afl-cc instruments at compile time; no extra flags needed here.
  # Define AFL_MAIN so the stdin entry point in each harness is compiled in.
  set(_fuzz_flags -DAFL_MAIN)
else()
  set(_fuzz_flags "")
endif()

# Helper to create one fuzz target.
function(add_fuzz_target name src)
  add_executable(${name} ${src})
  target_compile_options(${name} PRIVATE ${_fuzz_flags})
  target_link_options(${name} PRIVATE ${_fuzz_flags})
  target_include_directories(${name} PRIVATE
    "${CMAKE_SOURCE_DIR}/src/core"
  )
  target_link_libraries(${name} PRIVATE volatile_core)
endfunction()

add_fuzz_target(fuzz_json         fuzz_json.c)
add_fuzz_target(fuzz_compress4d   fuzz_compress4d.c)
