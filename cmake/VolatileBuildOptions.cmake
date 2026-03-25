# =============================================================================
# VolatileBuildOptions.cmake
# Advanced build options for the volatile project.
# Included from the top-level CMakeLists.txt immediately after project().
# =============================================================================

include_guard(GLOBAL)

# Detect compiler family once so all blocks below can branch on it.
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(VOLATILE_COMPILER_CLANG TRUE)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(VOLATILE_COMPILER_GCC TRUE)
endif()

# =============================================================================
# Build Acceleration
# =============================================================================

# ---------------------------------------------------------------------------
# ccache — detect and use automatically when available
# ---------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# ---------------------------------------------------------------------------
# Unity / Jumbo build — merge multiple translation units into one per batch,
# reducing redundant parsing of headers and improving dead-code elimination.
# ---------------------------------------------------------------------------
option(VOLATILE_UNITY_BUILD
  "Enable Unity/Jumbo build (fewer TUs, faster build)" OFF)
if(VOLATILE_UNITY_BUILD)
  set(CMAKE_UNITY_BUILD ON)
  set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)
  message(STATUS "Unity build enabled (batch size 16)")
endif()

# ---------------------------------------------------------------------------
# Precompiled headers — amortise parsing of heavyweight third-party headers.
# Applied to core, gui, and render targets in their own CMakeLists.txt via:
#   if(VOLATILE_PCH) target_precompile_headers(... REUSE_FROM volatile_pch) endif()
# ---------------------------------------------------------------------------
option(VOLATILE_PCH
  "Enable precompiled headers for SDL3/Vulkan/Nuklear" OFF)
if(VOLATILE_PCH)
  # Create a tiny interface library that owns the PCH so all targets can
  # REUSE_FROM it without recompiling the headers multiple times.
  add_library(volatile_pch INTERFACE)
  target_precompile_headers(volatile_pch INTERFACE
    <SDL3/SDL.h>
    <vulkan/vulkan.h>
  )
  message(STATUS "Precompiled headers enabled")
endif()

# =============================================================================
# Optimization Profiles
# =============================================================================

# ---------------------------------------------------------------------------
# Native tuning — emit instructions for the host CPU micro-architecture.
# Binaries built with this flag are NOT portable to other machines.
# ---------------------------------------------------------------------------
option(VOLATILE_OPT_NATIVE
  "Use -march=native for host-specific optimization" ON)
if(VOLATILE_OPT_NATIVE)
  add_compile_options(-march=native)
  message(STATUS "Native march enabled")
endif()

# ---------------------------------------------------------------------------
# Fast math — allows reordering of floating-point ops and disables strict
# IEEE 754 compliance.  Can break code that relies on NaN/Inf propagation.
# ---------------------------------------------------------------------------
option(VOLATILE_OPT_FAST_MATH "Enable fast-math (-ffast-math)" OFF)
if(VOLATILE_OPT_FAST_MATH)
  add_compile_options(-ffast-math)
endif()

# ---------------------------------------------------------------------------
# Full LTO — whole-program optimisation at link time.
# Significantly increases link time but can improve runtime 5-15%.
# Use ThinLTO for faster incremental LTO with Clang.
# ---------------------------------------------------------------------------
option(VOLATILE_OPT_LTO "Enable Link-Time Optimization" OFF)
option(VOLATILE_OPT_THINLTO
  "Enable ThinLTO (clang only, faster than full LTO)" OFF)

# ThinLTO cache — reuses work across incremental builds.
set(VOLATILE_THINLTO_CACHE_DIR "${CMAKE_BINARY_DIR}/thinlto-cache"
  CACHE PATH "ThinLTO cache directory")

if(VOLATILE_OPT_THINLTO AND VOLATILE_COMPILER_CLANG)
  add_compile_options(-flto=thin)
  add_link_options(
    -flto=thin
    -Wl,--thinlto-cache-dir=${VOLATILE_THINLTO_CACHE_DIR}
  )
  message(STATUS "ThinLTO enabled (cache: ${VOLATILE_THINLTO_CACHE_DIR})")
elseif(VOLATILE_OPT_LTO)
  # GCC and Clang both accept -flto for full LTO.
  add_compile_options(-flto)
  add_link_options(-flto)
  if(VOLATILE_COMPILER_GCC)
    # Parallelise the LTO link step using all available cores.
    cmake_host_system_information(RESULT _nproc QUERY NUMBER_OF_LOGICAL_CORES)
    add_link_options(-flto=${_nproc})
  endif()
  message(STATUS "Full LTO enabled")
endif()

# ---------------------------------------------------------------------------
# PGO — Profile-Guided Optimization.
# Two-phase workflow:
#   1. Build with VOLATILE_OPT_PGO_GENERATE=ON, run representative workloads.
#   2. Rebuild with VOLATILE_OPT_PGO_USE=ON to consume the recorded profiles.
# See scripts/pgo.sh for the automated workflow.
# ---------------------------------------------------------------------------
set(VOLATILE_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo-profiles"
  CACHE PATH "PGO profile directory")

option(VOLATILE_OPT_PGO_GENERATE "Generate PGO profile data" OFF)
option(VOLATILE_OPT_PGO_USE      "Use PGO profile data"      OFF)

if(VOLATILE_OPT_PGO_GENERATE AND VOLATILE_OPT_PGO_USE)
  message(FATAL_ERROR
    "VOLATILE_OPT_PGO_GENERATE and VOLATILE_OPT_PGO_USE are mutually exclusive")
endif()

if(VOLATILE_OPT_PGO_GENERATE)
  file(MAKE_DIRECTORY "${VOLATILE_PGO_PROFILE_DIR}")
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(-fprofile-generate=${VOLATILE_PGO_PROFILE_DIR})
    add_link_options(-fprofile-generate=${VOLATILE_PGO_PROFILE_DIR})
  else()
    add_compile_options(-fprofile-generate=${VOLATILE_PGO_PROFILE_DIR})
    add_link_options(-fprofile-generate=${VOLATILE_PGO_PROFILE_DIR})
  endif()
  message(STATUS "PGO: generating profiles to ${VOLATILE_PGO_PROFILE_DIR}")
endif()

if(VOLATILE_OPT_PGO_USE)
  if(VOLATILE_COMPILER_CLANG)
    # Clang expects a merged .profdata file.
    set(_pgo_file "${VOLATILE_PGO_PROFILE_DIR}/default.profdata")
    add_compile_options(-fprofile-use=${_pgo_file})
    add_link_options(-fprofile-use=${_pgo_file})
  else()
    add_compile_options(-fprofile-use=${VOLATILE_PGO_PROFILE_DIR})
    add_link_options(-fprofile-use=${VOLATILE_PGO_PROFILE_DIR})
  endif()
  message(STATUS "PGO: using profiles from ${VOLATILE_PGO_PROFILE_DIR}")
endif()

# ---------------------------------------------------------------------------
# BOLT — post-link binary optimiser (requires llvm-bolt in PATH).
# After building, run:  cmake --build . --target bolt-optimize
# ---------------------------------------------------------------------------
option(VOLATILE_OPT_BOLT
  "Post-link optimize with BOLT (requires llvm-bolt)" OFF)
if(VOLATILE_OPT_BOLT)
  find_program(LLVM_BOLT_PROGRAM llvm-bolt)
  find_program(PERF2BOLT_PROGRAM perf2bolt)
  if(NOT LLVM_BOLT_PROGRAM)
    message(WARNING "VOLATILE_OPT_BOLT requested but llvm-bolt not found in PATH")
  else()
    # The custom target is defined late (after subdirectories add executables)
    # via a deferred call so that the volatile_server target exists.
    cmake_language(DEFER CALL _volatile_add_bolt_target)
  endif()
endif()

function(_volatile_add_bolt_target)
  # Instrument the main server binary; adjust target name as needed.
  if(TARGET volatile_server)
    set(_bin "$<TARGET_FILE:volatile_server>")
    add_custom_target(bolt-optimize
      COMMENT "Applying BOLT optimizations to volatile_server"
      COMMAND ${LLVM_BOLT_PROGRAM}
        ${_bin}
        --data "${CMAKE_BINARY_DIR}/bolt.fdata"
        --dyno-stats
        --eliminate-unreachable
        --frame-opt=all
        --icf=1
        --indirect-call-promotion=all
        --jump-tables=aggressive
        --plt=all
        --reorder-blocks=ext-tsp
        --reorder-functions=hfsort+
        --split-all-cold
        --split-eh
        --use-gnu-stack
        -o "${_bin}.bolt"
      VERBATIM
    )
  endif()
endfunction()

# =============================================================================
# Sanitizers (expanded)
# =============================================================================

# Sanitizers are mutually exclusive with each other in most combinations.
# Enable at most one at a time; ASAN+UBSAN is a common safe pairing.

option(VOLATILE_ENABLE_ASAN  "AddressSanitizer"                    OFF)
option(VOLATILE_ENABLE_UBSAN "UndefinedBehaviorSanitizer"           OFF)
option(VOLATILE_ENABLE_TSAN  "ThreadSanitizer"                     OFF)
option(VOLATILE_ENABLE_LSAN  "LeakSanitizer (standalone)"          OFF)
option(VOLATILE_ENABLE_MSAN  "MemorySanitizer (clang only)"        OFF)

if(VOLATILE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
  message(STATUS "Sanitizer: AddressSanitizer enabled")
endif()

if(VOLATILE_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
  message(STATUS "Sanitizer: UndefinedBehaviorSanitizer enabled")
endif()

if(VOLATILE_ENABLE_TSAN)
  if(VOLATILE_ENABLE_ASAN)
    message(FATAL_ERROR "TSan and ASan cannot be used together")
  endif()
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
  message(STATUS "Sanitizer: ThreadSanitizer enabled")
endif()

if(VOLATILE_ENABLE_LSAN)
  add_compile_options(-fsanitize=leak)
  add_link_options(-fsanitize=leak)
  message(STATUS "Sanitizer: LeakSanitizer enabled")
endif()

if(VOLATILE_ENABLE_MSAN)
  if(NOT VOLATILE_COMPILER_CLANG)
    message(FATAL_ERROR "MemorySanitizer requires Clang")
  endif()
  add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
  add_link_options(-fsanitize=memory)
  message(STATUS "Sanitizer: MemorySanitizer enabled")
endif()

# ---------------------------------------------------------------------------
# Fuzzing — build fuzz harnesses under test/fuzz/.
# AFL++ mode: set CC=afl-cc before configuring.
# libFuzzer mode: requires Clang; adds -fsanitize=fuzzer to fuzz targets only.
# ---------------------------------------------------------------------------
option(VOLATILE_ENABLE_FUZZING
  "Build fuzz targets with AFL++ or libFuzzer" OFF)
set(VOLATILE_FUZZER "afl++"
  CACHE STRING "Fuzzer backend: afl++ or libfuzzer")
set_property(CACHE VOLATILE_FUZZER PROPERTY STRINGS "afl++" "libfuzzer")

if(VOLATILE_ENABLE_FUZZING)
  if(VOLATILE_FUZZER STREQUAL "afl++")
    # Expect CC=afl-cc (or afl-clang-fast) set in the environment.
    find_program(AFL_CC afl-cc afl-clang-fast)
    if(AFL_CC)
      message(STATUS "Fuzzing: AFL++ via ${AFL_CC}")
    else()
      message(WARNING
        "VOLATILE_ENABLE_FUZZING=afl++ but afl-cc not found in PATH. "
        "Set CC=afl-cc or install AFL++.")
    endif()
    add_subdirectory(test/fuzz)
  elseif(VOLATILE_FUZZER STREQUAL "libfuzzer")
    if(NOT VOLATILE_COMPILER_CLANG)
      message(FATAL_ERROR "libFuzzer requires Clang")
    endif()
    message(STATUS "Fuzzing: libFuzzer")
    add_subdirectory(test/fuzz)
  else()
    message(FATAL_ERROR "Unknown VOLATILE_FUZZER value: ${VOLATILE_FUZZER}")
  endif()
endif()

# =============================================================================
# Analysis & Reporting
# =============================================================================

# ---------------------------------------------------------------------------
# Coverage — instrument for gcov (GCC) or llvm-cov (Clang).
# After running tests, generate a report with:
#   gcov / lcov  (GCC)  or  llvm-cov show  (Clang)
# ---------------------------------------------------------------------------
option(VOLATILE_ENABLE_COVERAGE "Code coverage (gcov/llvm-cov)" OFF)
if(VOLATILE_ENABLE_COVERAGE)
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate)
  else()
    # GCC classic gcov instrumentation; -O0 avoids inlining hiding lines.
    add_compile_options(-fprofile-arcs -ftest-coverage -O0)
    add_link_options(--coverage)
  endif()
  message(STATUS "Coverage instrumentation enabled")
endif()

# ---------------------------------------------------------------------------
# Profiling — gprof (-pg) for GCC, or Clang's sampling profiler.
# ---------------------------------------------------------------------------
option(VOLATILE_ENABLE_PROFILING
  "Enable profiling (-pg or -fprofile-instr-generate)" OFF)
if(VOLATILE_ENABLE_PROFILING)
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(-fprofile-instr-generate)
    add_link_options(-fprofile-instr-generate)
  else()
    add_compile_options(-pg)
    add_link_options(-pg)
  endif()
  message(STATUS "Profiling instrumentation enabled")
endif()

# ---------------------------------------------------------------------------
# Vectorization report — useful for tuning hot loops.
# GCC:   -fopt-info-vec-all  (writes to stderr during compilation)
# Clang: -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
# ---------------------------------------------------------------------------
option(VOLATILE_VECTORIZATION_REPORT
  "Print auto-vectorization reports" OFF)
if(VOLATILE_VECTORIZATION_REPORT)
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(
      -Rpass=loop-vectorize
      -Rpass-missed=loop-vectorize
      -Rpass-analysis=loop-vectorize
    )
  else()
    add_compile_options(-fopt-info-vec-all)
  endif()
  message(STATUS "Vectorization report enabled")
endif()

# ---------------------------------------------------------------------------
# Full optimization report (Clang -Rpass).
# Dumps every inlining, unrolling, and vectorization decision to stderr.
# Very verbose — useful for targeted hot-path investigations.
# ---------------------------------------------------------------------------
option(VOLATILE_OPT_REPORT
  "Print full optimization report (clang -Rpass)" OFF)
if(VOLATILE_OPT_REPORT)
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(
      -Rpass=.*
      -Rpass-missed=.*
      -Rpass-analysis=.*
    )
  else()
    add_compile_options(-fopt-info-all)
  endif()
  message(STATUS "Full optimization report enabled")
endif()

# =============================================================================
# Extended Warnings
# =============================================================================

# Additional diagnostics beyond -Wall -Wextra -Wpedantic already set in the
# top-level CMakeLists.  These surface real bugs but may require suppression
# on third-party code or generated files.
option(VOLATILE_EXTRA_WARNINGS "Enable extensive compiler warnings" ON)
if(VOLATILE_EXTRA_WARNINGS)
  add_compile_options(
    -Wshadow
    -Wdouble-promotion
    -Wformat=2
    -Wundef
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wimplicit-fallthrough
    -Wstrict-prototypes
    -Wold-style-definition
    -Wmissing-prototypes
  )
  if(VOLATILE_COMPILER_CLANG)
    add_compile_options(
      -Weverything
      # Suppress noisy Weverything sub-warnings that are impractical to fix:
      -Wno-padded                      # struct padding is intentional in many places
      -Wno-declaration-after-statement # C99 mixed declarations are fine
      -Wno-unsafe-buffer-usage         # too many false positives on idiomatic C
      -Wno-disabled-macro-expansion    # recursive macros used intentionally
      # Redundant with the explicit list above but harmless:
      -Wno-unused-parameter
    )
  endif()
  message(STATUS "Extended warnings enabled")
endif()

# =============================================================================
# Linker Options
# =============================================================================

# ---------------------------------------------------------------------------
# lld — LLVM's linker.  Significantly faster than GNU ld, especially for
# large C++ or LTO builds.  Also required for ThinLTO on some platforms.
# ---------------------------------------------------------------------------
option(VOLATILE_USE_LLD "Use lld linker (faster than ld/gold)" OFF)
if(VOLATILE_USE_LLD)
  find_program(LLD_PROGRAM lld ld.lld)
  if(LLD_PROGRAM)
    add_link_options(-fuse-ld=lld)
    message(STATUS "Using lld linker: ${LLD_PROGRAM}")
  else()
    message(WARNING "VOLATILE_USE_LLD requested but lld not found in PATH")
  endif()
endif()

# ---------------------------------------------------------------------------
# ICF (Identical Code Folding) — merges functions with identical machine code.
# Can reduce binary size by 5-20% in code-heavy projects.
# Requires lld or gold; GNU ld does not support --icf.
# ---------------------------------------------------------------------------
option(VOLATILE_LINKER_ICF "Enable Identical Code Folding (--icf=all)" OFF)
if(VOLATILE_LINKER_ICF)
  add_link_options(LINKER:--icf=all)
  message(STATUS "Linker ICF enabled")
endif()

# ---------------------------------------------------------------------------
# GC sections — remove unused functions and data from the final binary.
# -ffunction-sections/-fdata-sections puts each symbol in its own section;
# --gc-sections then discards unreferenced sections at link time.
# Default ON because it almost always reduces binary size with no downside.
# ---------------------------------------------------------------------------
option(VOLATILE_LINKER_GC_SECTIONS
  "Enable -ffunction-sections -fdata-sections + --gc-sections" ON)
if(VOLATILE_LINKER_GC_SECTIONS)
  add_compile_options(-ffunction-sections -fdata-sections)
  add_link_options(LINKER:--gc-sections)
  message(STATUS "Linker GC sections enabled")
endif()
