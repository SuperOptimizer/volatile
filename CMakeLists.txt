cmake_minimum_required(VERSION 3.25)
project(volatile VERSION 0.1.0 LANGUAGES C)

# ---------------------------------------------------------------------------
# C standard — set on our own targets, not globally, so FetchContent
# subprojects (e.g. blosc2/zlib-ng) are not forced into C23.
# See src/core/CMakeLists.txt for per-target enforcement.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tooling
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Recommended: configure with -G Ninja
# cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..

# ---------------------------------------------------------------------------
# Advanced build options (optimization profiles, sanitizers, LTO, PGO, etc.)
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(VolatileBuildOptions)

# ---------------------------------------------------------------------------
# Global compiler flags (baseline diagnostics; extras in VolatileBuildOptions)
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(VOLATILE_BUILD_TESTS "Build C test suite" ON)

# ---------------------------------------------------------------------------
# Dependencies — system
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(SDL3 REQUIRED)

# ---------------------------------------------------------------------------
# Dependencies — FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# greatest.h — single-header C unit test framework
FetchContent_Declare(
  greatest
  GIT_REPOSITORY https://github.com/silentbicycle/greatest.git
  GIT_TAG        v1.5.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(greatest)

# c-blosc2 — fast compressed storage
FetchContent_Declare(
  blosc2
  GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
  GIT_TAG        v2.15.2
  GIT_SHALLOW    TRUE
)
set(BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(blosc2)

# Nuklear — immediate-mode GUI (header-only)
FetchContent_Declare(
  nuklear
  GIT_REPOSITORY https://github.com/Immediate-Mode-UI/Nuklear.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nuklear)

# nuklear_sdl3_renderer — SDL3 renderer backend bridge for Nuklear
# NOTE: single-header bridge lives in demo/sdl_renderer/ of the Nuklear repo
FetchContent_Declare(
  nuklear_sdl3_renderer
  GIT_REPOSITORY https://github.com/Immediate-Mode-UI/Nuklear.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  demo/sdl_renderer
)
# NOTE: We use nuklear_SOURCE_DIR for nuklear.h and
# ${CMAKE_SOURCE_DIR}/src/gui/vendor for the renderer bridge header,
# which we copy from the Nuklear demo tree at configure time.
set(nuklear_sdl3_renderer_SOURCE_DIR ${nuklear_SOURCE_DIR}/demo/sdl3_renderer)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/gui)
add_subdirectory(src/render)
add_subdirectory(src/gpu)
add_subdirectory(src/server)
add_subdirectory(src/cli)

if(VOLATILE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test/c)
  add_subdirectory(test/plugins)
endif()
