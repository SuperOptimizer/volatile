cmake_minimum_required(VERSION 3.25)
project(volatile VERSION 0.1.0 LANGUAGES C CXX)

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
option(VOLATILE_BUILD_TESTS    "Build C test suite"       ON)
option(VOLATILE_GUI_QT         "Build Qt6 GUI"            ON)
option(VOLATILE_GUI_NUKLEAR    "Build Nuklear/SDL3 GUI"   OFF)

# ---------------------------------------------------------------------------
# Dependencies — system (core always needs these)
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)

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

# Suppress all warnings on blosc2 and its internal deps (zlib-ng, zstd, lz4)
# These are third-party code we don't control
function(suppress_warnings_recursive dir)
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t IN LISTS targets)
    get_property(type TARGET "${t}" PROPERTY TYPE)
    if(type MATCHES "STATIC_LIBRARY|SHARED_LIBRARY|OBJECT_LIBRARY")
      target_compile_options("${t}" PRIVATE -w)
    endif()
  endforeach()
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(sd IN LISTS subdirs)
    suppress_warnings_recursive("${sd}")
  endforeach()
endfunction()
if(blosc2_SOURCE_DIR)
  suppress_warnings_recursive("${blosc2_SOURCE_DIR}")
endif()

# Nuklear + SDL3 — only fetched when Nuklear GUI is enabled
if(VOLATILE_GUI_NUKLEAR)
  find_package(SDL3 QUIET)
  if(SDL3_FOUND)
    # Nuklear — immediate-mode GUI (header-only)
    FetchContent_Declare(
      nuklear
      GIT_REPOSITORY https://github.com/Immediate-Mode-UI/Nuklear.git
      GIT_TAG        master
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nuklear)

    # nuklear_sdl3_renderer — SDL3 renderer backend bridge for Nuklear
    # NOTE: single-header bridge lives in demo/sdl3_renderer/ of the Nuklear repo.
    # We use nuklear_SOURCE_DIR for nuklear.h; the renderer header comes from there.
    set(nuklear_sdl3_renderer_SOURCE_DIR ${nuklear_SOURCE_DIR}/demo/sdl3_renderer)
  else()
    message(STATUS "SDL3 not found — disabling Nuklear GUI")
    set(VOLATILE_GUI_NUKLEAR OFF CACHE BOOL "" FORCE)
  endif()
endif()

# ---------------------------------------------------------------------------
# Subdirectories — core always built
# ---------------------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/render)
add_subdirectory(src/gpu)
add_subdirectory(src/server)
add_subdirectory(src/cli)

# Qt GUI — optional
if(VOLATILE_GUI_QT)
  find_package(Qt6 QUIET COMPONENTS Widgets Gui Core)
  if(Qt6_FOUND)
    message(STATUS "Qt6 found — building Qt GUI")
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gui_qt/CMakeLists.txt")
      add_subdirectory(src/gui_qt)
    endif()
  else()
    message(STATUS "Qt6 not found — skipping Qt GUI (Nuklear fallback available)")
  endif()
endif()

# Nuklear/SDL3 GUI — optional fallback
if(VOLATILE_GUI_NUKLEAR AND SDL3_FOUND)
  message(STATUS "SDL3 found — building Nuklear GUI")
  add_subdirectory(src/gui)
endif()

if(VOLATILE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test/c)
  add_subdirectory(test/plugins)
endif()
