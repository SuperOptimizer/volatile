add_library(volatile_render STATIC
  cmap.c
  camera.c
  tile.c
  overlay.c
  composite.c
)

target_include_directories(volatile_render PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(volatile_render PUBLIC volatile_core m)

set_target_properties(volatile_render PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
