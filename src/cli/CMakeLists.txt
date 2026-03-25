add_library(volatile_cli STATIC
  cli_compress.c
  cli_convert.c
  cli_flatten.c
  cli_stats.c
  cli_grow.c
  cli_serve.c
  cli_connect.c
  cli_render.c
  cli_zarr_ops.c
  cli_metrics.c
  cli_mask.c
  cli_inpaint.c
  cli_normals.c
  cli_winding.c
)

target_include_directories(volatile_cli PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/cli
)

target_link_libraries(volatile_cli PUBLIC volatile_core volatile_server volatile_render)

set_target_properties(volatile_cli PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

add_executable(volatile-cli main.c)

target_include_directories(volatile-cli PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/cli
)

target_link_libraries(volatile-cli PRIVATE volatile_cli)

set_target_properties(volatile-cli PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
