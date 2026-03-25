add_library(volatile_core STATIC
  log.c
  json.c
  hash.c
  thread.c
  chunk.c
  math.c
  net.c
  geom.c
  geom_mesh.c
  io.c
  imgproc.c
  vol.c
  vol_write.c
  vol_mirror.c
  cache.c
  compress4d.c
  compress4d_zarr.c
  profile.c
  sparse.c
  project.c
  tracer.c
)

target_include_directories(volatile_core PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(volatile_core PUBLIC
  m
  Threads::Threads
  CURL::libcurl
  blosc2_static
  SQLite::SQLite3
)

find_package(Threads REQUIRED)

set_target_properties(volatile_core PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
