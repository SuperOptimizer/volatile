add_library(volatile_server STATIC
  protocol.c
  db.c
  srv.c
  collab.c
  versioning.c
  review.c
  gitstore.c
  sync.c
  stream.c
)

target_include_directories(volatile_server PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(volatile_server PUBLIC volatile_core SQLite::SQLite3)

set_target_properties(volatile_server PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
