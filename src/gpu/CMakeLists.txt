find_package(Vulkan REQUIRED)

add_library(volatile_gpu STATIC
  vk.c
  gpu.c
  shader.c
  mtl.c
  dx12.c
)

target_include_directories(volatile_gpu PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(volatile_gpu PUBLIC
  volatile_core
  Vulkan::Vulkan
)

set_target_properties(volatile_gpu PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
