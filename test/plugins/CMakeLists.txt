# Build the test plugin as a shared library
add_library(test_plugin MODULE test_plugin.c)

target_include_directories(test_plugin PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(test_plugin PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  PREFIX ""            # produce test_plugin.so, not libtest_plugin.so
)

# Directory used by test_plugin_mgr at runtime
set(TEST_PLUGIN_DIR "$<TARGET_FILE_DIR:test_plugin>")

# Unit test executable for the plugin manager
add_executable(test_plugin_mgr test_plugin_mgr.c)

target_include_directories(test_plugin_mgr PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${greatest_SOURCE_DIR}
)

target_link_libraries(test_plugin_mgr PRIVATE volatile_core volatile_gui dl)

# Pass the plugin .so directory as a compile definition so the test can find it
target_compile_definitions(test_plugin_mgr PRIVATE
  TEST_PLUGIN_DIR="${TEST_PLUGIN_DIR}"
  TEST_PLUGIN_PATH="$<TARGET_FILE:test_plugin>"
)

set_target_properties(test_plugin_mgr PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

# Ensure the .so is built before the test runs
add_dependencies(test_plugin_mgr test_plugin)

add_test(NAME test_plugin_mgr COMMAND test_plugin_mgr)
set_tests_properties(test_plugin_mgr PROPERTIES TIMEOUT 10)
