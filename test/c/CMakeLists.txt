file(GLOB _test_sources CONFIGURE_DEPENDS test_*.c)

# ---------------------------------------------------------------------------
# Per-test link-library mapping.
# Tests are grouped by the libraries they actually need.
# volatile_gui is EXCLUDE_FROM_ALL — GUI tests are built on demand only.
# ---------------------------------------------------------------------------

# core only
set(_core_only_tests
  test_log test_hash test_json test_chunk test_thread
  test_math test_geom test_imgproc test_io test_net
  test_vol test_cache test_compress4d test_compress4d_zarr test_vol_mirror
  test_real_data
)

# core + render
set(_render_tests
  test_cmap test_composite test_overlay test_camera test_tile
  test_integration
)

# core + gpu
set(_gpu_tests
  test_vk test_gpu test_shader test_gpu_compute
)

# core + server
set(_server_tests
  test_db test_protocol test_srv test_review test_gitstore test_versioning test_stream
)

# core + server + render (collab uses overlay_list)
set(_server_render_tests
  test_collab
)

# core + server + render + cli
set(_cli_tests
  test_cli_convert test_cli_grow
  test_metrics test_mask test_inpaint
  test_compress4d_cli
)

# core + render + gui  (EXCLUDE_FROM_ALL — only built when volatile_gui is)
set(_gui_tests
  test_keybind test_seg test_settings test_viewer test_viewer3d
  test_annotate test_browse test_filewatcher test_focus_history
  test_nk_widgets test_console test_plugin_mgr test_surface_panel
  test_neural_trace test_crosshair test_seg_growth test_approval_mask
  test_layout test_drawing test_seeding test_vol_selector test_window_range
  test_toolbar test_slice_controller test_cred_dialog test_viewer_controls
  test_menubar test_settings_dialog test_seg_panels test_overlay_vol
)

# ---------------------------------------------------------------------------
# bench_compress4d — NOT a ctest; run manually: ./build/test/c/bench_compress4d
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_compress4d.c")
  add_executable(bench_compress4d bench_compress4d.c)
  target_include_directories(bench_compress4d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${greatest_SOURCE_DIR}
  )
  target_link_libraries(bench_compress4d PRIVATE volatile_core blosc2_static m)
  set_target_properties(bench_compress4d PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endif()

# test_tool_runner: compile tool_runner.c directly — no GUI/SDL deps needed.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_tool_runner.c")
  add_executable(test_tool_runner
    test_tool_runner.c
    ${CMAKE_SOURCE_DIR}/src/gui/tool_runner.c
  )
  target_include_directories(test_tool_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${greatest_SOURCE_DIR}
  )
  target_link_libraries(test_tool_runner PRIVATE volatile_core)
  add_test(NAME test_tool_runner COMMAND test_tool_runner)
  set_tests_properties(test_tool_runner PROPERTIES TIMEOUT 10)
endif()

# test_sync: compile sync.c directly with an inline mock gitstore (avoids
# duplicate-symbol clash if gitstore.c is also in volatile_server).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_sync.c")
  add_executable(test_sync
    test_sync.c
    ${CMAKE_SOURCE_DIR}/src/server/sync.c
  )
  target_include_directories(test_sync PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${greatest_SOURCE_DIR}
  )
  target_link_libraries(test_sync PRIVATE volatile_core pthread)
  add_test(NAME test_sync COMMAND test_sync)
  set_tests_properties(test_sync PROPERTIES TIMEOUT 10)
endif()

# tests handled specially above (not processed by the generic loop)
set(_special_tests test_sync test_tool_runner)

# tests not listed above (e.g. CLI tests) get core only
foreach(_src IN LISTS _test_sources)
  get_filename_component(_name ${_src} NAME_WE)
  if(_name IN_LIST _special_tests)
    continue()
  endif()

  if(_name IN_LIST _gui_tests)
    # GUI tests: only add executable if volatile_gui target exists
    if(TARGET volatile_gui)
      add_executable(${_name} EXCLUDE_FROM_ALL ${_src})
      target_link_libraries(${_name} PRIVATE volatile_core volatile_render volatile_gui)
      target_include_directories(${_name} PRIVATE ${greatest_SOURCE_DIR})
      add_test(NAME ${_name} COMMAND ${_name})
      set_tests_properties(${_name} PROPERTIES TIMEOUT 10)
    endif()
  else()
    add_executable(${_name} ${_src})
    target_include_directories(${_name} PRIVATE ${greatest_SOURCE_DIR})

    if(_name IN_LIST _render_tests)
      target_link_libraries(${_name} PRIVATE volatile_core volatile_render)
    elseif(_name IN_LIST _gpu_tests)
      target_link_libraries(${_name} PRIVATE volatile_core volatile_gpu)
    elseif(_name IN_LIST _server_tests)
      target_link_libraries(${_name} PRIVATE volatile_core volatile_server)
    elseif(_name IN_LIST _server_render_tests)
      target_link_libraries(${_name} PRIVATE volatile_core volatile_server volatile_render)
    elseif(_name IN_LIST _cli_tests)
      target_link_libraries(${_name} PRIVATE volatile_cli)
    else()
      # core_only_tests and anything unlisted
      target_link_libraries(${_name} PRIVATE volatile_core)
    endif()

    add_test(NAME ${_name} COMMAND ${_name})
    # GPU compute tests invoke glslc at runtime; give them more time.
    # Network integration tests need extra time for real HTTP downloads.
    if(_name STREQUAL "test_gpu_compute")
      set_tests_properties(${_name} PROPERTIES TIMEOUT 60)
    elseif(_name STREQUAL "test_real_data")
      set_tests_properties(${_name} PROPERTIES TIMEOUT 120)
    else()
      set_tests_properties(${_name} PROPERTIES TIMEOUT 10)
    endif()
  endif()
endforeach()
