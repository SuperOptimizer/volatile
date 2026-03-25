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
  test_vol test_cache test_compress4d
)

# core + render
set(_render_tests
  test_cmap test_composite test_overlay test_camera test_tile
)

# core + gpu
set(_gpu_tests
  test_vk test_gpu test_shader
)

# core + server
set(_server_tests
  test_db test_protocol test_srv
)

# core + server + render + cli
set(_cli_tests
  test_cli_convert test_cli_grow
)

# core + render + gui  (EXCLUDE_FROM_ALL — only built when volatile_gui is)
set(_gui_tests
  test_keybind test_seg test_settings test_viewer test_viewer3d
  test_annotate test_browse test_filewatcher test_focus_history
  test_nk_widgets test_console test_plugin_mgr test_surface_panel
  test_neural_trace test_crosshair test_seg_growth test_approval_mask
  test_layout test_drawing test_seeding test_vol_selector test_window_range
)

# tests not listed above (e.g. CLI tests) get core only
foreach(_src IN LISTS _test_sources)
  get_filename_component(_name ${_src} NAME_WE)

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
    elseif(_name IN_LIST _cli_tests)
      target_link_libraries(${_name} PRIVATE volatile_cli)
    else()
      # core_only_tests and anything unlisted
      target_link_libraries(${_name} PRIVATE volatile_core)
    endif()

    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES TIMEOUT 10)
  endif()
endforeach()
