# ---------------------------------------------------------------------------
# volatile_gui — SDL3 + Nuklear window scaffold
# EXCLUDE_FROM_ALL: only built when explicitly requested or by a GUI test.
# ---------------------------------------------------------------------------

add_library(volatile_gui STATIC EXCLUDE_FROM_ALL
  # files that were incorrectly placed in volatile_core
  browse.c
  annotate.c
  filewatcher.c
  focus_history.c
  # proper GUI files
  app.c
  keybind.c
  settings.c
  seg.c
  seg_growth.c
  surface_panel.c
  neural_trace.c
  viewer.c
  viewer3d.c
  nk_widgets.c
  console.c
  plugin.c
  crosshair.c
  slice_controller.c
  draw_panel.c
  file_dialog.c
  layout.c
  approval_mask.c
  drawing.c
  seeding.c
  vol_selector.c
  window_range.c
  toolbar.c
  menubar.c
  tool_runner.c
  cred_dialog.c
  viewer_controls.c
  settings_dialog.c
  overlay_vol.c
  seg_panels.c
  dt_panel.c
  point_panel.c
  vol_info_panel.c
  review_report.c
  about_dialog.c
  keybinds_dialog.c
  statusbar.c
  scalebar.c
)

target_include_directories(volatile_gui PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

# Nuklear is header-only; SDL3-renderer bridge header lives in its demo tree.
target_include_directories(volatile_gui PRIVATE
  ${nuklear_SOURCE_DIR}
  ${nuklear_sdl3_renderer_SOURCE_DIR}
)

target_link_libraries(volatile_gui
  PUBLIC  volatile_core volatile_render
  PRIVATE SDL3::SDL3 dl
)

set_target_properties(volatile_gui PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

# Suppress warnings on app.c (includes Nuklear implementation + SDL3 renderer header)
# These are third-party headers we don't control
set_source_files_properties(app.c PROPERTIES COMPILE_OPTIONS "-Wno-conversion;-Wno-sign-conversion;-Wno-float-conversion;-Wno-double-promotion")

# ---------------------------------------------------------------------------
# volatile_gui_app — standalone GUI executable
# ---------------------------------------------------------------------------
add_executable(volatile_gui_app EXCLUDE_FROM_ALL main.c)

target_include_directories(volatile_gui_app PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${nuklear_SOURCE_DIR}
  ${nuklear_sdl3_renderer_SOURCE_DIR}
)

target_link_libraries(volatile_gui_app
  PRIVATE volatile_gui SDL3::SDL3
)

set_target_properties(volatile_gui_app PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
