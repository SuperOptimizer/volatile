find_package(Qt6 REQUIRED COMPONENTS Widgets Gui Core)

# VTK is optional — VolumeViewer compiles to a stub without it
find_package(VTK QUIET COMPONENTS
  CommonCore CommonDataModel RenderingCore RenderingOpenGL2
  RenderingVolume RenderingVolumeOpenGL2 InteractionStyle FiltersCore GUISupportQt
)
if(VTK_FOUND)
  message(STATUS "VTK found — building VolumeViewer with VTK backend")
  set(VOLATILE_HAVE_VTK TRUE)
else()
  message(STATUS "VTK not found — VolumeViewer will be a placeholder widget")
  set(VOLATILE_HAVE_VTK FALSE)
endif()

add_executable(volatile-gui
  main.cpp
  MainWindow.cpp
  SliceViewer.cpp
  VolumeViewer.cpp
  KeybindManager.cpp
  AboutDialog.cpp
  KeybindsDialog.cpp
  SegmentationPanel.cpp
  SurfaceTreeWidget.cpp
  S3BrowserDialog.cpp
)

target_include_directories(volatile-gui PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(volatile-gui PRIVATE
  volatile_core
  Qt6::Widgets
  Qt6::Gui
  Qt6::Core
)

if(VOLATILE_HAVE_VTK)
  target_compile_definitions(volatile-gui PRIVATE HAVE_VTK)
  target_link_libraries(volatile-gui PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS volatile-gui MODULES ${VTK_LIBRARIES})
endif()

set_target_properties(volatile-gui PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  AUTOMOC ON
  AUTORCC ON
)
