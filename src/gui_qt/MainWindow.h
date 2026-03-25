#pragma once

#include <QMainWindow>
#include <QLabel>

class QDockWidget;
class QSplitter;
class QPlainTextEdit;
class QComboBox;
class QSlider;
class QTreeWidget;
class SliceViewer;
class VolumeViewer;

extern "C" {
#include "core/vol.h"
}

// VC3D-like main window:
//   Top:    QMenuBar  (File / Edit / View / Selection / Help)
//   Center: 2×2 grid of viewers (XY, XZ, YZ, 3D) in a QSplitter
//   Right:  QDockWidget — volume selector, window/level, surface tree, seg panels
//   Bottom: QDockWidget — log console
//   Bottom: QStatusBar
class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override;

  void openVolume(const QString &path);

private slots:
  void onOpenZarr();
  void onOpenVolpkg();
  void onOpenS3();
  void onCloseVolume();
  void onAbout();

  void onSliceChanged(float z);
  void onCursorMoved(float x, float y, float z);

private:
  void buildMenuBar();
  void buildViewerGrid();
  void buildRightDock();
  void buildConsoleDock();
  void updateStatusBar(float x, float y, float z);

  // Viewers
  SliceViewer  *m_xyViewer  = nullptr;
  SliceViewer  *m_xzViewer  = nullptr;
  SliceViewer  *m_yzViewer  = nullptr;
  VolumeViewer *m_3dViewer  = nullptr;

  // Right dock widgets
  QComboBox   *m_volCombo   = nullptr;
  QSlider     *m_windowSlider = nullptr;
  QSlider     *m_levelSlider  = nullptr;
  QTreeWidget *m_surfaceTree  = nullptr;

  // Console
  QPlainTextEdit *m_console = nullptr;

  // Status bar label
  QLabel      *m_coordLabel = nullptr;

  // Volume state
  volume      *m_vol        = nullptr;
};
