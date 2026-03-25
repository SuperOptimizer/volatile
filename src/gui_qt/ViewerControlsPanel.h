#pragma once

#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QComboBox>
#include <QSpinBox>

extern "C" {
#include "render/cmap.h"
#include "render/composite.h"
}

class SliceViewer;

// ---------------------------------------------------------------------------
// ViewerControlsPanel — dock panel that controls a SliceViewer.
//
// Connect to a viewer with setViewer(); the panel updates the viewer
// immediately on any change and also emits Qt signals for anything else
// that needs to track the same state.
// ---------------------------------------------------------------------------
class ViewerControlsPanel : public QDockWidget {
  Q_OBJECT
public:
  explicit ViewerControlsPanel(QWidget *parent = nullptr);

  // Bind to a viewer; can be called again to switch viewers.
  void setViewer(SliceViewer *viewer);

signals:
  void sliceChanged(float z);
  void zoomChanged(float zoom);
  void colormapChanged(int cmapId);
  void windowLevelChanged(float window, float level);
  void compositeModeChanged(int mode);

public slots:
  // Sync panel state from viewer (call when the viewer changes externally).
  void syncFromViewer();

private slots:
  void onSliceSpinChanged(double value);
  void onZoomSliderChanged(int value);
  void onZoomSpinChanged(double value);
  void onWindowSliderChanged(int value);
  void onLevelSliderChanged(int value);
  void onWindowSpinChanged(double value);
  void onLevelSpinChanged(double value);
  void onColormapChanged(int index);
  void onCompositeModeChanged(int index);

private:
  // Build a colormap combo item: colored gradient icon + name.
  void populateColormapCombo();

  // Slider <-> spinbox sync helpers (avoid recursive signals).
  void setZoomSilent(float zoom);
  void setWindowLevelSilent(float window, float level);

  SliceViewer   *m_viewer  = nullptr;

  QDoubleSpinBox *m_sliceSpin  = nullptr;

  QSlider        *m_zoomSlider = nullptr;
  QDoubleSpinBox *m_zoomSpin   = nullptr;

  // Window = width of intensity range; Level = centre.
  QSlider        *m_windowSlider = nullptr;
  QDoubleSpinBox *m_windowSpin   = nullptr;
  QSlider        *m_levelSlider  = nullptr;
  QDoubleSpinBox *m_levelSpin    = nullptr;

  QComboBox      *m_cmapCombo      = nullptr;
  QComboBox      *m_compositCombo  = nullptr;

  bool m_updating = false; // guard against recursive updates
};
