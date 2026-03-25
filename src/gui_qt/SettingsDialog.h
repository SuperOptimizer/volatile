#pragma once

#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>

extern "C" {
#include "gui/settings.h"
#include "render/cmap.h"
#include "render/composite.h"
}

// ---------------------------------------------------------------------------
// SettingsDialog — Qt mirror of the Nuklear settings_dialog.
//
// Reads and writes through the same C settings system using identical keys.
// Usage:
//   auto *dlg = new SettingsDialog(prefs, this);
//   dlg->exec();                // blocks; applies on OK
// Or non-modal:
//   dlg->show();
// ---------------------------------------------------------------------------
class SettingsDialog : public QDialog {
  Q_OBJECT
public:
  explicit SettingsDialog(settings *prefs, QWidget *parent = nullptr);

  // Re-read all values from the settings object.
  void loadFromSettings();

public slots:
  void accept() override;   // save + close
  void apply();             // save without closing

private:
  void buildUi();

  // Helpers: one per section
  class QGroupBox *buildPreprocessingGroup(QWidget *parent);
  class QGroupBox *buildNormalsGroup(QWidget *parent);
  class QGroupBox *buildViewGroup(QWidget *parent);
  class QGroupBox *buildOverlayGroup(QWidget *parent);
  class QGroupBox *buildRenderGroup(QWidget *parent);
  class QGroupBox *buildCompositeGroup(QWidget *parent);
  class QGroupBox *buildPostprocGroup(QWidget *parent);
  class QGroupBox *buildPerformanceGroup(QWidget *parent);

  void saveToSettings();

  // Convenience: populate a QComboBox with colormap names + gradient icons.
  static void populateCmapCombo(QComboBox *cb);

  settings *m_prefs;

  // ---- Preprocessing ----
  QDoubleSpinBox *m_preWinLo   = nullptr;
  QDoubleSpinBox *m_preWinHi   = nullptr;
  QComboBox      *m_preCmap    = nullptr;
  QDoubleSpinBox *m_preStretch = nullptr;

  // ---- Normals ----
  QCheckBox      *m_normHints      = nullptr;
  QCheckBox      *m_normNormals    = nullptr;
  QSpinBox       *m_normArrowCount = nullptr;
  QDoubleSpinBox *m_normArrowLen   = nullptr;

  // ---- View ----
  QDoubleSpinBox *m_viewZoomSens     = nullptr;
  QCheckBox      *m_viewResetOnSurf  = nullptr;

  // ---- Overlay ----
  QDoubleSpinBox *m_ovOpacity   = nullptr;
  QComboBox      *m_ovCmap      = nullptr;
  QDoubleSpinBox *m_ovThreshold = nullptr;
  QDoubleSpinBox *m_ovWinLo     = nullptr;
  QDoubleSpinBox *m_ovWinHi     = nullptr;

  // ---- Render ----
  QDoubleSpinBox *m_rsIntOpacity   = nullptr;
  QDoubleSpinBox *m_rsIntThickness = nullptr;
  QDoubleSpinBox *m_rsStride       = nullptr;

  // ---- Composite ----
  QSpinBox       *m_compFront        = nullptr;
  QSpinBox       *m_compBehind       = nullptr;
  QComboBox      *m_compMethod       = nullptr;
  QDoubleSpinBox *m_compExtinction   = nullptr;
  QDoubleSpinBox *m_compEmission     = nullptr;
  QDoubleSpinBox *m_compAmbient      = nullptr;
  QDoubleSpinBox *m_compAlphaMin     = nullptr;
  QDoubleSpinBox *m_compAlphaMax     = nullptr;
  QDoubleSpinBox *m_compAlphaOpacity = nullptr;
  QDoubleSpinBox *m_compAlphaCutoff  = nullptr;

  // ---- Post-processing ----
  QDoubleSpinBox *m_ppStretch        = nullptr;
  QSpinBox       *m_ppSmallCompRemove = nullptr;

  // ---- Performance ----
  QSpinBox       *m_perfRamCacheGB  = nullptr;
  QComboBox      *m_perfDownscale   = nullptr;
  QCheckBox      *m_perfFastInterp  = nullptr;
};
