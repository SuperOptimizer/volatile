#include "ViewerControlsPanel.h"
#include "SliceViewer.h"

#include <QFormLayout>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPainter>
#include <algorithm>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int   kZoomSliderMin  =    10; // 0.10x
static constexpr int   kZoomSliderMax  =  2000; // 20.00x
static constexpr int   kZoomSliderDef  =   100; // 1.00x
static constexpr float kZoomScale      = 100.0f;

static constexpr int   kWLSliderMax    = 1000;
static constexpr float kWLScale        = 1000.0f;

// ---------------------------------------------------------------------------
// Helper: build gradient icon for a colormap
// ---------------------------------------------------------------------------
static QIcon cmapIcon(cmap_id id) {
  constexpr int W = 64, H = 12;
  QPixmap px(W, H);
  QPainter p(&px);
  for (int x = 0; x < W; ++x) {
    double t = static_cast<double>(x) / (W - 1);
    cmap_rgb c = cmap_apply(id, t);
    p.fillRect(x, 0, 1, H, QColor(c.r, c.g, c.b));
  }
  return QIcon(px);
}

// ---------------------------------------------------------------------------
// ViewerControlsPanel
// ---------------------------------------------------------------------------
ViewerControlsPanel::ViewerControlsPanel(QWidget *parent)
  : QDockWidget(tr("Viewer Controls"), parent)
{
  setObjectName("ViewerControlsPanel");

  auto *container = new QWidget(this);
  auto *vlay = new QVBoxLayout(container);
  vlay->setContentsMargins(6, 6, 6, 6);
  vlay->setSpacing(6);

  // ---- Slice group ----
  {
    auto *grp  = new QGroupBox(tr("Slice"), container);
    auto *form = new QFormLayout(grp);

    m_sliceSpin = new QDoubleSpinBox;
    m_sliceSpin->setRange(0.0, 99999.0);
    m_sliceSpin->setDecimals(1);
    m_sliceSpin->setSingleStep(1.0);
    form->addRow(tr("Position:"), m_sliceSpin);

    vlay->addWidget(grp);
  }

  // ---- Zoom group ----
  {
    auto *grp  = new QGroupBox(tr("Zoom"), container);
    auto *form = new QFormLayout(grp);

    auto *zrow = new QHBoxLayout;
    m_zoomSlider = new QSlider(Qt::Horizontal);
    m_zoomSlider->setRange(kZoomSliderMin, kZoomSliderMax);
    m_zoomSlider->setValue(kZoomSliderDef);
    m_zoomSlider->setTickInterval(100);

    m_zoomSpin = new QDoubleSpinBox;
    m_zoomSpin->setRange(0.10, 20.0);
    m_zoomSpin->setDecimals(2);
    m_zoomSpin->setSingleStep(0.1);
    m_zoomSpin->setValue(1.0);
    m_zoomSpin->setFixedWidth(70);

    zrow->addWidget(m_zoomSlider, 1);
    zrow->addWidget(m_zoomSpin);
    form->addRow(tr("Zoom:"), zrow);

    vlay->addWidget(grp);
  }

  // ---- Window / Level group ----
  {
    auto *grp  = new QGroupBox(tr("Window / Level"), container);
    auto *form = new QFormLayout(grp);

    auto buildWLRow = [&](QSlider *&slider, QDoubleSpinBox *&spin) {
      auto *row = new QHBoxLayout;
      slider = new QSlider(Qt::Horizontal);
      slider->setRange(0, kWLSliderMax);
      spin = new QDoubleSpinBox;
      spin->setRange(0.0, 1.0);
      spin->setDecimals(3);
      spin->setSingleStep(0.01);
      spin->setFixedWidth(70);
      row->addWidget(slider, 1);
      row->addWidget(spin);
      return row;
    };

    auto *wrow = buildWLRow(m_windowSlider, m_windowSpin);
    m_windowSpin->setValue(1.0);
    m_windowSlider->setValue(kWLSliderMax);
    form->addRow(tr("Window:"), wrow);

    auto *lrow = buildWLRow(m_levelSlider, m_levelSpin);
    m_levelSpin->setValue(0.5);
    m_levelSlider->setValue(kWLSliderMax / 2);
    form->addRow(tr("Level:"), lrow);

    vlay->addWidget(grp);
  }

  // ---- Display group ----
  {
    auto *grp  = new QGroupBox(tr("Display"), container);
    auto *form = new QFormLayout(grp);

    m_cmapCombo = new QComboBox;
    populateColormapCombo();
    form->addRow(tr("Colormap:"), m_cmapCombo);

    m_compositCombo = new QComboBox;
    for (int i = 0; i < static_cast<int>(COMPOSITE_COUNT); ++i)
      m_compositCombo->addItem(QString::fromLatin1(composite_mode_name(
          static_cast<composite_mode>(i))));
    form->addRow(tr("Composite:"), m_compositCombo);

    vlay->addWidget(grp);
  }

  vlay->addStretch();
  setWidget(container);

  // ---- Wire signals ----
  connect(m_sliceSpin,    qOverload<double>(&QDoubleSpinBox::valueChanged),
          this,           &ViewerControlsPanel::onSliceSpinChanged);
  connect(m_zoomSlider,   &QSlider::valueChanged,
          this,           &ViewerControlsPanel::onZoomSliderChanged);
  connect(m_zoomSpin,     qOverload<double>(&QDoubleSpinBox::valueChanged),
          this,           &ViewerControlsPanel::onZoomSpinChanged);
  connect(m_windowSlider, &QSlider::valueChanged,
          this,           &ViewerControlsPanel::onWindowSliderChanged);
  connect(m_levelSlider,  &QSlider::valueChanged,
          this,           &ViewerControlsPanel::onLevelSliderChanged);
  connect(m_windowSpin,   qOverload<double>(&QDoubleSpinBox::valueChanged),
          this,           &ViewerControlsPanel::onWindowSpinChanged);
  connect(m_levelSpin,    qOverload<double>(&QDoubleSpinBox::valueChanged),
          this,           &ViewerControlsPanel::onLevelSpinChanged);
  connect(m_cmapCombo,    qOverload<int>(&QComboBox::currentIndexChanged),
          this,           &ViewerControlsPanel::onColormapChanged);
  connect(m_compositCombo,qOverload<int>(&QComboBox::currentIndexChanged),
          this,           &ViewerControlsPanel::onCompositeModeChanged);
}

// ---------------------------------------------------------------------------
// setViewer
// ---------------------------------------------------------------------------
void ViewerControlsPanel::setViewer(SliceViewer *viewer) {
  if (m_viewer == viewer) return;

  if (m_viewer) {
    disconnect(m_viewer, &SliceViewer::sliceChanged, this, &ViewerControlsPanel::syncFromViewer);
  }

  m_viewer = viewer;

  if (m_viewer) {
    connect(m_viewer, &SliceViewer::sliceChanged, this, &ViewerControlsPanel::syncFromViewer);
    syncFromViewer();
  }
}

// ---------------------------------------------------------------------------
// syncFromViewer
// ---------------------------------------------------------------------------
void ViewerControlsPanel::syncFromViewer() {
  if (!m_viewer || m_updating) return;
  m_updating = true;

  m_sliceSpin->setValue(static_cast<double>(m_viewer->currentSlice()));
  // zoom, window/level, cmap not exposed on SliceViewer yet — leave as-is.

  m_updating = false;
}

// ---------------------------------------------------------------------------
// Slot implementations
// ---------------------------------------------------------------------------
void ViewerControlsPanel::onSliceSpinChanged(double value) {
  if (m_updating) return;
  if (m_viewer) m_viewer->setSlice(static_cast<float>(value));
  emit sliceChanged(static_cast<float>(value));
}

void ViewerControlsPanel::onZoomSliderChanged(int value) {
  if (m_updating) return;
  m_updating = true;
  float zoom = value / kZoomScale;
  m_zoomSpin->setValue(static_cast<double>(zoom));
  m_updating = false;
  emit zoomChanged(zoom);
}

void ViewerControlsPanel::onZoomSpinChanged(double value) {
  if (m_updating) return;
  m_updating = true;
  m_zoomSlider->setValue(static_cast<int>(value * kZoomScale));
  m_updating = false;
  emit zoomChanged(static_cast<float>(value));
}

void ViewerControlsPanel::onWindowSliderChanged(int value) {
  if (m_updating) return;
  m_updating = true;
  float window = value / kWLScale;
  m_windowSpin->setValue(static_cast<double>(window));
  m_updating = false;
  if (m_viewer) m_viewer->setWindowLevel(window,
      static_cast<float>(m_levelSpin->value()));
  emit windowLevelChanged(window, static_cast<float>(m_levelSpin->value()));
}

void ViewerControlsPanel::onLevelSliderChanged(int value) {
  if (m_updating) return;
  m_updating = true;
  float level = value / kWLScale;
  m_levelSpin->setValue(static_cast<double>(level));
  m_updating = false;
  if (m_viewer) m_viewer->setWindowLevel(
      static_cast<float>(m_windowSpin->value()), level);
  emit windowLevelChanged(static_cast<float>(m_windowSpin->value()), level);
}

void ViewerControlsPanel::onWindowSpinChanged(double value) {
  if (m_updating) return;
  m_updating = true;
  m_windowSlider->setValue(static_cast<int>(value * kWLScale));
  m_updating = false;
  if (m_viewer) m_viewer->setWindowLevel(static_cast<float>(value),
      static_cast<float>(m_levelSpin->value()));
  emit windowLevelChanged(static_cast<float>(value),
                          static_cast<float>(m_levelSpin->value()));
}

void ViewerControlsPanel::onLevelSpinChanged(double value) {
  if (m_updating) return;
  m_updating = true;
  m_levelSlider->setValue(static_cast<int>(value * kWLScale));
  m_updating = false;
  if (m_viewer) m_viewer->setWindowLevel(
      static_cast<float>(m_windowSpin->value()), static_cast<float>(value));
  emit windowLevelChanged(static_cast<float>(m_windowSpin->value()),
                          static_cast<float>(value));
}

void ViewerControlsPanel::onColormapChanged(int index) {
  if (m_updating) return;
  if (m_viewer) m_viewer->setColormap(index);
  emit colormapChanged(index);
}

void ViewerControlsPanel::onCompositeModeChanged(int index) {
  if (m_updating) return;
  emit compositeModeChanged(index);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------
void ViewerControlsPanel::populateColormapCombo() {
  for (int i = 0; i < cmap_count(); ++i) {
    const char *name = cmap_name(static_cast<cmap_id>(i));
    m_cmapCombo->addItem(cmapIcon(static_cast<cmap_id>(i)),
                         QString::fromLatin1(name ? name : ""));
  }
}
