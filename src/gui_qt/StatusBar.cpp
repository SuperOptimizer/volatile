#include "StatusBar.h"

#include <QLabel>
#include <QFrame>
#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

QLabel *makeLabel(const QString &toolTip) {
  auto *l = new QLabel;
  l->setToolTip(toolTip);
  l->setMinimumWidth(80);
  l->setAlignment(Qt::AlignCenter);
  return l;
}

QFrame *makeSep() {
  auto *f = new QFrame;
  f->setFrameShape(QFrame::VLine);
  f->setFrameShadow(QFrame::Sunken);
  return f;
}

} // namespace

// ---------------------------------------------------------------------------
// StatusBar
// ---------------------------------------------------------------------------
StatusBar::StatusBar(QWidget *parent) : QStatusBar(parent) {
  setSizeGripEnabled(false);

  m_cursor = makeLabel(tr("Cursor position (x, y, z)"));
  m_value  = makeLabel(tr("Voxel value at cursor"));
  m_zoom   = makeLabel(tr("Zoom level"));
  m_lod    = makeLabel(tr("Level of detail (0 = full resolution)"));
  m_fps    = makeLabel(tr("Frames per second"));
  m_memory = makeLabel(tr("Volume memory usage"));

  // Right-to-left: memory | fps | lod | zoom || value | cursor
  addPermanentWidget(m_cursor);
  addPermanentWidget(makeSep());
  addPermanentWidget(m_value);
  addPermanentWidget(makeSep());
  addPermanentWidget(m_zoom);
  addPermanentWidget(makeSep());
  addPermanentWidget(m_lod);
  addPermanentWidget(makeSep());
  addPermanentWidget(m_fps);
  addPermanentWidget(makeSep());
  addPermanentWidget(m_memory);

  // Initial placeholder text
  setCursor(0, 0, 0, std::numeric_limits<float>::quiet_NaN());
  setZoom(1.0f);
  setLod(0);
  setFps(0.0f);
  setMemoryMB(0);
}

void StatusBar::setCursor(float x, float y, float z, float value) {
  m_cursor->setText(QString::asprintf("(%.1f, %.1f, %.1f)", x, y, z));
  if (std::isnan(value))
    m_value->setText("val: —");
  else
    m_value->setText(QString::asprintf("val: %.3g", value));
}

void StatusBar::setZoom(float zoom) {
  m_zoom->setText(QString::asprintf("×%.2f", zoom));
}

void StatusBar::setLod(int lod) {
  if (lod == 0)
    m_lod->setText("LOD: full");
  else
    m_lod->setText(QString::asprintf("LOD: 1/%d", 1 << lod));
}

void StatusBar::setFps(float fps) {
  m_fps->setText(QString::asprintf("%.0f fps", fps));
}

void StatusBar::setMemoryMB(int mb) {
  if (mb < 1024)
    m_memory->setText(QString::asprintf("%d MB", mb));
  else
    m_memory->setText(QString::asprintf("%.1f GB", mb / 1024.0));
}
