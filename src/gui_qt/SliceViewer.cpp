#include "SliceViewer.h"

#include <QPainter>
#include <QPaintEvent>
#include <QWheelEvent>
#include <QMouseEvent>
#include <algorithm>
#include <cmath>

SliceViewer::SliceViewer(Axis axis, QWidget *parent)
  : QWidget(parent), m_axis(axis)
{
  setMouseTracking(true);
  setAttribute(Qt::WA_OpaquePaintEvent);
}

// ---------------------------------------------------------------------------
// Public setters
// ---------------------------------------------------------------------------

void SliceViewer::setVolume(volume *vol) {
  m_vol   = vol;
  m_dirty = true;
  update();
}

void SliceViewer::setSlice(float z) {
  if (z == m_slice) return;
  m_slice = z;
  m_dirty = true;
  update();
  emit sliceChanged(m_slice);
}

void SliceViewer::setColormap(int cmapId) {
  m_cmapId = cmapId;
  m_dirty  = true;
  update();
}

void SliceViewer::setWindowLevel(float window, float level) {
  m_window = window;
  m_level  = level;
  m_dirty  = true;
  update();
}

float SliceViewer::currentSlice() const { return m_slice; }
SliceViewer::Axis SliceViewer::axis() const { return m_axis; }

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

// Screen pixel → surface (image) coordinates (before volume mapping).
QPointF SliceViewer::screenToSurface(QPoint screen) const {
  return QPointF{
    (screen.x() - m_pan.x()) / m_scale,
    (screen.y() - m_pan.y()) / m_scale
  };
}

QPoint SliceViewer::surfaceToScreen(QPointF surface) const {
  return QPoint{
    static_cast<int>(std::round(surface.x() * m_scale + m_pan.x())),
    static_cast<int>(std::round(surface.y() * m_scale + m_pan.y()))
  };
}

// Screen pixel → (vx, vy, vz) volume coordinates for the current axis/slice.
// For XY axis: surface(col,row) → (x=col, y=row, z=slice)
// For XZ axis: surface(col,row) → (x=col, y=slice, z=row)
// For YZ axis: surface(col,row) → (x=slice, y=col, z=row)
void SliceViewer::screenToVolume(QPoint screen, float &vx, float &vy, float &vz) const {
  QPointF s = screenToSurface(screen);
  float col = static_cast<float>(s.x());
  float row = static_cast<float>(s.y());
  switch (m_axis) {
    case XY: vx = col; vy = row; vz = m_slice; break;
    case XZ: vx = col; vy = m_slice; vz = row; break;
    case YZ: vx = m_slice; vy = col; vz = row; break;
  }
}

// ---------------------------------------------------------------------------
// renderSlice — sample volume and build QImage
// ---------------------------------------------------------------------------

void SliceViewer::renderSlice() {
  const int W = width();
  const int H = height();
  if (W <= 0 || H <= 0) return;

  if (m_image.width() != W || m_image.height() != H)
    m_image = QImage(W, H, QImage::Format_RGB32);

  if (!m_vol) {
    m_image.fill(Qt::black);
    m_dirty = false;
    return;
  }

  // Window/level: normalise raw sample s → t ∈ [0,1]
  // t = (s - (level - window/2)) / window
  const float wl_min = m_level - m_window * 0.5f;
  const cmap_id cid  = static_cast<cmap_id>(
    std::clamp(m_cmapId, 0, static_cast<int>(CMAP_COUNT) - 1));

  for (int row = 0; row < H; ++row) {
    QRgb *line = reinterpret_cast<QRgb *>(m_image.scanLine(row));
    for (int col = 0; col < W; ++col) {
      // Map screen pixel to volume coordinates
      QPointF s = screenToSurface(QPoint{col, row});
      float fc = static_cast<float>(s.x());
      float fr = static_cast<float>(s.y());

      float vx, vy, vz;
      switch (m_axis) {
        case XY: vx = fc; vy = fr; vz = m_slice; break;
        case XZ: vx = fc; vy = m_slice; vz = fr; break;
        case YZ: vx = m_slice; vy = fc; vz = fr; break;
      }

      float raw = vol_sample(m_vol, 0, vz, vy, vx);
      // vol_sample returns value in [0, 255] range for uint8 volumes;
      // normalise to [0,1] before window/level.
      float norm = raw / 255.0f;
      float t    = (norm - wl_min) / m_window;
      t = std::clamp(t, 0.0f, 1.0f);

      cmap_rgb c = cmap_apply(cid, static_cast<double>(t));
      line[col]  = qRgb(c.r, c.g, c.b);
    }
  }
  m_dirty = false;
}

// ---------------------------------------------------------------------------
// paintEvent
// ---------------------------------------------------------------------------

void SliceViewer::paintEvent(QPaintEvent * /*e*/) {
  if (m_dirty) renderSlice();

  QPainter p(this);
  if (!m_image.isNull())
    p.drawImage(0, 0, m_image);
  else
    p.fillRect(rect(), Qt::black);
}

// ---------------------------------------------------------------------------
// wheelEvent — scroll = zoom (centred on cursor), Shift+scroll = slice step
// ---------------------------------------------------------------------------

void SliceViewer::wheelEvent(QWheelEvent *e) {
  const float delta = static_cast<float>(e->angleDelta().y());
  if (delta == 0.0f) return;

  if (e->modifiers() & Qt::ShiftModifier) {
    // Shift+wheel: scroll through slices
    float step = (delta > 0) ? 1.0f : -1.0f;
    setSlice(m_slice + step);
  } else {
    // Wheel: zoom centred on cursor
    const float factor = (delta > 0) ? 1.15f : (1.0f / 1.15f);
    QPointF cursor = e->position();
    // Adjust pan so the point under the cursor stays fixed:
    // new_pan = cursor - factor * (cursor - old_pan)
    m_pan   = cursor - factor * (cursor - m_pan);
    m_scale *= factor;
    m_dirty  = true;
    update();
  }
}

// ---------------------------------------------------------------------------
// Mouse: left drag = pan
// ---------------------------------------------------------------------------

void SliceViewer::mousePressEvent(QMouseEvent *e) {
  if (e->button() == Qt::LeftButton) {
    m_dragging   = true;
    m_dragOrigin = e->pos();
    m_panAtDrag  = m_pan;
    e->accept();
  }
}

void SliceViewer::mouseMoveEvent(QMouseEvent *e) {
  if (m_dragging && (e->buttons() & Qt::LeftButton)) {
    QPoint delta = e->pos() - m_dragOrigin;
    m_pan   = m_panAtDrag + QPointF(delta);
    m_dirty = true;
    update();
  }

  // Emit cursor world position for crosshair sync
  float vx = 0.0f, vy = 0.0f, vz = 0.0f;
  screenToVolume(e->pos(), vx, vy, vz);
  emit cursorMoved(vx, vy, vz);
}

void SliceViewer::mouseReleaseEvent(QMouseEvent *e) {
  if (e->button() == Qt::LeftButton && m_dragging) {
    m_dragging = false;
    // Emit click at release position if not a drag (within 4 px)
    QPoint delta = e->pos() - m_dragOrigin;
    if (std::abs(delta.x()) <= 4 && std::abs(delta.y()) <= 4) {
      float vx = 0.0f, vy = 0.0f, vz = 0.0f;
      screenToVolume(e->pos(), vx, vy, vz);
      emit clicked(vx, vy, vz);
    }
    e->accept();
  }
}
