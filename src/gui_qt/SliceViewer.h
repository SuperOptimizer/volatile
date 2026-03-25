#pragma once

#include <QWidget>
#include <QImage>
#include <QPointF>

extern "C" {
#include "core/vol.h"
#include "render/cmap.h"
}

class SliceViewer : public QWidget {
  Q_OBJECT
public:
  enum Axis { XY = 0, XZ = 1, YZ = 2 };

  explicit SliceViewer(Axis axis, QWidget *parent = nullptr);

  void setVolume(volume *vol);
  void setSlice(float z);
  void setColormap(int cmapId);
  void setWindowLevel(float window, float level);

  float currentSlice() const;
  Axis  axis() const;

signals:
  void sliceChanged(float z);
  void cursorMoved(float x, float y, float z);  // 3D world position
  void clicked(float x, float y, float z);

protected:
  void paintEvent(QPaintEvent *e) override;
  void wheelEvent(QWheelEvent *e) override;
  void mousePressEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  void mouseReleaseEvent(QMouseEvent *e) override;

private:
  void   renderSlice();
  QPointF screenToSurface(QPoint screen) const;
  QPoint  surfaceToScreen(QPointF surface) const;

  // Maps a screen pixel to (x, y, z) volume coordinates given current axis/slice.
  void screenToVolume(QPoint screen, float &vx, float &vy, float &vz) const;

  Axis    m_axis;
  volume *m_vol    = nullptr;
  float   m_slice  = 0.0f;
  float   m_scale  = 1.0f;
  QPointF m_pan    = {0.0, 0.0};
  int     m_cmapId = 0;
  float   m_window = 1.0f;
  float   m_level  = 0.5f;
  QImage  m_image;   // cached rendered slice
  bool    m_dirty  = true;

  // Pan drag state
  bool   m_dragging   = false;
  QPoint m_dragOrigin;
  QPointF m_panAtDrag;
};
