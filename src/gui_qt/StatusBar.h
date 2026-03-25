#pragma once

#include <QStatusBar>

class QLabel;

// ---------------------------------------------------------------------------
// StatusBar — custom status bar with dedicated fields for viewer state.
//
// Usage:
//   auto *sb = new StatusBar(this);
//   setStatusBar(sb);
//   // per-frame:
//   sb->setCursor(x, y, z, value);
//   sb->setZoom(2.5f);
//   sb->setLod(0);
//   sb->setFps(60.0f);
//   sb->setMemoryMB(1024);
// ---------------------------------------------------------------------------
class StatusBar : public QStatusBar {
  Q_OBJECT
public:
  explicit StatusBar(QWidget *parent = nullptr);

  // Cursor world position and voxel intensity at that position.
  // Pass NaN for value to show "—".
  void setCursor(float x, float y, float z, float value);

  // Current zoom factor (1.0 = 1:1).
  void setZoom(float zoom);

  // Current level-of-detail tier (0 = full resolution).
  void setLod(int lod);

  // Frames per second.
  void setFps(float fps);

  // GPU/CPU memory used by volume data (megabytes).
  void setMemoryMB(int mb);

private:
  QLabel *m_cursor  = nullptr;
  QLabel *m_value   = nullptr;
  QLabel *m_zoom    = nullptr;
  QLabel *m_lod     = nullptr;
  QLabel *m_fps     = nullptr;
  QLabel *m_memory  = nullptr;
};
