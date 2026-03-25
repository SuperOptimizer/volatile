#pragma once

#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <cstdint>

// ---------------------------------------------------------------------------
// SurfaceTreeWidget — shows all loaded surfaces in a tree view.
// Columns: Name | ID | Area | Status
// Replaces the Nuklear surface_panel from src/gui/surface_panel.h.
// ---------------------------------------------------------------------------

struct SurfaceEntry {
  int64_t id;
  QString name;
  QString volumeId;
  float   areaVx2;   // area in voxels^2
  float   areaCm2;   // area in cm^2 (0 if voxel size unknown)
  bool    visible;
  bool    approved;
  int     rowCount;
  int     colCount;
};

class SurfaceTreeWidget : public QTreeWidget {
  Q_OBJECT

public:
  explicit SurfaceTreeWidget(QWidget *parent = nullptr);
  ~SurfaceTreeWidget() override = default;

  void addSurface(const SurfaceEntry &entry);
  void removeSurface(int64_t id);
  void updateSurface(const SurfaceEntry &entry);
  void clearSurfaces();

  int64_t selectedId() const;
  bool    isSurfaceVisible(int64_t id) const;

signals:
  void surfaceNavigationRequested(int64_t id);
  void surfaceVisibilityChanged(int64_t id, bool visible);
  void surfaceRenameRequested(int64_t id, const QString &newName);
  void surfaceDeleteRequested(int64_t id);
  void surfaceExportRequested(int64_t id);

protected:
  void mouseDoubleClickEvent(QMouseEvent *event) override;
  void contextMenuEvent(QContextMenuEvent *event) override;

private slots:
  void onItemChanged(QTreeWidgetItem *item, int column);
  void onRenameAction();
  void onDeleteAction();
  void onExportAction();

private:
  enum Column { ColName = 0, ColId, ColArea, ColStatus, ColCount };

  QTreeWidgetItem *itemForId(int64_t id) const;
  void populateItem(QTreeWidgetItem *item, const SurfaceEntry &entry) const;

  bool m_updatingItem = false;
};
