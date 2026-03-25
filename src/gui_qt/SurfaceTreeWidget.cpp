#include "SurfaceTreeWidget.h"

#include <QHeaderView>
#include <QMenu>
#include <QAction>
#include <QInputDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QContextMenuEvent>

// ---------------------------------------------------------------------------
// SurfaceTreeWidget
// ---------------------------------------------------------------------------

SurfaceTreeWidget::SurfaceTreeWidget(QWidget *parent)
    : QTreeWidget(parent) {
  setColumnCount(ColCount);
  setHeaderLabels({tr("Name"), tr("ID"), tr("Area"), tr("Status")});

  header()->setStretchLastSection(false);
  header()->setSectionResizeMode(ColName,   QHeaderView::Stretch);
  header()->setSectionResizeMode(ColId,     QHeaderView::ResizeToContents);
  header()->setSectionResizeMode(ColArea,   QHeaderView::ResizeToContents);
  header()->setSectionResizeMode(ColStatus, QHeaderView::ResizeToContents);

  setSelectionMode(QAbstractItemView::SingleSelection);
  setRootIsDecorated(false);
  setAlternatingRowColors(true);
  setUniformRowHeights(true);
  setSortingEnabled(true);
  sortByColumn(ColId, Qt::AscendingOrder);

  connect(this, &QTreeWidget::itemChanged,
          this, &SurfaceTreeWidget::onItemChanged);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void SurfaceTreeWidget::addSurface(const SurfaceEntry &entry) {
  auto *item = new QTreeWidgetItem(this);
  item->setData(ColId, Qt::UserRole, static_cast<qlonglong>(entry.id));
  populateItem(item, entry);
}

void SurfaceTreeWidget::removeSurface(int64_t id) {
  auto *item = itemForId(id);
  if (item)
    delete item;
}

void SurfaceTreeWidget::updateSurface(const SurfaceEntry &entry) {
  auto *item = itemForId(entry.id);
  if (!item)
    return;
  m_updatingItem = true;
  populateItem(item, entry);
  m_updatingItem = false;
}

void SurfaceTreeWidget::clearSurfaces() {
  clear();
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

int64_t SurfaceTreeWidget::selectedId() const {
  const auto items = selectedItems();
  if (items.isEmpty())
    return -1;
  return items.first()->data(ColId, Qt::UserRole).toLongLong();
}

bool SurfaceTreeWidget::isSurfaceVisible(int64_t id) const {
  const auto *item = itemForId(id);
  if (!item)
    return false;
  return item->checkState(ColName) == Qt::Checked;
}

// ---------------------------------------------------------------------------
// Mouse / context menu
// ---------------------------------------------------------------------------

void SurfaceTreeWidget::mouseDoubleClickEvent(QMouseEvent *event) {
  QTreeWidget::mouseDoubleClickEvent(event);
  const int64_t id = selectedId();
  if (id != -1)
    emit surfaceNavigationRequested(id);
}

void SurfaceTreeWidget::contextMenuEvent(QContextMenuEvent *event) {
  const int64_t id = selectedId();
  if (id == -1) {
    QTreeWidget::contextMenuEvent(event);
    return;
  }

  QMenu menu(this);
  auto *renameAct = menu.addAction(tr("Rename..."));
  auto *exportAct = menu.addAction(tr("Export..."));
  menu.addSeparator();
  auto *deleteAct = menu.addAction(tr("Delete"));
  deleteAct->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));

  connect(renameAct, &QAction::triggered, this, &SurfaceTreeWidget::onRenameAction);
  connect(exportAct, &QAction::triggered, this, &SurfaceTreeWidget::onExportAction);
  connect(deleteAct, &QAction::triggered, this, &SurfaceTreeWidget::onDeleteAction);

  menu.exec(event->globalPos());
}

// ---------------------------------------------------------------------------
// Private slots
// ---------------------------------------------------------------------------

void SurfaceTreeWidget::onItemChanged(QTreeWidgetItem *item, int column) {
  if (m_updatingItem || column != ColName)
    return;
  const int64_t id      = item->data(ColId, Qt::UserRole).toLongLong();
  const bool    visible = (item->checkState(ColName) == Qt::Checked);
  emit surfaceVisibilityChanged(id, visible);
}

void SurfaceTreeWidget::onRenameAction() {
  const int64_t id = selectedId();
  if (id == -1)
    return;
  auto *item = itemForId(id);
  const QString current = item ? item->text(ColName) : QString{};
  bool ok = false;
  const QString newName = QInputDialog::getText(
      this, tr("Rename Surface"), tr("New name:"), QLineEdit::Normal, current, &ok);
  if (ok && !newName.isEmpty()) {
    if (item) {
      m_updatingItem = true;
      item->setText(ColName, newName);
      m_updatingItem = false;
    }
    emit surfaceRenameRequested(id, newName);
  }
}

void SurfaceTreeWidget::onDeleteAction() {
  const int64_t id = selectedId();
  if (id == -1)
    return;
  const int ret = QMessageBox::question(
      this, tr("Delete Surface"),
      tr("Delete surface %1? This cannot be undone.").arg(id),
      QMessageBox::Yes | QMessageBox::No);
  if (ret == QMessageBox::Yes)
    emit surfaceDeleteRequested(id);
}

void SurfaceTreeWidget::onExportAction() {
  const int64_t id = selectedId();
  if (id != -1)
    emit surfaceExportRequested(id);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QTreeWidgetItem *SurfaceTreeWidget::itemForId(int64_t id) const {
  for (int i = 0; i < topLevelItemCount(); ++i) {
    auto *item = topLevelItem(i);
    if (item->data(ColId, Qt::UserRole).toLongLong() == static_cast<qlonglong>(id))
      return item;
  }
  return nullptr;
}

void SurfaceTreeWidget::populateItem(QTreeWidgetItem *item,
                                     const SurfaceEntry &entry) const {
  item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
  item->setCheckState(ColName, entry.visible ? Qt::Checked : Qt::Unchecked);
  item->setText(ColName, entry.name);
  item->setText(ColId,   QString::number(entry.id));
  // Show area in cm^2 if available, otherwise voxels^2.
  if (entry.areaCm2 > 0.0f)
    item->setText(ColArea, QStringLiteral("%1 cm²").arg(
        static_cast<double>(entry.areaCm2), 0, 'f', 2));
  else
    item->setText(ColArea, QStringLiteral("%1 vx²").arg(
        static_cast<double>(entry.areaVx2), 0, 'f', 0));
  item->setText(ColStatus, entry.approved ? tr("Approved") : tr("Pending"));
  item->setForeground(ColStatus,
      entry.approved ? QColor(Qt::darkGreen) : QColor(Qt::darkYellow));
}
