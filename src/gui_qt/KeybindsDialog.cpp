#include "KeybindsDialog.h"
#include "KeybindManager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QLabel>
#include <QKeyEvent>
#include <QMessageBox>

// Column indices
static constexpr int COL_ACTION  = 0;
static constexpr int COL_CURRENT = 1;
static constexpr int COL_DEFAULT = 2;
static constexpr int N_COLS      = 3;

KeybindsDialog::KeybindsDialog(KeybindManager *mgr, QWidget *parent)
  : QDialog(parent), m_mgr(mgr) {
  setWindowTitle(tr("Keyboard Shortcuts"));
  resize(560, 480);

  auto *root = new QVBoxLayout(this);

  // Instructions
  auto *hint = new QLabel(
    tr("Click a cell in the <b>Current Key</b> column, then press a new key combination."));
  hint->setWordWrap(true);
  root->addWidget(hint);

  // Table
  m_table = new QTableWidget(0, N_COLS, this);
  m_table->setHorizontalHeaderLabels({tr("Action"), tr("Current Key"), tr("Default Key")});
  m_table->horizontalHeader()->setSectionResizeMode(COL_ACTION,  QHeaderView::Stretch);
  m_table->horizontalHeader()->setSectionResizeMode(COL_CURRENT, QHeaderView::ResizeToContents);
  m_table->horizontalHeader()->setSectionResizeMode(COL_DEFAULT, QHeaderView::ResizeToContents);
  m_table->setSelectionMode(QAbstractItemView::SingleSelection);
  m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  m_table->setAlternatingRowColors(true);
  m_table->verticalHeader()->hide();
  root->addWidget(m_table);

  connect(m_table, &QTableWidget::cellClicked, this, &KeybindsDialog::onCellClicked);

  // Button row
  auto *btn_row = new QHBoxLayout;
  auto *reset   = new QPushButton(tr("Reset to Defaults"));
  connect(reset, &QPushButton::clicked, this, &KeybindsDialog::resetToDefaults);
  btn_row->addWidget(reset);
  btn_row->addStretch();
  auto *ok = new QPushButton(tr("OK"));
  ok->setDefault(true);
  connect(ok, &QPushButton::clicked, this, &QDialog::accept);
  btn_row->addWidget(ok);
  auto *cancel = new QPushButton(tr("Cancel"));
  connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
  btn_row->addWidget(cancel);
  root->addLayout(btn_row);

  populate();
}

void KeybindsDialog::populate() {
  m_table->setRowCount(0);
  m_ids = m_mgr->actionIds();

  m_table->setRowCount(m_ids.size());
  for (int row = 0; row < m_ids.size(); ++row) {
    const QString &id = m_ids[row];
    auto mk = [](const QString &s) {
      auto *item = new QTableWidgetItem(s);
      item->setFlags(item->flags() & ~Qt::ItemIsEditable);
      return item;
    };
    m_table->setItem(row, COL_ACTION,  mk(m_mgr->description(id)));
    m_table->setItem(row, COL_CURRENT, mk(m_mgr->shortcut(id).toString(QKeySequence::NativeText)));
    m_table->setItem(row, COL_DEFAULT, mk(m_mgr->defaultShortcut(id).toString(QKeySequence::NativeText)));
  }
}

void KeybindsDialog::onCellClicked(int row, int col) {
  if (col != COL_CURRENT) { clearCapture(); return; }

  clearCapture();
  m_capturing   = true;
  m_captureRow  = row;

  auto *item = m_table->item(row, COL_CURRENT);
  if (item) {
    item->setText(tr("Press key…"));
    item->setBackground(QColor(255, 240, 180));
  }
  m_table->setFocus();
}

void KeybindsDialog::keyPressEvent(QKeyEvent *e) {
  if (!m_capturing) { QDialog::keyPressEvent(e); return; }

  // Ignore lone modifiers
  const Qt::Key key = static_cast<Qt::Key>(e->key());
  if (key == Qt::Key_Control || key == Qt::Key_Shift ||
      key == Qt::Key_Alt     || key == Qt::Key_Meta) {
    return;
  }
  // Escape cancels capture
  if (key == Qt::Key_Escape) { clearCapture(); return; }

  QKeySequence seq(e->keyCombination());
  applyCapture(seq);
  e->accept();
}

void KeybindsDialog::applyCapture(const QKeySequence &key) {
  if (m_captureRow < 0 || m_captureRow >= m_ids.size()) { clearCapture(); return; }

  const QString &id = m_ids[m_captureRow];
  m_mgr->setShortcut(id, key);

  auto *item = m_table->item(m_captureRow, COL_CURRENT);
  if (item) {
    item->setText(key.toString(QKeySequence::NativeText));
    item->setBackground(QColor(200, 240, 200));
  }
  clearCapture();
  emit shortcutsChanged();
}

void KeybindsDialog::clearCapture() {
  if (m_capturing && m_captureRow >= 0 && m_captureRow < m_ids.size()) {
    // Restore background if we didn't just set a key (i.e., escape was pressed
    // without applying — restore the current text)
    auto *item = m_table->item(m_captureRow, COL_CURRENT);
    if (item) {
      const QString &id = m_ids[m_captureRow];
      item->setText(m_mgr->shortcut(id).toString(QKeySequence::NativeText));
      item->setBackground(QColor());  // reset to default
    }
  }
  m_capturing  = false;
  m_captureRow = -1;
}

void KeybindsDialog::resetToDefaults() {
  auto btn = QMessageBox::question(this, tr("Reset Shortcuts"),
               tr("Reset all keyboard shortcuts to their defaults?"));
  if (btn != QMessageBox::Yes) return;

  m_mgr->resetToDefaults();
  populate();
  emit shortcutsChanged();
}
