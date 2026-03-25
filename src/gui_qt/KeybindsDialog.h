#pragma once

#include <QDialog>

class QTableWidget;
class KeybindManager;

class KeybindsDialog : public QDialog {
  Q_OBJECT
public:
  explicit KeybindsDialog(KeybindManager *mgr, QWidget *parent = nullptr);

signals:
  void shortcutsChanged();

protected:
  void keyPressEvent(QKeyEvent *e) override;

private slots:
  void onCellClicked(int row, int col);
  void resetToDefaults();

private:
  void populate();
  void applyCapture(const QKeySequence &key);
  void clearCapture();

  KeybindManager *m_mgr;
  QTableWidget   *m_table;

  // capture state
  bool   m_capturing = false;
  int    m_captureRow = -1;
  QStringList m_ids;  // parallel to table rows
};
