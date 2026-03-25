#pragma once

#include <QDockWidget>
#include <QPlainTextEdit>
#include <QComboBox>
#include <QToolBar>

extern "C" {
#include "core/log.h"
}

// ---------------------------------------------------------------------------
// ConsoleWidget — dockable log output, wired to the C log_callback system.
// ---------------------------------------------------------------------------
class ConsoleWidget : public QDockWidget {
  Q_OBJECT
public:
  explicit ConsoleWidget(QWidget *parent = nullptr);
  ~ConsoleWidget() override;

  // Add a single log entry.  Thread-safe: posts to the main thread if needed.
  void addMessage(int level, const char *file, int line, const char *msg);

  void clear();
  void setMinLevel(int level);

  // Install/remove this widget as the global log callback.
  void installLogCallback();
  void removeLogCallback();

private slots:
  void onLevelFilterChanged(int index);

private:
  // Append already-formatted HTML on the GUI thread.
  Q_INVOKABLE void appendHtml(const QString &html);

  static void logCallbackEntry(void *ctx, log_level_t level,
                               const char *file, int line, const char *msg);

  QPlainTextEdit *m_text  = nullptr;
  QComboBox      *m_levelFilter = nullptr;
  int             m_minLevel    = 0;
};
