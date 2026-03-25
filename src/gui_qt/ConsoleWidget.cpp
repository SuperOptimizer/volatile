#include "ConsoleWidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QToolBar>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QMetaObject>
#include <QThread>

// ---------------------------------------------------------------------------
// Level metadata
// ---------------------------------------------------------------------------
namespace {

struct LevelMeta {
  const char *name;
  const char *color; // CSS colour for the message text
};

constexpr LevelMeta kLevels[] = {
  { "DEBUG", "#808080" },
  { "INFO",  "#d4d4d4" },
  { "WARN",  "#e5c07b" },
  { "ERROR", "#e06c75" },
  { "FATAL", "#ff0000" },
};

constexpr int kLevelCount = static_cast<int>(std::size(kLevels));

const char *levelColor(int level) {
  if (level < 0 || level >= kLevelCount) return "#d4d4d4";
  return kLevels[level].color;
}

const char *levelName(int level) {
  if (level < 0 || level >= kLevelCount) return "?";
  return kLevels[level].name;
}

} // namespace

// ---------------------------------------------------------------------------
// ConsoleWidget
// ---------------------------------------------------------------------------
ConsoleWidget::ConsoleWidget(QWidget *parent)
  : QDockWidget(tr("Console"), parent)
{
  setObjectName("ConsoleWidget");

  auto *container = new QWidget(this);
  auto *vlay = new QVBoxLayout(container);
  vlay->setContentsMargins(2, 2, 2, 2);
  vlay->setSpacing(2);

  // --- Toolbar row ---
  auto *hlay = new QHBoxLayout;
  hlay->setContentsMargins(0, 0, 0, 0);

  hlay->addWidget(new QLabel(tr("Min level:"), container));

  m_levelFilter = new QComboBox(container);
  for (int i = 0; i < kLevelCount; ++i)
    m_levelFilter->addItem(QString::fromLatin1(kLevels[i].name));
  m_levelFilter->setCurrentIndex(m_minLevel);
  hlay->addWidget(m_levelFilter);

  hlay->addStretch();

  auto *clearBtn = new QPushButton(tr("Clear"), container);
  hlay->addWidget(clearBtn);

  vlay->addLayout(hlay);

  // --- Log text ---
  m_text = new QPlainTextEdit(container);
  m_text->setReadOnly(true);
  m_text->setMaximumBlockCount(5000); // cap memory
  m_text->setWordWrapMode(QTextOption::NoWrap);

  // Dark terminal palette
  QPalette pal = m_text->palette();
  pal.setColor(QPalette::Base, QColor(0x1e, 0x1e, 0x1e));
  pal.setColor(QPalette::Text, QColor(0xd4, 0xd4, 0xd4));
  m_text->setPalette(pal);
  m_text->setFont(QFont("Monospace", 9));

  vlay->addWidget(m_text, 1);
  setWidget(container);

  connect(clearBtn,      &QPushButton::clicked,
          this,          &ConsoleWidget::clear);
  connect(m_levelFilter, qOverload<int>(&QComboBox::currentIndexChanged),
          this,          &ConsoleWidget::onLevelFilterChanged);
}

ConsoleWidget::~ConsoleWidget() {
  removeLogCallback();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void ConsoleWidget::addMessage(int level, const char *file, int line, const char *msg) {
  if (level < m_minLevel) return;

  // Format: [LEVEL] file:line  message
  const QString text = QString::asprintf(
      "<span style='color:%s'>[%s] %s:%d &nbsp; %s</span>",
      levelColor(level), levelName(level),
      file ? file : "", line, msg ? msg : "");

  // May be called from any thread — dispatch to GUI thread.
  if (QThread::currentThread() == thread()) {
    appendHtml(text);
  } else {
    QMetaObject::invokeMethod(this, "appendHtml",
                              Qt::QueuedConnection,
                              Q_ARG(QString, text));
  }
}

void ConsoleWidget::clear() {
  m_text->clear();
}

void ConsoleWidget::setMinLevel(int level) {
  m_minLevel = level;
  if (m_levelFilter && m_levelFilter->currentIndex() != level)
    m_levelFilter->setCurrentIndex(level);
}

void ConsoleWidget::installLogCallback() {
  log_set_callback(logCallbackEntry, this);
}

void ConsoleWidget::removeLogCallback() {
  // Only remove if we are the current holder.
  log_set_callback(nullptr, nullptr);
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------
void ConsoleWidget::appendHtml(const QString &html) {
  // Use appendHtml on the underlying document via the cursor for speed.
  QTextCursor cursor = m_text->textCursor();
  cursor.movePosition(QTextCursor::End);
  cursor.insertHtml(html);
  cursor.insertText("\n");

  // Auto-scroll if already at bottom.
  QScrollBar *sb = m_text->verticalScrollBar();
  if (sb->value() >= sb->maximum() - 4)
    sb->setValue(sb->maximum());
}

void ConsoleWidget::onLevelFilterChanged(int index) {
  m_minLevel = index;
}

void ConsoleWidget::logCallbackEntry(void *ctx, log_level_t level,
                                     const char *file, int line,
                                     const char *msg) {
  auto *self = static_cast<ConsoleWidget *>(ctx);
  self->addMessage(static_cast<int>(level), file, line, msg);
}
