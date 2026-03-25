#include "KeybindManager.h"
#include <QList>
#include <algorithm>

KeybindManager::KeybindManager(QObject *parent) : QObject(parent) {}

void KeybindManager::registerAction(const QString &id, const QKeySequence &defaultKey,
                                    const QString &description) {
  if (m_entries.contains(id)) return;  // already registered; don't clobber
  m_entries.insert(id, KeybindEntry{defaultKey, defaultKey, description});
}

void KeybindManager::setShortcut(const QString &id, const QKeySequence &key) {
  auto it = m_entries.find(id);
  if (it == m_entries.end()) return;
  it->currentKey = key;
  emit shortcutChanged(id, key);
}

QKeySequence KeybindManager::shortcut(const QString &id) const {
  auto it = m_entries.constFind(id);
  return it != m_entries.constEnd() ? it->currentKey : QKeySequence{};
}

QKeySequence KeybindManager::defaultShortcut(const QString &id) const {
  auto it = m_entries.constFind(id);
  return it != m_entries.constEnd() ? it->defaultKey : QKeySequence{};
}

QString KeybindManager::description(const QString &id) const {
  auto it = m_entries.constFind(id);
  return it != m_entries.constEnd() ? it->description : QString{};
}

QStringList KeybindManager::actionIds() const {
  QStringList ids = m_entries.keys();
  std::sort(ids.begin(), ids.end());
  return ids;
}

QAction *KeybindManager::createAction(const QString &id, QObject *parent) {
  auto it = m_entries.constFind(id);
  if (it == m_entries.constEnd()) return nullptr;

  auto *action = new QAction(it->description, parent);
  action->setShortcut(it->currentKey);
  action->setObjectName(id);

  // Forward triggered() as actionTriggered(id)
  connect(action, &QAction::triggered, this, [this, id]() {
    emit actionTriggered(id);
  });
  // Keep shortcut in sync when it changes
  connect(this, &KeybindManager::shortcutChanged, action,
          [action, id](const QString &changedId, const QKeySequence &key) {
    if (changedId == id) action->setShortcut(key);
  });
  return action;
}

void KeybindManager::resetToDefaults() {
  for (auto &e : m_entries) e.currentKey = e.defaultKey;
}

void KeybindManager::loadFromSettings(QSettings &s) {
  s.beginGroup(QStringLiteral("keybinds"));
  for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
    if (s.contains(it.key())) {
      QString raw = s.value(it.key()).toString();
      it->currentKey = QKeySequence(raw, QKeySequence::PortableText);
    }
  }
  s.endGroup();
}

void KeybindManager::saveToSettings(QSettings &s) const {
  s.beginGroup(QStringLiteral("keybinds"));
  for (auto it = m_entries.constBegin(); it != m_entries.constEnd(); ++it)
    s.setValue(it.key(), it->currentKey.toString(QKeySequence::PortableText));
  s.endGroup();
}

// ---------------------------------------------------------------------------
// VC3D full shortcut set
// ---------------------------------------------------------------------------

void KeybindManager::registerDefaultActions(KeybindManager &m) {
  using QKS = QKeySequence;
  // --- File ---
  m.registerAction("file.open",        QKS::Open,                 "Open volume");
  m.registerAction("file.save",        QKS::Save,                 "Save project");
  m.registerAction("file.save_as",     QKS::SaveAs,               "Save project as…");
  m.registerAction("file.close",       QKS::Close,                "Close volume");
  m.registerAction("file.quit",        QKS::Quit,                 "Quit");

  // --- Edit ---
  m.registerAction("edit.undo",        QKS::Undo,                 "Undo");
  m.registerAction("edit.redo",        QKS::Redo,                 "Redo");
  m.registerAction("edit.copy",        QKS::Copy,                 "Copy");
  m.registerAction("edit.paste",       QKS::Paste,                "Paste");
  m.registerAction("edit.delete",      QKS::Delete,               "Delete");
  m.registerAction("edit.select_all",  QKS::SelectAll,            "Select all");
  m.registerAction("edit.preferences", QKS::Preferences,          "Preferences");

  // --- View ---
  m.registerAction("view.zoom_in",     QKS{Qt::Key_Plus},         "Zoom in");
  m.registerAction("view.zoom_out",    QKS{Qt::Key_Minus},        "Zoom out");
  m.registerAction("view.zoom_reset",  QKS{Qt::Key_0},            "Reset zoom");
  m.registerAction("view.fit",         QKS{Qt::Key_F},            "Fit view to data");
  m.registerAction("view.fullscreen",  QKS{Qt::Key_F11},          "Fullscreen");

  // --- Navigation ---
  m.registerAction("nav.pan_left",     QKS{Qt::Key_Left},         "Pan left");
  m.registerAction("nav.pan_right",    QKS{Qt::Key_Right},        "Pan right");
  m.registerAction("nav.pan_up",       QKS{Qt::Key_Up},           "Pan up");
  m.registerAction("nav.pan_down",     QKS{Qt::Key_Down},         "Pan down");
  m.registerAction("nav.slice_next",   QKS{Qt::Key_Period},       "Next slice");
  m.registerAction("nav.slice_prev",   QKS{Qt::Key_Comma},        "Previous slice");
  m.registerAction("nav.slice_first",  QKS{Qt::Key_Home},         "First slice");
  m.registerAction("nav.slice_last",   QKS{Qt::Key_End},          "Last slice");
  m.registerAction("nav.back",         QKS::Back,                 "Navigate back");
  m.registerAction("nav.forward",      QKS::Forward,              "Navigate forward");

  // --- Tools ---
  m.registerAction("tool.brush",       QKS{Qt::Key_B},            "Brush tool");
  m.registerAction("tool.eraser",      QKS{Qt::Key_E},            "Eraser tool");
  m.registerAction("tool.line",        QKS{Qt::Key_L},            "Line tool");
  m.registerAction("tool.pushpull",    QKS{Qt::Key_P},            "Push-pull tool");
  m.registerAction("tool.pan",         QKS{Qt::Key_Space},        "Pan (hold)");
  m.registerAction("tool.measure",     QKS{Qt::Key_M},            "Measure tool");
  m.registerAction("tool.grow",        QKS{Qt::Key_G},            "Grow segment");
  m.registerAction("tool.annotation",  QKS{Qt::Key_A},            "Annotation tool");

  // --- Brush size ---
  m.registerAction("brush.size_up",    QKS{Qt::Key_BracketRight}, "Brush size up");
  m.registerAction("brush.size_down",  QKS{Qt::Key_BracketLeft},  "Brush size down");

  // --- Windows/panels ---
  m.registerAction("panel.seg",        QKS{Qt::CTRL | Qt::Key_1}, "Toggle segmentation panel");
  m.registerAction("panel.volume",     QKS{Qt::CTRL | Qt::Key_2}, "Toggle volume browser");
  m.registerAction("panel.3d",         QKS{Qt::CTRL | Qt::Key_3}, "Toggle 3D view");
  m.registerAction("panel.console",    QKS{Qt::CTRL | Qt::Key_QuoteLeft}, "Toggle console");

  // --- Visibility ---
  m.registerAction("vis.toggle_seg",   QKS{Qt::Key_S},            "Toggle segment overlay");
  m.registerAction("vis.toggle_annot", QKS{Qt::Key_N},            "Toggle annotations");
  m.registerAction("vis.toggle_grid",  QKS{Qt::Key_Slash},        "Toggle grid");

  // --- Rendering ---
  m.registerAction("render.mip",       QKS{Qt::CTRL | Qt::Key_M}, "MIP rendering");
  m.registerAction("render.iso",       QKS{Qt::CTRL | Qt::Key_I}, "Iso-surface rendering");
  m.registerAction("render.alpha",     QKS{Qt::CTRL | Qt::Key_A}, "Alpha compositing");

  // --- Help ---
  m.registerAction("help.about",       QKS::HelpContents,         "About Volatile");
  m.registerAction("help.keybinds",    QKS{Qt::Key_F1},           "Show keybindings");
}
