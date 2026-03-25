#pragma once

#include <QObject>
#include <QAction>
#include <QKeySequence>
#include <QSettings>
#include <QString>
#include <QHash>

struct KeybindEntry {
  QKeySequence defaultKey;
  QKeySequence currentKey;
  QString      description;
};

class KeybindManager : public QObject {
  Q_OBJECT
public:
  explicit KeybindManager(QObject *parent = nullptr);

  // Register an action with a default shortcut; safe to call multiple times.
  void registerAction(const QString &id, const QKeySequence &defaultKey,
                      const QString &description);

  void         setShortcut(const QString &id, const QKeySequence &key);
  QKeySequence shortcut(const QString &id) const;
  QKeySequence defaultShortcut(const QString &id) const;
  QString      description(const QString &id) const;

  // Returns sorted list of all registered action IDs.
  QStringList actionIds() const;

  // Create a QAction whose triggered() is connected to actionTriggered(id).
  QAction *createAction(const QString &id, QObject *parent = nullptr);

  void resetToDefaults();
  void loadFromSettings(QSettings &s);
  void saveToSettings(QSettings &s) const;

  // Register the full VC3D shortcut set.
  static void registerDefaultActions(KeybindManager &mgr);

signals:
  void actionTriggered(const QString &id);
  void shortcutChanged(const QString &id, const QKeySequence &key);

private:
  QHash<QString, KeybindEntry> m_entries;
};
