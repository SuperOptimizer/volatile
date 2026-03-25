#pragma once

#include <QDialog>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QString>
#include <QVector>

class QLineEdit;
class QTreeView;
class QPushButton;
class QStatusBar;
class QLabel;
class QSplitter;
class QListWidget;
class QFutureWatcher;

// ---------------------------------------------------------------------------
// S3Item — one row in the S3 tree model (prefix/"folder" or object)
// ---------------------------------------------------------------------------

struct S3Item {
  QString name;      // display name (last path component)
  QString fullKey;   // full S3 key or prefix
  bool    isPrefix;  // true = "folder" (prefix), false = object
  qint64  sizeBytes; // 0 for prefixes
};

// ---------------------------------------------------------------------------
// S3Model — QAbstractItemModel backed by s3_list_objects from core/net.h.
// Provides a flat list for the current prefix; parent() is always the root.
// ---------------------------------------------------------------------------

class S3Model : public QAbstractItemModel {
  Q_OBJECT

public:
  explicit S3Model(QObject *parent = nullptr);

  // Set credentials before calling refresh().
  void setAccessKey(const QString &key);
  void setSecretKey(const QString &key);
  void setRegion(const QString &region);
  void setEndpoint(const QString &endpoint);

  // Fetch objects under bucket/prefix on a worker thread.
  void refresh(const QString &bucket, const QString &prefix);

  // Current listing
  const QVector<S3Item> &items() const { return m_items; }
  QString currentBucket() const { return m_bucket; }
  QString currentPrefix() const { return m_prefix; }
  bool    isLoading()     const { return m_loading; }

  // QAbstractItemModel interface
  QModelIndex   index(int row, int column,
                      const QModelIndex &parent = {}) const override;
  QModelIndex   parent(const QModelIndex &child) const override;
  int           rowCount(const QModelIndex &parent = {}) const override;
  int           columnCount(const QModelIndex &parent = {}) const override;
  QVariant      data(const QModelIndex &index,
                     int role = Qt::DisplayRole) const override;
  QVariant      headerData(int section, Qt::Orientation orientation,
                           int role = Qt::DisplayRole) const override;

signals:
  void loadingChanged(bool loading);
  void loadError(const QString &message);

private slots:
  void onWorkerFinished();

private:
  enum Column { ColName = 0, ColSize, ColCount };

  QString         m_accessKey;
  QString         m_secretKey;
  QString         m_region;
  QString         m_endpoint;
  QString         m_bucket;
  QString         m_prefix;
  QVector<S3Item> m_items;
  bool            m_loading = false;

  // Background fetch state
  struct FetchResult {
    QVector<S3Item> items;
    QString         error;
  };
  QFutureWatcher<FetchResult> *m_watcher = nullptr;
};

// ---------------------------------------------------------------------------
// S3BrowserDialog — QDialog that wraps S3Model in a tree view with controls.
// Replaces the Nuklear s3_browser from src/gui/s3_browser.h.
// ---------------------------------------------------------------------------

class S3BrowserDialog : public QDialog {
  Q_OBJECT

public:
  explicit S3BrowserDialog(QWidget *parent = nullptr);
  ~S3BrowserDialog() override = default;

  // Pre-fill credentials (e.g. from env or credential dialog).
  void setAccessKey(const QString &key);
  void setSecretKey(const QString &key);
  void setRegion(const QString &region);
  void setEndpoint(const QString &endpoint);

  // Returns the selected S3 URL in s3://bucket/key form.
  // Valid after exec() returns QDialog::Accepted.
  QString selectedUrl() const;

  // Bookmarks
  void addBookmark(const QString &name, const QString &url);

signals:
  void urlSelected(const QString &url);

private slots:
  void onConnectClicked();
  void onNavigateUp();
  void onItemActivated(const QModelIndex &index);
  void onSelectClicked();
  void onLoadingChanged(bool loading);
  void onLoadError(const QString &message);
  void onBookmarkActivated();

private:
  void navigateTo(const QString &prefix);
  void updatePathBar();
  void buildCredentialGroup(QWidget *parent, QVBoxLayout *layout);
  void buildBrowserSection(QWidget *parent, QVBoxLayout *layout);

  // Credential controls
  QLineEdit *m_bucketEdit;
  QLineEdit *m_accessKeyEdit;
  QLineEdit *m_secretKeyEdit;
  QLineEdit *m_regionEdit;
  QLineEdit *m_endpointEdit;

  // Browser controls
  QLabel      *m_pathLabel;
  QTreeView   *m_treeView;
  QPushButton *m_upButton;
  QPushButton *m_connectButton;
  QPushButton *m_selectButton;
  QListWidget *m_bookmarkList;

  // Status
  QLabel *m_statusLabel;

  S3Model *m_model;
  QString  m_selectedUrl;
};
