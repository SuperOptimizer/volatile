#include "S3BrowserDialog.h"

// Core C net.h is plain C — use extern "C" to avoid name-mangling issues.
extern "C" {
#include "core/net.h"
}

#include <QLineEdit>
#include <QTreeView>
#include <QPushButton>
#include <QLabel>
#include <QListWidget>
#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QIcon>
#include <QStyle>
#include <QApplication>
#include <QMessageBox>

// ---------------------------------------------------------------------------
// XML parsing helper — pulls S3 ListBucketResult keys/prefixes.
// We parse the raw HTTP response body from s3_list_objects ourselves using
// a simple string-scan approach (avoids pulling in a full XML library).
// ---------------------------------------------------------------------------

static QVector<S3Item> parseS3Xml(const QByteArray &xml) {
  QVector<S3Item> result;

  // Extract all <Prefix> tags (common prefixes = "folders")
  int pos = 0;
  while (true) {
    const int open  = xml.indexOf("<CommonPrefixes>", pos);
    if (open < 0) break;
    const int start = xml.indexOf("<Prefix>", open) + 8;
    const int end   = xml.indexOf("</Prefix>", start);
    if (start < 8 || end < 0) break;
    const QString full = QString::fromUtf8(xml.mid(start, end - start));
    const QString name = full.endsWith('/') ? full.chopped(1).section('/', -1) + '/'
                                            : full.section('/', -1);
    result.push_back({name, full, true, 0});
    pos = end + 9;
  }

  // Extract all <Contents> (objects)
  pos = 0;
  while (true) {
    const int open = xml.indexOf("<Contents>", pos);
    if (open < 0) break;
    const int close = xml.indexOf("</Contents>", open);
    if (close < 0) break;

    const QByteArray block = xml.mid(open, close - open);

    const int ks = block.indexOf("<Key>") + 5;
    const int ke = block.indexOf("</Key>", ks);
    const int ss = block.indexOf("<Size>") + 6;
    const int se = block.indexOf("</Size>", ss);

    if (ks >= 5 && ke > ks) {
      const QString key  = QString::fromUtf8(block.mid(ks, ke - ks));
      const qint64  size = (ss >= 6 && se > ss)
                               ? block.mid(ss, se - ss).toLongLong()
                               : 0;
      const QString name = key.section('/', -1);
      if (!name.isEmpty())
        result.push_back({name, key, false, size});
    }
    pos = close + 11;
  }

  return result;
}

// ---------------------------------------------------------------------------
// S3Model
// ---------------------------------------------------------------------------

S3Model::S3Model(QObject *parent) : QAbstractItemModel(parent) {}

void S3Model::setAccessKey(const QString &key) { m_accessKey = key; }
void S3Model::setSecretKey(const QString &key) { m_secretKey = key; }
void S3Model::setRegion(const QString &region) { m_region    = region; }
void S3Model::setEndpoint(const QString &ep)   { m_endpoint  = ep; }

void S3Model::refresh(const QString &bucket, const QString &prefix) {
  if (m_loading) return;
  m_bucket  = bucket;
  m_prefix  = prefix;
  m_loading = true;
  emit loadingChanged(true);

  // Capture credentials for the worker lambda.
  const QString accessKey = m_accessKey;
  const QString secretKey = m_secretKey;
  const QString region    = m_region;
  const QString endpoint  = m_endpoint;
  const QString bkt       = bucket;
  const QString pfx       = prefix;

  auto *watcher = new QFutureWatcher<FetchResult>(this);
  m_watcher     = watcher;
  connect(watcher, &QFutureWatcher<FetchResult>::finished,
          this,    &S3Model::onWorkerFinished);

  const QFuture<FetchResult> future = QtConcurrent::run([=]() -> FetchResult {
    s3_credentials creds{};
    qstrncpy(creds.access_key, accessKey.toUtf8().constData(),
             sizeof(creds.access_key) - 1);
    qstrncpy(creds.secret_key, secretKey.toUtf8().constData(),
             sizeof(creds.secret_key) - 1);
    qstrncpy(creds.region,     region.toUtf8().constData(),
             sizeof(creds.region) - 1);
    qstrncpy(creds.endpoint,   endpoint.toUtf8().constData(),
             sizeof(creds.endpoint) - 1);

    http_response *resp = s3_list_objects(&creds, bkt.toUtf8().constData(),
                                          pfx.toUtf8().constData(), 10000);
    FetchResult result;
    if (!resp) {
      result.error = QStringLiteral("Network error (null response)");
      return result;
    }
    if (resp->status_code != 200) {
      result.error = QStringLiteral("HTTP %1: %2")
                         .arg(resp->status_code)
                         .arg(resp->error ? QString::fromUtf8(resp->error)
                                          : QStringLiteral("unknown"));
      http_response_free(resp);
      return result;
    }
    const QByteArray body(reinterpret_cast<const char *>(resp->data),
                          static_cast<qsizetype>(resp->size));
    http_response_free(resp);
    result.items = parseS3Xml(body);
    return result;
  });

  watcher->setFuture(future);
}

void S3Model::onWorkerFinished() {
  auto *watcher = m_watcher;
  m_watcher     = nullptr;
  m_loading     = false;

  const FetchResult res = watcher->result();
  watcher->deleteLater();

  if (!res.error.isEmpty()) {
    emit loadError(res.error);
    emit loadingChanged(false);
    return;
  }

  beginResetModel();
  m_items = res.items;
  endResetModel();
  emit loadingChanged(false);
}

QModelIndex S3Model::index(int row, int column,
                            const QModelIndex &parent) const {
  if (parent.isValid() || row < 0 || row >= m_items.size()
      || column < 0 || column >= ColCount)
    return {};
  return createIndex(row, column);
}

QModelIndex S3Model::parent(const QModelIndex & /*child*/) const {
  return {};
}

int S3Model::rowCount(const QModelIndex &parent) const {
  return parent.isValid() ? 0 : static_cast<int>(m_items.size());
}

int S3Model::columnCount(const QModelIndex & /*parent*/) const {
  return ColCount;
}

QVariant S3Model::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || index.row() >= m_items.size())
    return {};
  const S3Item &item = m_items.at(index.row());

  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case ColName: return item.name;
      case ColSize:
        if (item.isPrefix) return tr("—");
        if (item.sizeBytes < 1024)
          return QStringLiteral("%1 B").arg(item.sizeBytes);
        if (item.sizeBytes < 1024 * 1024)
          return QStringLiteral("%1 KB").arg(item.sizeBytes / 1024);
        return QStringLiteral("%1 MB").arg(item.sizeBytes / (1024 * 1024));
      default: return {};
    }
  }

  if (role == Qt::DecorationRole && index.column() == ColName) {
    const auto *style = QApplication::style();
    return item.isPrefix
               ? style->standardIcon(QStyle::SP_DirIcon)
               : style->standardIcon(QStyle::SP_FileIcon);
  }

  if (role == Qt::UserRole)
    return item.fullKey;

  return {};
}

QVariant S3Model::headerData(int section, Qt::Orientation orientation,
                              int role) const {
  if (orientation != Qt::Horizontal || role != Qt::DisplayRole)
    return {};
  switch (section) {
    case ColName: return tr("Name");
    case ColSize: return tr("Size");
    default:      return {};
  }
}

// ---------------------------------------------------------------------------
// S3BrowserDialog
// ---------------------------------------------------------------------------

S3BrowserDialog::S3BrowserDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("S3 Browser"));
  setMinimumSize(700, 480);
  resize(800, 560);

  m_model = new S3Model(this);
  connect(m_model, &S3Model::loadingChanged, this, &S3BrowserDialog::onLoadingChanged);
  connect(m_model, &S3Model::loadError,      this, &S3BrowserDialog::onLoadError);

  auto *mainLayout = new QVBoxLayout(this);

  // ---------- Credential group ----------
  auto *credGroup  = new QGroupBox(tr("Credentials"), this);
  auto *credLayout = new QFormLayout(credGroup);
  credLayout->setSpacing(4);

  m_bucketEdit    = new QLineEdit(credGroup);
  m_bucketEdit->setPlaceholderText(tr("my-bucket"));
  m_accessKeyEdit = new QLineEdit(credGroup);
  m_accessKeyEdit->setPlaceholderText(tr("AKIAIOSFODNN7EXAMPLE"));
  m_secretKeyEdit = new QLineEdit(credGroup);
  m_secretKeyEdit->setEchoMode(QLineEdit::Password);
  m_secretKeyEdit->setPlaceholderText(tr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"));
  m_regionEdit    = new QLineEdit(credGroup);
  m_regionEdit->setPlaceholderText(tr("us-east-1"));
  m_endpointEdit  = new QLineEdit(credGroup);
  m_endpointEdit->setPlaceholderText(tr("https://s3.amazonaws.com (optional)"));

  credLayout->addRow(tr("Bucket:"),         m_bucketEdit);
  credLayout->addRow(tr("Access Key:"),     m_accessKeyEdit);
  credLayout->addRow(tr("Secret Key:"),     m_secretKeyEdit);
  credLayout->addRow(tr("Region:"),         m_regionEdit);
  credLayout->addRow(tr("Endpoint URL:"),   m_endpointEdit);

  m_connectButton = new QPushButton(tr("Connect"), credGroup);
  credLayout->addRow(m_connectButton);
  mainLayout->addWidget(credGroup);

  // ---------- Path bar ----------
  auto *pathRow = new QHBoxLayout;
  m_upButton  = new QPushButton(
      style()->standardIcon(QStyle::SP_ArrowUp), {}, this);
  m_upButton->setToolTip(tr("Up one level"));
  m_upButton->setEnabled(false);
  m_pathLabel = new QLabel(tr("/"), this);
  pathRow->addWidget(m_upButton);
  pathRow->addWidget(m_pathLabel, 1);
  mainLayout->addLayout(pathRow);

  // ---------- Splitter: bookmark sidebar + tree ----------
  auto *splitter = new QSplitter(Qt::Horizontal, this);

  m_bookmarkList = new QListWidget(splitter);
  m_bookmarkList->setMaximumWidth(180);
  m_bookmarkList->setAlternatingRowColors(true);
  splitter->addWidget(m_bookmarkList);

  m_treeView = new QTreeView(splitter);
  m_treeView->setModel(m_model);
  m_treeView->setRootIsDecorated(false);
  m_treeView->setAlternatingRowColors(true);
  m_treeView->setSelectionMode(QAbstractItemView::SingleSelection);
  m_treeView->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  m_treeView->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  splitter->addWidget(m_treeView);
  splitter->setStretchFactor(1, 3);
  mainLayout->addWidget(splitter, 1);

  // ---------- Status + buttons ----------
  m_statusLabel = new QLabel(tr("Not connected"), this);
  mainLayout->addWidget(m_statusLabel);

  auto *buttonBox = new QDialogButtonBox(this);
  m_selectButton  = new QPushButton(tr("Select"), buttonBox);
  m_selectButton->setEnabled(false);
  buttonBox->addButton(m_selectButton, QDialogButtonBox::AcceptRole);
  buttonBox->addButton(QDialogButtonBox::Cancel);
  mainLayout->addWidget(buttonBox);

  // ---------- Connections ----------
  connect(m_connectButton, &QPushButton::clicked,
          this, &S3BrowserDialog::onConnectClicked);
  connect(m_upButton, &QPushButton::clicked,
          this, &S3BrowserDialog::onNavigateUp);
  connect(m_treeView, &QTreeView::activated,
          this, &S3BrowserDialog::onItemActivated);
  connect(m_treeView->selectionModel(), &QItemSelectionModel::selectionChanged,
          this, [this](const QItemSelection &sel, const QItemSelection &) {
    m_selectButton->setEnabled(!sel.isEmpty());
  });
  connect(m_selectButton, &QPushButton::clicked,
          this, &S3BrowserDialog::onSelectClicked);
  connect(m_bookmarkList, &QListWidget::itemActivated,
          this, &S3BrowserDialog::onBookmarkActivated);
  connect(buttonBox, &QDialogButtonBox::rejected,
          this, &QDialog::reject);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void S3BrowserDialog::setAccessKey(const QString &key) {
  m_accessKeyEdit->setText(key);
}
void S3BrowserDialog::setSecretKey(const QString &key) {
  m_secretKeyEdit->setText(key);
}
void S3BrowserDialog::setRegion(const QString &region) {
  m_regionEdit->setText(region);
}
void S3BrowserDialog::setEndpoint(const QString &endpoint) {
  m_endpointEdit->setText(endpoint);
}

QString S3BrowserDialog::selectedUrl() const {
  return m_selectedUrl;
}

void S3BrowserDialog::addBookmark(const QString &name, const QString &url) {
  auto *item = new QListWidgetItem(name, m_bookmarkList);
  item->setData(Qt::UserRole, url);
}

// ---------------------------------------------------------------------------
// Private slots
// ---------------------------------------------------------------------------

void S3BrowserDialog::onConnectClicked() {
  const QString bucket = m_bucketEdit->text().trimmed();
  if (bucket.isEmpty()) {
    m_statusLabel->setText(tr("Please enter a bucket name."));
    return;
  }

  m_model->setAccessKey(m_accessKeyEdit->text().trimmed());
  m_model->setSecretKey(m_secretKeyEdit->text().trimmed());
  m_model->setRegion(m_regionEdit->text().trimmed());
  m_model->setEndpoint(m_endpointEdit->text().trimmed());

  navigateTo({});
}

void S3BrowserDialog::onNavigateUp() {
  const QString prefix = m_model->currentPrefix();
  if (prefix.isEmpty())
    return;
  // Strip last component (trailing slash already included in prefixes).
  const QString parent = prefix.chopped(1).section('/', 0, -2);
  navigateTo(parent.isEmpty() ? QString{} : parent + '/');
}

void S3BrowserDialog::onItemActivated(const QModelIndex &index) {
  if (!index.isValid())
    return;
  const QString key      = m_model->data(index, Qt::UserRole).toString();
  const bool    isPrefix = m_model->items().at(index.row()).isPrefix;
  if (isPrefix)
    navigateTo(key);
}

void S3BrowserDialog::onSelectClicked() {
  const auto sel = m_treeView->selectionModel()->selectedIndexes();
  if (sel.isEmpty())
    return;
  const QString key = m_model->data(sel.first(), Qt::UserRole).toString();
  m_selectedUrl     = QStringLiteral("s3://%1/%2").arg(
      m_model->currentBucket(), key);
  emit urlSelected(m_selectedUrl);
  accept();
}

void S3BrowserDialog::onLoadingChanged(bool loading) {
  m_connectButton->setEnabled(!loading);
  m_upButton->setEnabled(!loading && !m_model->currentPrefix().isEmpty());
  m_statusLabel->setText(loading ? tr("Loading…") : tr("Ready"));
}

void S3BrowserDialog::onLoadError(const QString &message) {
  m_statusLabel->setText(tr("Error: %1").arg(message));
  QMessageBox::warning(this, tr("S3 Error"), message);
}

void S3BrowserDialog::onBookmarkActivated() {
  const auto *item = m_bookmarkList->currentItem();
  if (!item)
    return;
  const QString url = item->data(Qt::UserRole).toString();
  char bucket[256]{}, key[2048]{};
  if (s3_parse_url(url.toUtf8().constData(), bucket, sizeof(bucket),
                    key, sizeof(key))) {
    m_bucketEdit->setText(QString::fromUtf8(bucket));
    m_model->setAccessKey(m_accessKeyEdit->text().trimmed());
    m_model->setSecretKey(m_secretKeyEdit->text().trimmed());
    m_model->setRegion(m_regionEdit->text().trimmed());
    m_model->setEndpoint(m_endpointEdit->text().trimmed());
    navigateTo(QString::fromUtf8(key));
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void S3BrowserDialog::navigateTo(const QString &prefix) {
  m_model->refresh(m_bucketEdit->text().trimmed(), prefix);
  updatePathBar();
}

void S3BrowserDialog::updatePathBar() {
  const QString bucket = m_model->currentBucket();
  const QString prefix = m_model->currentPrefix();
  m_pathLabel->setText(bucket.isEmpty() ? QStringLiteral("/")
                                        : QStringLiteral("s3://%1/%2").arg(bucket, prefix));
  m_upButton->setEnabled(!prefix.isEmpty());
}
