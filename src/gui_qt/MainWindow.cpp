#include "MainWindow.h"
#include "SliceViewer.h"
#include "VolumeViewer.h"

#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QDockWidget>
#include <QSplitter>
#include <QPlainTextEdit>
#include <QComboBox>
#include <QSlider>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QStatusBar>
#include <QApplication>

extern "C" {
#include "core/log.h"
}

// Log hook — forwards to the console QPlainTextEdit
static QPlainTextEdit *g_consoleWidget = nullptr;

static void qtLogHook(void *, log_level_t level, const char *file, int line,
                      const char *msg) {
  if (!g_consoleWidget) return;
  static const char *prefixes[] = {"DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};
  const char *pfx = (level < 5) ? prefixes[level] : "?????";
  QString text = QString("[%1] %2:%3: %4").arg(pfx).arg(file).arg(line).arg(msg);
  g_consoleWidget->appendPlainText(text);
}

MainWindow::MainWindow(QWidget *parent)
  : QMainWindow(parent) {
  setWindowTitle("Volatile");
  resize(1600, 900);

  buildMenuBar();
  buildViewerGrid();
  buildRightDock();
  buildConsoleDock();

  // Status bar
  m_coordLabel = new QLabel("Ready");
  statusBar()->addPermanentWidget(m_coordLabel);
  statusBar()->showMessage("No volume loaded");

  // Wire log hook
  g_consoleWidget = m_console;
  log_set_callback(qtLogHook, nullptr);

  LOG_INFO("MainWindow ready");
}

MainWindow::~MainWindow() {
  log_set_callback(nullptr, nullptr);
  g_consoleWidget = nullptr;
  vol_free(m_vol);
}

// ---------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------
void MainWindow::buildMenuBar() {
  auto *file = menuBar()->addMenu("&File");
  file->addAction("Open &Zarr...",   this, &MainWindow::onOpenZarr,   QKeySequence::Open);
  file->addAction("Open &volpkg...", this, &MainWindow::onOpenVolpkg);
  file->addAction("Open &S3...",     this, &MainWindow::onOpenS3);
  file->addSeparator();
  file->addAction("&Close Volume",   this, &MainWindow::onCloseVolume);
  file->addSeparator();
  file->addAction("&Quit", qApp, &QApplication::quit, QKeySequence::Quit);

  auto *edit = menuBar()->addMenu("&Edit");
  edit->addAction("&Undo",  QKeySequence::Undo);
  edit->addAction("&Redo",  QKeySequence::Redo);

  auto *view = menuBar()->addMenu("&View");
  view->addAction("Reset &Camera");
  view->addAction("Sync &Cursor");

  menuBar()->addMenu("&Selection");

  auto *help = menuBar()->addMenu("&Help");
  help->addAction("&About Volatile", this, &MainWindow::onAbout);
}

// ---------------------------------------------------------------------------
// 2×2 viewer grid
// ---------------------------------------------------------------------------
void MainWindow::buildViewerGrid() {
  m_xyViewer = new SliceViewer(SliceViewer::XY, this);
  m_xzViewer = new SliceViewer(SliceViewer::XZ, this);
  m_yzViewer = new SliceViewer(SliceViewer::YZ, this);
  m_3dViewer = new VolumeViewer(this);

  // Label each viewer
  auto makeViewer = [](QWidget *w, const QString &label) -> QWidget * {
    auto *frame = new QWidget;
    auto *vlay  = new QVBoxLayout(frame);
    vlay->setContentsMargins(0, 0, 0, 0);
    vlay->setSpacing(0);
    auto *lbl = new QLabel(label, frame);
    lbl->setStyleSheet("background: #2a2a2a; color: #aaa; padding: 2px 4px;");
    lbl->setFixedHeight(20);
    vlay->addWidget(lbl);
    vlay->addWidget(w);
    return frame;
  };

  auto *topSplit = new QSplitter(Qt::Horizontal);
  topSplit->addWidget(makeViewer(m_xyViewer, "XY (axial)"));
  topSplit->addWidget(makeViewer(m_xzViewer, "XZ (coronal)"));
  topSplit->setStretchFactor(0, 1);
  topSplit->setStretchFactor(1, 1);

  auto *botSplit = new QSplitter(Qt::Horizontal);
  botSplit->addWidget(makeViewer(m_yzViewer, "YZ (sagittal)"));
  botSplit->addWidget(makeViewer(m_3dViewer, "3D"));
  botSplit->setStretchFactor(0, 1);
  botSplit->setStretchFactor(1, 1);

  auto *mainSplit = new QSplitter(Qt::Vertical);
  mainSplit->addWidget(topSplit);
  mainSplit->addWidget(botSplit);
  mainSplit->setStretchFactor(0, 1);
  mainSplit->setStretchFactor(1, 1);

  setCentralWidget(mainSplit);

  // Connect viewer signals
  connect(m_xyViewer, &SliceViewer::cursorMoved, this, &MainWindow::onCursorMoved);
  connect(m_xzViewer, &SliceViewer::cursorMoved, this, &MainWindow::onCursorMoved);
  connect(m_yzViewer, &SliceViewer::cursorMoved, this, &MainWindow::onCursorMoved);
}

// ---------------------------------------------------------------------------
// Right dock: volume selector + window/level + surface tree + seg stub
// ---------------------------------------------------------------------------
void MainWindow::buildRightDock() {
  auto *dock = new QDockWidget("Controls", this);
  dock->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);
  dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

  auto *container = new QWidget;
  auto *lay       = new QVBoxLayout(container);
  lay->setContentsMargins(4, 4, 4, 4);

  // Volume selector
  auto *volGroup = new QGroupBox("Volume");
  auto *volLay   = new QVBoxLayout(volGroup);
  m_volCombo = new QComboBox;
  m_volCombo->addItem("(no volume)");
  volLay->addWidget(m_volCombo);
  lay->addWidget(volGroup);

  // Window / Level
  auto *wlGroup = new QGroupBox("Window / Level");
  auto *wlLay   = new QVBoxLayout(wlGroup);
  auto *wRow    = new QHBoxLayout;
  wRow->addWidget(new QLabel("Window:"));
  m_windowSlider = new QSlider(Qt::Horizontal);
  m_windowSlider->setRange(1, 255);
  m_windowSlider->setValue(255);
  wRow->addWidget(m_windowSlider);
  auto *lRow    = new QHBoxLayout;
  lRow->addWidget(new QLabel("Level: "));
  m_levelSlider = new QSlider(Qt::Horizontal);
  m_levelSlider->setRange(0, 255);
  m_levelSlider->setValue(128);
  lRow->addWidget(m_levelSlider);
  wlLay->addLayout(wRow);
  wlLay->addLayout(lRow);
  lay->addWidget(wlGroup);

  // Surface tree
  auto *surfGroup = new QGroupBox("Surfaces");
  auto *surfLay   = new QVBoxLayout(surfGroup);
  m_surfaceTree = new QTreeWidget;
  m_surfaceTree->setHeaderLabels({"Name", "ID", "Area"});
  m_surfaceTree->setColumnWidth(0, 120);
  m_surfaceTree->setMaximumHeight(200);
  surfLay->addWidget(m_surfaceTree);
  lay->addWidget(surfGroup);

  // Segmentation (stub)
  lay->addWidget(new QGroupBox("Segmentation"));
  lay->addStretch();

  dock->setWidget(container);
  addDockWidget(Qt::RightDockWidgetArea, dock);

  // Expose toggle in View menu
  menuBar()->findChildren<QMenu *>().first();
}

// ---------------------------------------------------------------------------
// Bottom dock: log console
// ---------------------------------------------------------------------------
void MainWindow::buildConsoleDock() {
  auto *dock = new QDockWidget("Console", this);
  dock->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
  dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable |
                    QDockWidget::DockWidgetClosable);

  m_console = new QPlainTextEdit;
  m_console->setReadOnly(true);
  m_console->setStyleSheet("background:#1a1a1a; color:#ddd; font-family: monospace;");
  m_console->setFixedHeight(150);
  dock->setWidget(m_console);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
}

// ---------------------------------------------------------------------------
// Volume open
// ---------------------------------------------------------------------------
void MainWindow::openVolume(const QString &path) {
  vol_free(m_vol);
  m_vol = nullptr;

  m_vol = vol_open(path.toUtf8().constData());
  if (!m_vol) {
    statusBar()->showMessage(QString("Failed to open: %1").arg(path));
    LOG_WARN("Failed to open: %s", path.toUtf8().constData());
    return;
  }

  LOG_INFO("Opened %s (%d levels)", path.toUtf8().constData(), vol_num_levels(m_vol));
  statusBar()->showMessage(QString("Loaded: %1").arg(path));

  m_xyViewer->setVolume(m_vol);
  m_xzViewer->setVolume(m_vol);
  m_yzViewer->setVolume(m_vol);
  m_3dViewer->setVolume(m_vol);

  if (m_volCombo->count() == 1 && m_volCombo->itemText(0) == "(no volume)")
    m_volCombo->setItemText(0, path);
  else
    m_volCombo->addItem(path);
}

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------
void MainWindow::onOpenZarr() {
  QString path = QFileDialog::getExistingDirectory(
    this, "Open Zarr Directory", QDir::homePath());
  if (!path.isEmpty()) openVolume(path);
}

void MainWindow::onOpenVolpkg() {
  QString path = QFileDialog::getExistingDirectory(
    this, "Open volpkg Directory", QDir::homePath(),
    QFileDialog::ShowDirsOnly);
  if (!path.isEmpty()) openVolume(path);
}

void MainWindow::onOpenS3() {
  // TODO: S3 browser dialog
  QMessageBox::information(this, "S3 Browser",
    "S3 browser not yet implemented.\nEnter s3://bucket/path manually.");
}

void MainWindow::onCloseVolume() {
  vol_free(m_vol);
  m_vol = nullptr;
  m_xyViewer->setVolume(nullptr);
  m_xzViewer->setVolume(nullptr);
  m_yzViewer->setVolume(nullptr);
  m_3dViewer->setVolume(nullptr);
  statusBar()->showMessage("No volume loaded");
}

void MainWindow::onAbout() {
  QMessageBox::about(this, "About Volatile",
    "<b>Volatile</b><br>"
    "Volume segmentation tool<br>"
    "Qt6 GUI frontend for volatile_core");
}

void MainWindow::onSliceChanged(float) {
  // Cross-viewer slice sync could go here
}

void MainWindow::onCursorMoved(float x, float y, float z) {
  updateStatusBar(x, y, z);
}

void MainWindow::updateStatusBar(float x, float y, float z) {
  m_coordLabel->setText(
    QString("x=%1 y=%2 z=%3")
      .arg(x, 0, 'f', 1)
      .arg(y, 0, 'f', 1)
      .arg(z, 0, 'f', 1));
}
