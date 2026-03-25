#include "SegmentationPanel.h"

#include <QToolBox>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QListWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QFrame>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static QWidget *labelledSlider(const QString &label, int min, int max, int value,
                                QSlider **sliderOut, QLabel **labelOut,
                                QWidget *parent = nullptr) {
  auto *w   = new QWidget(parent);
  auto *row = new QHBoxLayout(w);
  row->setContentsMargins(0, 0, 0, 0);
  row->setSpacing(6);

  auto *lbl  = new QLabel(label, w);
  auto *val  = new QLabel(QString::number(value), w);
  auto *sld  = new QSlider(Qt::Horizontal, w);
  sld->setRange(min, max);
  sld->setValue(value);
  val->setMinimumWidth(32);

  row->addWidget(lbl, 1);
  row->addWidget(sld, 3);
  row->addWidget(val);

  if (sliderOut) *sliderOut = sld;
  if (labelOut)  *labelOut  = val;
  return w;
}

// ---------------------------------------------------------------------------
// SegmentationPanel
// ---------------------------------------------------------------------------

SegmentationPanel::SegmentationPanel(QWidget *parent)
    : QDockWidget(tr("Segmentation"), parent) {
  setObjectName(QStringLiteral("SegmentationPanel"));
  setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

  m_toolBox = new QToolBox(this);
  m_toolBox->setObjectName(QStringLiteral("segToolBox"));

  m_toolBox->addItem(buildEditingSection(),    tr("Editing"));
  m_toolBox->addItem(buildGrowthSection(),     tr("Growth"));
  m_toolBox->addItem(buildCorrectionsSection(),tr("Corrections"));
  m_toolBox->addItem(buildApprovalMaskSection(),tr("Approval Mask"));
  m_toolBox->addItem(buildCustomParamsSection(),tr("Custom Params"));
  m_toolBox->addItem(buildNeuralTracerSection(),tr("Neural Tracer"));

  // Wrap toolbox in a scroll area so the panel doesn't grow unboundedly.
  auto *scroll = new QScrollArea(this);
  scroll->setWidget(m_toolBox);
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  setWidget(scroll);
}

// ---------------------------------------------------------------------------
// Section builders
// ---------------------------------------------------------------------------

QWidget *SegmentationPanel::buildEditingSection() {
  auto *w = new QWidget;
  auto *v = new QVBoxLayout(w);
  v->setContentsMargins(8, 8, 8, 8);
  v->setSpacing(6);

  v->addWidget(labelledSlider(tr("Radius"), 1, 100, 15, &m_radiusSlider, &m_radiusLabel, w));
  v->addWidget(labelledSlider(tr("Sigma"),  1, 100, 10, &m_sigmaSlider,  &m_sigmaLabel,  w));
  v->addStretch();

  connect(m_radiusSlider, &QSlider::valueChanged, this, &SegmentationPanel::onBrushRadiusChanged);
  connect(m_sigmaSlider,  &QSlider::valueChanged, this, &SegmentationPanel::onBrushSigmaChanged);
  return w;
}

QWidget *SegmentationPanel::buildGrowthSection() {
  auto *w = new QWidget;
  auto *f = new QFormLayout(w);
  f->setContentsMargins(8, 8, 8, 8);
  f->setSpacing(6);

  m_methodCombo = new QComboBox(w);
  m_methodCombo->addItem(tr("Tracer"),        0);
  m_methodCombo->addItem(tr("Extrapolation"), 1);
  m_methodCombo->addItem(tr("Corrections"),   2);
  f->addRow(tr("Method:"), m_methodCombo);

  m_directionCombo = new QComboBox(w);
  m_directionCombo->addItem(tr("All"),   0);
  m_directionCombo->addItem(tr("Up"),    1);
  m_directionCombo->addItem(tr("Down"),  2);
  m_directionCombo->addItem(tr("Left"),  3);
  m_directionCombo->addItem(tr("Right"), 4);
  f->addRow(tr("Direction:"), m_directionCombo);

  m_generationsSpin = new QSpinBox(w);
  m_generationsSpin->setRange(1, 500);
  m_generationsSpin->setValue(1);
  f->addRow(tr("Generations:"), m_generationsSpin);

  m_growButton = new QPushButton(tr("Grow"), w);
  m_growButton->setDefault(false);
  f->addRow(m_growButton);

  connect(m_growButton, &QPushButton::clicked, this, &SegmentationPanel::onGrowClicked);
  return w;
}

QWidget *SegmentationPanel::buildCorrectionsSection() {
  auto *w = new QWidget;
  auto *v = new QVBoxLayout(w);
  v->setContentsMargins(8, 8, 8, 8);
  v->setSpacing(6);

  m_correctionList = new QListWidget(w);
  m_correctionList->setMaximumHeight(120);
  v->addWidget(m_correctionList);

  auto *uvRow = new QHBoxLayout;
  uvRow->setSpacing(4);
  m_correctionU = new QDoubleSpinBox(w);
  m_correctionU->setRange(0.0, 1.0);
  m_correctionU->setSingleStep(0.01);
  m_correctionU->setDecimals(3);
  m_correctionU->setPrefix(QStringLiteral("U: "));
  m_correctionV = new QDoubleSpinBox(w);
  m_correctionV->setRange(0.0, 1.0);
  m_correctionV->setSingleStep(0.01);
  m_correctionV->setDecimals(3);
  m_correctionV->setPrefix(QStringLiteral("V: "));
  uvRow->addWidget(m_correctionU);
  uvRow->addWidget(m_correctionV);
  v->addLayout(uvRow);

  auto *btnRow = new QHBoxLayout;
  m_addCorrectionButton    = new QPushButton(tr("Add"), w);
  m_removeCorrectionButton = new QPushButton(tr("Remove"), w);
  btnRow->addWidget(m_addCorrectionButton);
  btnRow->addWidget(m_removeCorrectionButton);
  v->addLayout(btnRow);
  v->addStretch();

  connect(m_addCorrectionButton,    &QPushButton::clicked,
          this, &SegmentationPanel::onAddCorrectionClicked);
  connect(m_removeCorrectionButton, &QPushButton::clicked,
          this, &SegmentationPanel::onRemoveCorrectionClicked);
  return w;
}

QWidget *SegmentationPanel::buildApprovalMaskSection() {
  auto *w = new QWidget;
  auto *v = new QVBoxLayout(w);
  v->setContentsMargins(8, 8, 8, 8);
  v->setSpacing(6);

  v->addWidget(labelledSlider(tr("Brush Size"), 1, 100, 20,
                              &m_brushSizeSlider, &m_brushSizeLabel, w));

  auto *btnRow = new QHBoxLayout;
  m_paintApproveButton = new QPushButton(tr("Approve"), w);
  m_paintRejectButton  = new QPushButton(tr("Reject"),  w);
  m_eraseButton        = new QPushButton(tr("Erase"),   w);
  btnRow->addWidget(m_paintApproveButton);
  btnRow->addWidget(m_paintRejectButton);
  btnRow->addWidget(m_eraseButton);
  v->addLayout(btnRow);
  v->addStretch();

  connect(m_paintApproveButton, &QPushButton::clicked, this,
          [this]() { emit approvalPaintRequested(true); });
  connect(m_paintRejectButton, &QPushButton::clicked, this,
          [this]() { emit approvalPaintRequested(false); });
  return w;
}

QWidget *SegmentationPanel::buildCustomParamsSection() {
  auto *w = new QWidget;
  auto *v = new QVBoxLayout(w);
  v->setContentsMargins(8, 8, 8, 8);
  v->setSpacing(6);

  m_jsonEditor = new QPlainTextEdit(w);
  m_jsonEditor->setPlaceholderText(tr("{\n  \"key\": value\n}"));
  QFont mono(QStringLiteral("Monospace"));
  mono.setStyleHint(QFont::TypeWriter);
  mono.setPointSize(9);
  m_jsonEditor->setFont(mono);
  m_jsonEditor->setMaximumHeight(160);
  v->addWidget(m_jsonEditor);

  m_applyJsonButton = new QPushButton(tr("Apply"), w);
  v->addWidget(m_applyJsonButton);
  v->addStretch();
  return w;
}

QWidget *SegmentationPanel::buildNeuralTracerSection() {
  auto *w = new QWidget;
  auto *v = new QVBoxLayout(w);
  v->setContentsMargins(8, 8, 8, 8);
  v->setSpacing(6);

  auto *f = new QFormLayout;
  f->setSpacing(4);
  m_modelCombo = new QComboBox(w);
  m_modelCombo->setEditable(true);
  m_modelCombo->addItem(tr("default"));
  f->addRow(tr("Model:"), m_modelCombo);
  v->addLayout(f);

  auto *btnRow = new QHBoxLayout;
  m_neuralStartButton = new QPushButton(tr("Start"), w);
  m_neuralStopButton  = new QPushButton(tr("Stop"),  w);
  m_neuralStopButton->setEnabled(false);
  btnRow->addWidget(m_neuralStartButton);
  btnRow->addWidget(m_neuralStopButton);
  v->addLayout(btnRow);

  m_neuralStatusLabel = new QLabel(tr("Stopped"), w);
  m_neuralStatusLabel->setAlignment(Qt::AlignCenter);
  v->addWidget(m_neuralStatusLabel);
  v->addStretch();

  connect(m_neuralStartButton, &QPushButton::clicked,
          this, &SegmentationPanel::onNeuralStartClicked);
  connect(m_neuralStopButton,  &QPushButton::clicked,
          this, &SegmentationPanel::onNeuralStopClicked);
  return w;
}

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------

void SegmentationPanel::onGrowClicked() {
  emit growRequested(m_methodCombo->currentData().toInt(),
                     m_directionCombo->currentData().toInt(),
                     m_generationsSpin->value());
}

void SegmentationPanel::onBrushRadiusChanged(int value) {
  m_radiusLabel->setText(QString::number(value));
  emit brushChanged(static_cast<float>(value) * 0.1f,
                    static_cast<float>(m_sigmaSlider->value()) * 0.1f);
}

void SegmentationPanel::onBrushSigmaChanged(int value) {
  m_sigmaLabel->setText(QString::number(value));
  emit brushChanged(static_cast<float>(m_radiusSlider->value()) * 0.1f,
                    static_cast<float>(value) * 0.1f);
}

void SegmentationPanel::onAddCorrectionClicked() {
  float u = static_cast<float>(m_correctionU->value());
  float v = static_cast<float>(m_correctionV->value());
  m_correctionList->addItem(
      QStringLiteral("U=%1  V=%2").arg(u, 0, 'f', 3).arg(v, 0, 'f', 3));
  emit correctionAdded(u, v);
}

void SegmentationPanel::onRemoveCorrectionClicked() {
  const auto items = m_correctionList->selectedItems();
  for (auto *item : items)
    delete item;
}

void SegmentationPanel::onNeuralStartClicked() {
  m_neuralStartButton->setEnabled(false);
  m_neuralStopButton->setEnabled(true);
  m_neuralStatusLabel->setText(tr("Running"));
  emit neuralTracerStartRequested(m_modelCombo->currentText());
}

void SegmentationPanel::onNeuralStopClicked() {
  m_neuralStopButton->setEnabled(false);
  m_neuralStartButton->setEnabled(true);
  m_neuralStatusLabel->setText(tr("Stopped"));
  emit neuralTracerStopRequested();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

float SegmentationPanel::brushRadius() const {
  return static_cast<float>(m_radiusSlider->value()) * 0.1f;
}

float SegmentationPanel::brushSigma() const {
  return static_cast<float>(m_sigmaSlider->value()) * 0.1f;
}

int SegmentationPanel::growMethod() const {
  return m_methodCombo->currentData().toInt();
}

int SegmentationPanel::growDirection() const {
  return m_directionCombo->currentData().toInt();
}

int SegmentationPanel::growGenerations() const {
  return m_generationsSpin->value();
}
