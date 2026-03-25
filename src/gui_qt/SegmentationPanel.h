#pragma once

#include <QDockWidget>
#include <QWidget>

class QToolBox;
class QSlider;
class QDoubleSpinBox;
class QSpinBox;
class QComboBox;
class QCheckBox;
class QPushButton;
class QPlainTextEdit;
class QListWidget;
class QLabel;

// ---------------------------------------------------------------------------
// SegmentationPanel — QDockWidget hosting all segmentation tool panels.
// Replaces the Nuklear seg_panels stack from src/gui/seg_panels.h.
// ---------------------------------------------------------------------------

class SegmentationPanel : public QDockWidget {
  Q_OBJECT

public:
  explicit SegmentationPanel(QWidget *parent = nullptr);
  ~SegmentationPanel() override = default;

  // Query current UI state
  float brushRadius() const;
  float brushSigma() const;
  int   growMethod() const;
  int   growDirection() const;
  int   growGenerations() const;

signals:
  void growRequested(int method, int direction, int generations);
  void brushChanged(float radius, float sigma);
  void approvalPaintRequested(bool approved);
  void correctionAdded(float u, float v);
  void neuralTracerStartRequested(const QString &modelPath);
  void neuralTracerStopRequested();

private slots:
  void onGrowClicked();
  void onBrushRadiusChanged(int value);
  void onBrushSigmaChanged(int value);
  void onAddCorrectionClicked();
  void onRemoveCorrectionClicked();
  void onNeuralStartClicked();
  void onNeuralStopClicked();

private:
  QWidget *buildEditingSection();
  QWidget *buildGrowthSection();
  QWidget *buildCorrectionsSection();
  QWidget *buildApprovalMaskSection();
  QWidget *buildCustomParamsSection();
  QWidget *buildNeuralTracerSection();

  QToolBox *m_toolBox;

  // Editing
  QSlider        *m_radiusSlider;
  QSlider        *m_sigmaSlider;
  QLabel         *m_radiusLabel;
  QLabel         *m_sigmaLabel;

  // Growth
  QComboBox      *m_methodCombo;
  QComboBox      *m_directionCombo;
  QSpinBox       *m_generationsSpin;
  QPushButton    *m_growButton;

  // Corrections
  QListWidget    *m_correctionList;
  QDoubleSpinBox *m_correctionU;
  QDoubleSpinBox *m_correctionV;
  QPushButton    *m_addCorrectionButton;
  QPushButton    *m_removeCorrectionButton;

  // Approval mask
  QSlider        *m_brushSizeSlider;
  QLabel         *m_brushSizeLabel;
  QPushButton    *m_paintApproveButton;
  QPushButton    *m_paintRejectButton;
  QPushButton    *m_eraseButton;

  // Custom params
  QPlainTextEdit *m_jsonEditor;
  QPushButton    *m_applyJsonButton;

  // Neural tracer
  QComboBox      *m_modelCombo;
  QPushButton    *m_neuralStartButton;
  QPushButton    *m_neuralStopButton;
  QLabel         *m_neuralStatusLabel;
};
