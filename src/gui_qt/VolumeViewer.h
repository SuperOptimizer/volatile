#pragma once

#include <QWidget>

#ifdef HAVE_VTK
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolume.h>
#include <vtkImageData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkVolumeProperty.h>
#include <QVTKOpenGLNativeWidget.h>
#endif

extern "C" {
#include "core/vol.h"
}

class VolumeViewer : public QWidget {
  Q_OBJECT

public:
  enum RenderMode { MIP, IsoSurface, TransferFunction };

  explicit VolumeViewer(QWidget *parent = nullptr);
  ~VolumeViewer() override;

  void setVolume(volume *vol);
  void setRenderMode(RenderMode mode);
  void setIsoValue(float iso);

#ifdef HAVE_VTK
  void setTransferFunction(vtkColorTransferFunction *color,
                           vtkPiecewiseFunction *opacity);
#endif

  void resetCamera();

private:
  void loadVolumeData();
  void applyRenderMode();

#ifdef HAVE_VTK
  QVTKOpenGLNativeWidget                   *m_vtkWidget      = nullptr;
  vtkSmartPointer<vtkRenderer>              m_renderer;
  vtkSmartPointer<vtkRenderWindow>          m_renderWindow;
  vtkSmartPointer<vtkSmartVolumeMapper>     m_volumeMapper;
  vtkSmartPointer<vtkVolume>                m_volume;
  vtkSmartPointer<vtkImageData>             m_imageData;
  vtkSmartPointer<vtkPolyDataMapper>        m_isoMapper;
  vtkSmartPointer<vtkActor>                 m_isoActor;
  vtkSmartPointer<vtkColorTransferFunction> m_colorTF;
  vtkSmartPointer<vtkPiecewiseFunction>     m_opacityTF;
  vtkSmartPointer<vtkVolumeProperty>        m_volumeProperty;
#endif

  volume    *m_coreVol  = nullptr;
  RenderMode m_mode     = MIP;
  float      m_isoValue = 128.0f;
};
