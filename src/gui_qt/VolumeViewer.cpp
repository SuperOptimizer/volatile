#include "VolumeViewer.h"

#include <QLabel>
#include <QVBoxLayout>
#include <cstring>

#ifdef HAVE_VTK

#include <vtkContourFilter.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkRenderWindowInteractor.h>

extern "C" {
#include "core/vol.h"
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

VolumeViewer::VolumeViewer(QWidget *parent)
    : QWidget(parent)
    , m_renderer(vtkSmartPointer<vtkRenderer>::New())
    , m_renderWindow(vtkSmartPointer<vtkRenderWindow>::New())
    , m_volumeMapper(vtkSmartPointer<vtkSmartVolumeMapper>::New())
    , m_volume(vtkSmartPointer<vtkVolume>::New())
    , m_imageData(vtkSmartPointer<vtkImageData>::New())
    , m_isoMapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    , m_isoActor(vtkSmartPointer<vtkActor>::New())
    , m_colorTF(vtkSmartPointer<vtkColorTransferFunction>::New())
    , m_opacityTF(vtkSmartPointer<vtkPiecewiseFunction>::New())
    , m_volumeProperty(vtkSmartPointer<vtkVolumeProperty>::New())
{
  m_vtkWidget = new QVTKOpenGLNativeWidget(this);

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(m_vtkWidget);

  // Wire up render window
  m_renderWindow->AddRenderer(m_renderer);
  m_vtkWidget->setRenderWindow(m_renderWindow);

  // Trackball camera interaction
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  m_renderWindow->GetInteractor()->SetInteractorStyle(style);

  m_renderer->SetBackground(0.1, 0.1, 0.1);

  // Default transfer function: greyscale ramp
  m_colorTF->AddRGBPoint(0.0,   0.0, 0.0, 0.0);
  m_colorTF->AddRGBPoint(255.0, 1.0, 1.0, 1.0);
  m_opacityTF->AddPoint(0.0,   0.0);
  m_opacityTF->AddPoint(64.0,  0.0);
  m_opacityTF->AddPoint(255.0, 1.0);

  m_volumeProperty->SetColor(m_colorTF);
  m_volumeProperty->SetScalarOpacity(m_opacityTF);
  m_volumeProperty->ShadeOn();
  m_volumeProperty->SetInterpolationTypeToLinear();

  m_volume->SetMapper(m_volumeMapper);
  m_volume->SetProperty(m_volumeProperty);

  m_isoActor->SetMapper(m_isoMapper);
}

VolumeViewer::~VolumeViewer() = default;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void VolumeViewer::setVolume(volume *vol)
{
  m_coreVol = vol;
  if (vol) {
    loadVolumeData();
    applyRenderMode();
    resetCamera();
  }
}

void VolumeViewer::setRenderMode(RenderMode mode)
{
  m_mode = mode;
  if (m_coreVol)
    applyRenderMode();
}

void VolumeViewer::setIsoValue(float iso)
{
  m_isoValue = iso;
  if (m_mode == IsoSurface && m_coreVol)
    applyRenderMode();
}

void VolumeViewer::setTransferFunction(vtkColorTransferFunction *color,
                                       vtkPiecewiseFunction *opacity)
{
  if (color)  m_volumeProperty->SetColor(color);
  if (opacity) m_volumeProperty->SetScalarOpacity(opacity);
  if (m_mode == TransferFunction)
    m_renderWindow->Render();
}

void VolumeViewer::resetCamera()
{
  m_renderer->ResetCamera();
  m_renderWindow->Render();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void VolumeViewer::loadVolumeData()
{
  if (!m_coreVol) return;

  // Read level-0 metadata to determine dimensions
  const zarr_level_meta *meta = vol_level_meta(m_coreVol, 0);
  if (!meta || meta->ndim < 3) return;

  // shape is [Z, Y, X] in C order
  const int64_t nz = meta->shape[meta->ndim - 3];
  const int64_t ny = meta->shape[meta->ndim - 2];
  const int64_t nx = meta->shape[meta->ndim - 1];

  const int64_t cz = meta->chunk_shape[meta->ndim - 3];
  const int64_t cy = meta->chunk_shape[meta->ndim - 2];
  const int64_t cx = meta->chunk_shape[meta->ndim - 1];

  m_imageData->SetDimensions(static_cast<int>(nx),
                             static_cast<int>(ny),
                             static_cast<int>(nz));
  m_imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
  auto *voxels = static_cast<unsigned char *>(
      m_imageData->GetScalarPointer());
  std::memset(voxels, 0, static_cast<size_t>(nx * ny * nz));

  // Number of chunks in each dimension
  const int64_t ncz = (nz + cz - 1) / cz;
  const int64_t ncy = (ny + cy - 1) / cy;
  const int64_t ncx = (nx + cx - 1) / cx;

  for (int64_t kz = 0; kz < ncz; ++kz) {
    for (int64_t ky = 0; ky < ncy; ++ky) {
      for (int64_t kx = 0; kx < ncx; ++kx) {
        int64_t coords[3] = {kz, ky, kx};
        size_t chunk_size = 0;
        uint8_t *chunk = vol_read_chunk(m_coreVol, 0, coords, &chunk_size);
        if (!chunk) continue;

        // Chunk origin in voxel space
        const int64_t oz = kz * cz;
        const int64_t oy = ky * cy;
        const int64_t ox = kx * cx;

        // Actual extent of this chunk (may be smaller at volume boundary)
        const int64_t ez = std::min(cz, nz - oz);
        const int64_t ey = std::min(cy, ny - oy);
        const int64_t ex = std::min(cx, nx - ox);

        // Copy chunk voxels into vtkImageData (both use Z-Y-X / C order)
        for (int64_t z = 0; z < ez; ++z) {
          for (int64_t y = 0; y < ey; ++y) {
            const int64_t chunk_offset = (z * cy + y) * cx;
            const int64_t img_offset   = ((oz + z) * ny + (oy + y)) * nx + ox;
            const int64_t copy_len     = ex;
            if (static_cast<size_t>(chunk_offset + copy_len) <= chunk_size)
              std::memcpy(voxels + img_offset,
                          chunk  + chunk_offset,
                          static_cast<size_t>(copy_len));
          }
        }
        free(chunk);
      }
    }
  }

  m_imageData->Modified();
  m_volumeMapper->SetInputData(m_imageData);
}

void VolumeViewer::applyRenderMode()
{
  // Remove all props to start fresh
  m_renderer->RemoveAllViewProps();

  switch (m_mode) {
    case MIP: {
      m_volumeMapper->SetBlendModeToMaximumIntensity();
      m_renderer->AddVolume(m_volume);
      break;
    }

    case IsoSurface: {
      vtkNew<vtkContourFilter> contour;
      contour->SetInputData(m_imageData);
      contour->SetValue(0, static_cast<double>(m_isoValue));
      contour->Update();

      m_isoMapper->SetInputConnection(contour->GetOutputPort());
      m_isoMapper->ScalarVisibilityOff();
      m_renderer->AddActor(m_isoActor);
      break;
    }

    case TransferFunction: {
      m_volumeMapper->SetBlendModeToComposite();
      m_renderer->AddVolume(m_volume);
      break;
    }
  }

  m_renderWindow->Render();
}

// ---------------------------------------------------------------------------
// VTK not available: stub implementation
// ---------------------------------------------------------------------------

#else // !HAVE_VTK

VolumeViewer::VolumeViewer(QWidget *parent) : QWidget(parent)
{
  auto *label  = new QLabel(QStringLiteral("VTK not available"), this);
  auto *layout = new QVBoxLayout(this);
  layout->addWidget(label);
}

VolumeViewer::~VolumeViewer() = default;

void VolumeViewer::setVolume(volume *)        {}
void VolumeViewer::setRenderMode(RenderMode)  {}
void VolumeViewer::setIsoValue(float)         {}
void VolumeViewer::resetCamera()              {}
void VolumeViewer::loadVolumeData()           {}
void VolumeViewer::applyRenderMode()          {}

#endif // HAVE_VTK
