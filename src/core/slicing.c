#include "core/slicing.h"
#include "core/log.h"

// vol_sample(vol, level, z, y, x) — level 0 = full resolution

void slice_volume_plane(const volume *vol, const plane_surface *plane,
                        float *out, int width, int height, float scale) {
  REQUIRE(plane && out && width > 0 && height > 0,
          "slice_volume_plane: invalid args");
  // vol == NULL is allowed: vol_sample returns 0.0 for NULL vol
  float hw = (float)(width  - 1) * 0.5f;
  float hh = (float)(height - 1) * 0.5f;

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float u = ((float)col - hw) * scale;
      float v = ((float)row - hh) * scale;
      vec3f p = plane_surface_sample(plane, u, v);
      out[row * width + col] = vol_sample(vol, 0, p.z, p.y, p.x);
    }
  }
}

void slice_volume_quad(const volume *vol, const quad_surface *surf,
                       float *out, int rows, int cols) {
  REQUIRE(surf && out && rows > 0 && cols > 0,
          "slice_volume_quad: invalid args");
  // vol == NULL is allowed: vol_sample returns 0.0 for NULL vol

  int use_rows = rows < surf->rows ? rows : surf->rows;
  int use_cols = cols < surf->cols ? cols : surf->cols;

  for (int r = 0; r < use_rows; r++) {
    for (int c = 0; c < use_cols; c++) {
      vec3f p = quad_surface_get(surf, r, c);
      out[r * cols + c] = vol_sample(vol, 0, p.z, p.y, p.x);
    }
  }
}
