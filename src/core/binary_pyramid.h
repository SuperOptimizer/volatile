#pragma once
#include <stdbool.h>

// ---------------------------------------------------------------------------
// binary_pyramid — compact sparse 3D binary mask with mipmap pyramid
//
// The pyramid has log2(max_dim)+1 levels. Level 0 is full resolution.
// Each coarser level cell is 1 if ANY fine cell in its 2x2x2 block is set.
// This allows O(1) region queries by checking coarser levels first.
// ---------------------------------------------------------------------------

typedef struct binary_pyramid binary_pyramid;

binary_pyramid *binary_pyramid_new(int d, int h, int w);
void            binary_pyramid_free(binary_pyramid *p);

void binary_pyramid_set(binary_pyramid *p, int z, int y, int x, bool value);
bool binary_pyramid_get(const binary_pyramid *p, int z, int y, int x);

// Fast query: is ANY voxel set in [z0,z1) × [y0,y1) × [x0,x1)?
bool binary_pyramid_any_in_region(const binary_pyramid *p,
                                  int z0, int y0, int x0,
                                  int z1, int y1, int x1);

// Count of set voxels at level 0.
int  binary_pyramid_count(const binary_pyramid *p);
