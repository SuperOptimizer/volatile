#define _POSIX_C_SOURCE 200809L

#include "core/log.h"
#include "core/vol.h"
#include "cli/cli_compress.h"
#include "cli/cli_convert.h"
#include "cli/cli_flatten.h"
#include "cli/cli_stats.h"
#include "cli/cli_grow.h"
#include "cli/cli_serve.h"
#include "cli/cli_connect.h"
#include "cli/cli_render.h"
#include "cli/cli_zarr_ops.h"
#include "cli/cli_metrics.h"
#include "cli/cli_mask.h"
#include "cli/cli_inpaint.h"
#include "cli/cli_normals.h"
#include "cli/cli_winding.h"
#include "cli/cli_diff.h"
#include "cli/cli_transform.h"
#include "cli/cli_video.h"
#include "cli/cli_mirror.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_usage(void) {
  puts("Usage: volatile <command> [args]\n");
  puts("Commands:");
  puts("  info     <path>                              Show volume metadata");
  puts("  sample   <path> <z> <y> <x> [--level N]     Sample a voxel value");
  puts("  convert  <in> <out> [--format zarr|tiff|nrrd] Convert formats");
  puts("  rechunk  <zarr> --output <out> --chunk-size Z,Y,X  Rechunk volume");
  puts("  stats    <path>                              Volume statistics");
  puts("  compress <path> [--output <out>]             Re-compress with compress4d");
  puts("  compress4d <input.zarr> --output <out.c4d>  Encode pyramid to .c4d");
  puts("  decompress4d <input.c4d> --output <out.zarr> Decode .c4d to zarr");
  puts("  compress4d-info <input.c4d>                  Print .c4d metadata");
  puts("  serve    [--port N] [--data DIR] [--db PATH] Start the multi-user server");
  puts("  connect  <host:port> [--volume ID]           Connect and test a server");
  puts("  flatten  <surface> --output <out.obj>          UV-flatten quad surface (LSCM)");
  puts("  grow     <path> --seed <z,y,x> [opts]        Grow surface from seed point");
  puts("  render     <surface> --volume <vol> --output <out.tiff>  Render surface to image");
  puts("  downsample <zarr> --output <out> [--factor 2]            2x mean downsampling");
  puts("  threshold  <zarr> --output <out> [--low N] [--high N]   Binarize/clip voxels");
  puts("  merge      <zarr1> <zarr2> --output <out> [--op max|add|mask]  Element-wise merge");
  puts("  extract    <zarr> --bbox z0,y0,x0,z1,y1,x1 --output <out>     Sub-region crop");
  puts("  mirror     <remote_url> [--cache-dir DIR] [--rechunk Z,Y,X] [--compress4d]");
  puts("  metrics  <surface>                            Surface area, curvature, smoothness");
  puts("  mask     <surface> --volume <vol> --output <mask>  Binary mask from surface");
  puts("  inpaint  <image> --mask <mask> --output <out>  Telea inpainting for hole-filling");
  puts("  normals  <volume> --output <normals.zarr>    Per-voxel normals via structure tensor");
  puts("  winding    <surface.obj> --output <w.zarr> --shape Z,Y,X  Winding number field");
  puts("  diff       <surface1> <surface2> [--output <diff.tiff>]  Per-vertex distance stats");
  puts("  transform  <surface> --matrix <4x4.json> --output <out>  Apply 4x4 matrix transform");
  puts("  video      <surface> --volume <vol> --output <out.mp4>   Render slice video");
  puts("  version                                      Print version string");
  puts("  help                                         Print this message");
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

static int cmd_version(void) {
  printf("volatile %s\n", volatile_version());
  return 0;
}

static int cmd_help(void) {
  print_usage();
  return 0;
}

static int cmd_info(int argc, char **argv) {
  if (argc < 1) {
    fprintf(stderr, "usage: volatile info <path>\n");
    return 1;
  }
  const char *path = argv[0];

  volume *v = vol_open(path);
  if (!v) {
    fprintf(stderr, "error: could not open volume: %s\n", path);
    return 1;
  }

  int nlevels = vol_num_levels(v);
  printf("path:    %s\n", vol_path(v));
  printf("source:  %s\n", vol_is_remote(v) ? "remote" : "local");
  printf("levels:  %d\n", nlevels);

  for (int lvl = 0; lvl < nlevels; lvl++) {
    const zarr_level_meta *m = vol_level_meta(v, lvl);
    if (!m) continue;

    printf("\nlevel %d:\n", lvl);
    printf("  ndim:   %d\n", m->ndim);

    printf("  shape:  [");
    for (int d = 0; d < m->ndim; d++)
      printf("%s%lld", d ? ", " : "", (long long)m->shape[d]);
    puts("]");

    printf("  chunks: [");
    for (int d = 0; d < m->ndim; d++)
      printf("%s%lld", d ? ", " : "", (long long)m->chunk_shape[d]);
    puts("]");

    printf("  dtype:  %s\n", dtype_name(m->dtype));
    printf("  order:  %c\n", m->order);

    if (m->compressor_id[0])
      printf("  codec:  %s/%s (level %d, shuffle %d)\n",
             m->compressor_id, m->compressor_cname,
             m->compressor_clevel, m->compressor_shuffle);
    else
      puts("  codec:  none");
  }

  vol_free(v);
  return 0;
}

static int cmd_sample(int argc, char **argv) {
  // parse: <path> <z> <y> <x> [--level N]
  if (argc < 4) {
    fprintf(stderr, "usage: volatile sample <path> <z> <y> <x> [--level N]\n");
    return 1;
  }

  const char *path = argv[0];
  float z = (float)atof(argv[1]);
  float y = (float)atof(argv[2]);
  float x = (float)atof(argv[3]);
  int level = 0;

  for (int i = 4; i < argc - 1; i++) {
    if (strcmp(argv[i], "--level") == 0)
      level = atoi(argv[i + 1]);
  }

  volume *v = vol_open(path);
  if (!v) {
    fprintf(stderr, "error: could not open volume: %s\n", path);
    return 1;
  }

  if (level < 0 || level >= vol_num_levels(v)) {
    fprintf(stderr, "error: level %d out of range (0..%d)\n",
            level, vol_num_levels(v) - 1);
    vol_free(v);
    return 1;
  }

  float val = vol_sample(v, level, z, y, x);
  printf("%.6g\n", (double)val);

  vol_free(v);
  return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }

  const char *cmd = argv[1];
  // argv+2 / argc-2 are the subcommand arguments
  char **sub_argv = argv + 2;
  int    sub_argc = argc - 2;

  if (strcmp(cmd, "version") == 0) return cmd_version();
  if (strcmp(cmd, "help")    == 0) return cmd_help();
  if (strcmp(cmd, "info")    == 0) return cmd_info(sub_argc, sub_argv);
  if (strcmp(cmd, "sample")  == 0) return cmd_sample(sub_argc, sub_argv);
  if (strcmp(cmd, "convert") == 0) return cmd_convert(sub_argc, sub_argv);
  if (strcmp(cmd, "rechunk") == 0) return cmd_rechunk(sub_argc, sub_argv);
  if (strcmp(cmd, "stats")   == 0) return cmd_stats(sub_argc, sub_argv);
  if (strcmp(cmd, "compress")        == 0) return cmd_compress(sub_argc, sub_argv);
  if (strcmp(cmd, "compress4d")      == 0) return cmd_compress4d(sub_argc, sub_argv);
  if (strcmp(cmd, "decompress4d")    == 0) return cmd_decompress4d(sub_argc, sub_argv);
  if (strcmp(cmd, "compress4d-info") == 0) return cmd_compress4d_info(sub_argc, sub_argv);
  if (strcmp(cmd, "serve")   == 0) return cmd_serve(sub_argc, sub_argv);
  if (strcmp(cmd, "connect") == 0) return cmd_connect(sub_argc, sub_argv);
  if (strcmp(cmd, "flatten") == 0) return cmd_flatten(sub_argc, sub_argv);
  if (strcmp(cmd, "grow")    == 0) return cmd_grow(sub_argc, sub_argv);
  if (strcmp(cmd, "render")     == 0) return cmd_render(sub_argc, sub_argv);
  if (strcmp(cmd, "downsample") == 0) return cmd_downsample(sub_argc, sub_argv);
  if (strcmp(cmd, "threshold")  == 0) return cmd_threshold(sub_argc, sub_argv);
  if (strcmp(cmd, "merge")      == 0) return cmd_merge(sub_argc, sub_argv);
  if (strcmp(cmd, "extract")    == 0) return cmd_extract(sub_argc, sub_argv);
  if (strcmp(cmd, "mirror")     == 0) return cmd_mirror(sub_argc, sub_argv);
  if (strcmp(cmd, "metrics")    == 0) return cmd_metrics(sub_argc, sub_argv);
  if (strcmp(cmd, "mask")       == 0) return cmd_mask(sub_argc, sub_argv);
  if (strcmp(cmd, "inpaint")    == 0) return cmd_inpaint(sub_argc, sub_argv);
  if (strcmp(cmd, "normals")    == 0) return cmd_normals(sub_argc, sub_argv);
  if (strcmp(cmd, "winding")    == 0) return cmd_winding(sub_argc, sub_argv);
  if (strcmp(cmd, "diff")       == 0) return cmd_diff(sub_argc, sub_argv);
  if (strcmp(cmd, "transform")  == 0) return cmd_transform(sub_argc, sub_argv);
  if (strcmp(cmd, "video")      == 0) return cmd_video(sub_argc, sub_argv);

  fprintf(stderr, "unknown command: %s\n\n", cmd);
  print_usage();
  return 1;
}
