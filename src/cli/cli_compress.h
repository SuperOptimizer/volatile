#pragma once

// Re-compress a zarr volume with a different codec (original compress command).
int cmd_compress(int argc, char **argv);

// Encode a zarr pyramid to a .c4d file using compress4d.
int cmd_compress4d(int argc, char **argv);

// Decode a .c4d file back to a zarr volume.
int cmd_decompress4d(int argc, char **argv);

// Print metadata from a .c4d file.
int cmd_compress4d_info(int argc, char **argv);
