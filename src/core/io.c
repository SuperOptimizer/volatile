#include "core/io.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

size_t dtype_size(dtype_t dt) {
  switch (dt) {
    case DTYPE_U8:  return 1;
    case DTYPE_U16: return 2;
    case DTYPE_F32: return 4;
    case DTYPE_F64: return 8;
  }
  return 0;
}

const char *dtype_name(dtype_t dt) {
  switch (dt) {
    case DTYPE_U8:  return "uint8";
    case DTYPE_U16: return "uint16";
    case DTYPE_F32: return "float32";
    case DTYPE_F64: return "float64";
  }
  return "unknown";
}

void image_free(image *img) {
  if (!img) return;
  free(img->data);
  free(img);
}

// ---------------------------------------------------------------------------
// TIFF reader
// ---------------------------------------------------------------------------

// TIFF tag IDs we care about
#define TIFF_TAG_IMAGE_WIDTH        256
#define TIFF_TAG_IMAGE_HEIGHT       257
#define TIFF_TAG_BITS_PER_SAMPLE    258
#define TIFF_TAG_COMPRESSION        259
#define TIFF_TAG_PHOTOMETRIC        262
#define TIFF_TAG_STRIP_OFFSETS      273
#define TIFF_TAG_SAMPLES_PER_PIXEL  277
#define TIFF_TAG_STRIP_BYTE_COUNTS  279
#define TIFF_TAG_SAMPLE_FORMAT      339

// TIFF data types
#define TIFF_TYPE_SHORT  3
#define TIFF_TYPE_LONG   4

typedef struct {
  uint8_t *buf;
  size_t   size;
  bool     little_endian;
} tiff_ctx;

static uint16_t tiff_u16(tiff_ctx *ctx, size_t off) {
  if (off + 2 > ctx->size) return 0;
  uint16_t v;
  memcpy(&v, ctx->buf + off, 2);
  if (!ctx->little_endian) v = (uint16_t)((v >> 8) | (v << 8));
  return v;
}

static uint32_t tiff_u32(tiff_ctx *ctx, size_t off) {
  if (off + 4 > ctx->size) return 0;
  uint32_t v;
  memcpy(&v, ctx->buf + off, 4);
  if (!ctx->little_endian) {
    v = ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |
        ((v & 0x00FF0000u) >>  8) | ((v & 0xFF000000u) >> 24);
  }
  return v;
}

// read a tag value (short or long, single value or offset-based)
static uint32_t tiff_tag_value(tiff_ctx *ctx, size_t entry_off) {
  uint16_t type  = tiff_u16(ctx, entry_off + 2);
  uint32_t count = tiff_u32(ctx, entry_off + 4);
  // value/offset is at entry_off+8
  if (type == TIFF_TYPE_SHORT && count == 1) return tiff_u16(ctx, entry_off + 8);
  if (type == TIFF_TYPE_LONG  && count == 1) return tiff_u32(ctx, entry_off + 8);
  // for multi-value, just return the first
  if (type == TIFF_TYPE_SHORT && count > 1) {
    uint32_t off = tiff_u32(ctx, entry_off + 8);
    return tiff_u16(ctx, off);
  }
  if (type == TIFF_TYPE_LONG && count > 1) {
    uint32_t off = tiff_u32(ctx, entry_off + 8);
    return tiff_u32(ctx, off);
  }
  return 0;
}

// collect all strip offsets and byte counts into caller-supplied arrays
static int tiff_tag_array(tiff_ctx *ctx, size_t entry_off, uint32_t *out, int max_count) {
  uint16_t type  = tiff_u16(ctx, entry_off + 2);
  uint32_t count = tiff_u32(ctx, entry_off + 4);
  if (count == 0) return 0;
  int n = (int)count < max_count ? (int)count : max_count;
  if (count == 1) {
    // value fits in the 4-byte field
    if (type == TIFF_TYPE_SHORT) out[0] = tiff_u16(ctx, entry_off + 8);
    else                          out[0] = tiff_u32(ctx, entry_off + 8);
    return 1;
  }
  uint32_t data_off = tiff_u32(ctx, entry_off + 8);
  for (int i = 0; i < n; i++) {
    if (type == TIFF_TYPE_SHORT) out[i] = tiff_u16(ctx, data_off + (size_t)i * 2);
    else                          out[i] = tiff_u32(ctx, data_off + (size_t)i * 4);
  }
  return n;
}

image *tiff_read(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { LOG_ERROR("tiff_read: cannot open %s: %s", path, strerror(errno)); return NULL; }

  fseek(f, 0, SEEK_END);
  long fsz = ftell(f);
  rewind(f);
  if (fsz <= 0) { fclose(f); return NULL; }

  uint8_t *buf = malloc((size_t)fsz);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) { free(buf); fclose(f); return NULL; }
  fclose(f);

  tiff_ctx ctx = { .buf = buf, .size = (size_t)fsz };

  // byte order
  if (buf[0] == 'I' && buf[1] == 'I') ctx.little_endian = true;
  else if (buf[0] == 'M' && buf[1] == 'M') ctx.little_endian = false;
  else { LOG_ERROR("tiff_read: bad byte-order marker"); free(buf); return NULL; }

  uint16_t magic = tiff_u16(&ctx, 2);
  if (magic != 42) { LOG_ERROR("tiff_read: not a TIFF (magic=%u)", magic); free(buf); return NULL; }

  uint32_t ifd_off = tiff_u32(&ctx, 4);

  // collect all IFDs (pages/frames)
  // first pass: count pages
  int page_count = 0;
  {
    uint32_t off = ifd_off;
    while (off != 0 && off + 2 <= (uint32_t)fsz) {
      uint16_t nentries = tiff_u16(&ctx, off);
      off += 2 + (uint32_t)nentries * 12;
      if (off + 4 > (uint32_t)fsz) break;
      off = tiff_u32(&ctx, off);
      page_count++;
    }
  }
  if (page_count == 0) { free(buf); return NULL; }

  // read first IFD for image dimensions
  uint16_t nentries = tiff_u16(&ctx, ifd_off);
  int width = 0, height = 0, bits = 8, samples = 1, compression = 1, sample_fmt = 1;

  // NOTE: max 256 strips per page — adequate for typical scientific images
  #define MAX_STRIPS 256
  uint32_t strip_offsets[MAX_STRIPS]    = {0};
  uint32_t strip_byte_counts[MAX_STRIPS]= {0};
  int      nstrips = 0;

  for (int i = 0; i < nentries; i++) {
    size_t eoff = ifd_off + 2 + (size_t)i * 12;
    uint16_t tag = tiff_u16(&ctx, eoff);
    switch (tag) {
      case TIFF_TAG_IMAGE_WIDTH:       width       = (int)tiff_tag_value(&ctx, eoff); break;
      case TIFF_TAG_IMAGE_HEIGHT:      height      = (int)tiff_tag_value(&ctx, eoff); break;
      case TIFF_TAG_BITS_PER_SAMPLE:   bits        = (int)tiff_tag_value(&ctx, eoff); break;
      case TIFF_TAG_COMPRESSION:       compression = (int)tiff_tag_value(&ctx, eoff); break;
      case TIFF_TAG_SAMPLES_PER_PIXEL: samples     = (int)tiff_tag_value(&ctx, eoff); break;

      case TIFF_TAG_SAMPLE_FORMAT:     sample_fmt  = (int)tiff_tag_value(&ctx, eoff); break;
      case TIFF_TAG_STRIP_OFFSETS:
        nstrips = tiff_tag_array(&ctx, eoff, strip_offsets, MAX_STRIPS); break;
      case TIFF_TAG_STRIP_BYTE_COUNTS:
        tiff_tag_array(&ctx, eoff, strip_byte_counts, MAX_STRIPS); break;
      default: break;
    }
  }

  if (compression != 1) {
    LOG_ERROR("tiff_read: compression %d not supported (only uncompressed)", compression);
    free(buf); return NULL;
  }

  dtype_t dtype;
  if (bits == 8  && sample_fmt == 1) dtype = DTYPE_U8;
  else if (bits == 16 && sample_fmt == 1) dtype = DTYPE_U16;
  else if (bits == 32 && sample_fmt == 3) dtype = DTYPE_F32;
  else {
    LOG_ERROR("tiff_read: unsupported bits=%d sample_fmt=%d", bits, sample_fmt);
    free(buf); return NULL;
  }

  size_t pixel_bytes = dtype_size(dtype);
  size_t page_bytes  = (size_t)width * (size_t)height * (size_t)samples * pixel_bytes;
  size_t total_bytes = page_bytes * (size_t)page_count;

  uint8_t *data = malloc(total_bytes);
  if (!data) { free(buf); return NULL; }

  // read all pages
  uint32_t cur_ifd = ifd_off;
  for (int page = 0; page < page_count; page++) {
    uint8_t *page_dst = data + (size_t)page * page_bytes;

    // re-parse this IFD for strip info (for page > 0)
    if (page > 0) {
      uint16_t ne = tiff_u16(&ctx, cur_ifd);
      nstrips = 0;
      for (int i = 0; i < ne; i++) {
        size_t eoff = cur_ifd + 2 + (size_t)i * 12;
        uint16_t tag = tiff_u16(&ctx, eoff);
        if (tag == TIFF_TAG_STRIP_OFFSETS)
          nstrips = tiff_tag_array(&ctx, eoff, strip_offsets, MAX_STRIPS);
        else if (tag == TIFF_TAG_STRIP_BYTE_COUNTS)
          tiff_tag_array(&ctx, eoff, strip_byte_counts, MAX_STRIPS);
      }
    }

    size_t dst_off = 0;
    for (int s = 0; s < nstrips; s++) {
      uint32_t src_off = strip_offsets[s];
      uint32_t nbytes  = strip_byte_counts[s];
      if (src_off + nbytes > (uint32_t)fsz) break;
      size_t copy = nbytes < (page_bytes - dst_off) ? nbytes : (page_bytes - dst_off);
      memcpy(page_dst + dst_off, buf + src_off, copy);
      dst_off += copy;
    }

    // advance to next IFD
    uint16_t ne = tiff_u16(&ctx, cur_ifd);
    uint32_t next = tiff_u32(&ctx, cur_ifd + 2 + (uint32_t)ne * 12);
    cur_ifd = next;
  }

  free(buf);

  image *img = calloc(1, sizeof(image));
  if (!img) { free(data); return NULL; }
  img->width     = width;
  img->height    = height;
  img->depth     = page_count;
  img->channels  = samples;
  img->dtype     = dtype;
  img->data      = data;
  img->data_size = total_bytes;
  return img;
}

// ---------------------------------------------------------------------------
// NRRD reader
// ---------------------------------------------------------------------------

static void nrrd_trim(char *s) {
  // trim trailing whitespace/CR
  int n = (int)strlen(s);
  while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ' || s[n-1] == '\t'))
    s[--n] = '\0';
}

static dtype_t nrrd_parse_type(const char *val) {
  if (strstr(val, "uint8")  || strstr(val, "unsigned char"))  return DTYPE_U8;
  if (strstr(val, "uint16") || strstr(val, "unsigned short")) return DTYPE_U16;
  if (strstr(val, "float"))                                   return DTYPE_F32;
  if (strstr(val, "double"))                                  return DTYPE_F64;
  return DTYPE_U8;
}

nrrd_data *nrrd_read(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { LOG_ERROR("nrrd_read: cannot open %s: %s", path, strerror(errno)); return NULL; }

  char line[1024];
  // validate magic
  if (!fgets(line, sizeof(line), f)) { fclose(f); return NULL; }
  if (strncmp(line, "NRRD", 4) != 0) {
    LOG_ERROR("nrrd_read: not an NRRD file"); fclose(f); return NULL;
  }

  nrrd_data *n = calloc(1, sizeof(nrrd_data));
  if (!n) { fclose(f); return NULL; }

  bool is_gzip    = false;
  bool detached   = false;
  char data_file[256] = {0};
  long data_start = 0;

  // parse header lines until blank line
  while (fgets(line, sizeof(line), f)) {
    nrrd_trim(line);
    if (line[0] == '\0') break;           // blank line = end of header
    if (line[0] == '#') continue;         // comment

    char *colon = strchr(line, ':');
    if (!colon) continue;
    *colon = '\0';
    char *key = line;
    char *val = colon + 1;
    while (*val == ' ' || *val == '\t') val++;

    if (strcmp(key, "type") == 0) {
      n->dtype = nrrd_parse_type(val);
    } else if (strcmp(key, "dimension") == 0) {
      n->ndim = atoi(val);
      if (n->ndim > 8) n->ndim = 8;
    } else if (strcmp(key, "sizes") == 0) {
      char *tok = strtok(val, " \t");
      for (int i = 0; i < n->ndim && tok; i++, tok = strtok(NULL, " \t"))
        n->sizes[i] = atoi(tok);
    } else if (strcmp(key, "encoding") == 0) {
      is_gzip = (strncmp(val, "gzip", 4) == 0 || strncmp(val, "gz", 2) == 0);
    } else if (strcmp(key, "data file") == 0) {
      detached = true;
      strncpy(data_file, val, sizeof(data_file) - 1);
    } else if (strcmp(key, "space directions") == 0) {
      // parse up to 8 vectors of the form (a,b,c)
      char *p = val;
      for (int i = 0; i < n->ndim; i++) {
        while (*p && *p != '(') p++;
        if (!*p) break;
        p++;
        sscanf(p, "%f,%f,%f", &n->space_directions[i][0],
               &n->space_directions[i][1], &n->space_directions[i][2]);
        while (*p && *p != ')') p++;
        if (*p) p++;
      }
    } else if (strcmp(key, "space origin") == 0) {
      char *p = val;
      while (*p && *p != '(') p++;
      if (*p) { p++; sscanf(p, "%f,%f,%f", &n->space_origin[0], &n->space_origin[1], &n->space_origin[2]); }
    }
  }

  if (is_gzip) {
    LOG_ERROR("nrrd_read: gzip encoding not supported (raw only)");
    fclose(f); free(n); return NULL;
  }

  // compute total element count
  size_t total_elems = 1;
  for (int i = 0; i < n->ndim; i++) total_elems *= (size_t)n->sizes[i];
  n->data_size = total_elems * dtype_size(n->dtype);

  n->data = malloc(n->data_size);
  if (!n->data) { fclose(f); free(n); return NULL; }

  if (detached) {
    fclose(f);
    f = fopen(data_file, "rb");
    if (!f) { LOG_ERROR("nrrd_read: cannot open detached data %s", data_file); free(n->data); free(n); return NULL; }
    data_start = 0;
  } else {
    data_start = ftell(f);
  }

  fseek(f, data_start, SEEK_SET);
  size_t nr = fread(n->data, 1, n->data_size, f);
  fclose(f);

  if (nr != n->data_size) {
    LOG_WARN("nrrd_read: expected %zu bytes, got %zu", n->data_size, nr);
    n->data_size = nr;
  }

  return n;
}

void nrrd_free(nrrd_data *n) {
  if (!n) return;
  free(n->data);
  free(n);
}

// ---------------------------------------------------------------------------
// PPM / PGM
// ---------------------------------------------------------------------------

// skip whitespace and comments in PPM header
static void ppm_skip(FILE *f) {
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); continue; }
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
    ungetc(c, f);
    break;
  }
}

static int ppm_read_int(FILE *f) {
  ppm_skip(f);
  int v = 0;
  int c;
  while ((c = fgetc(f)) != EOF && c >= '0' && c <= '9')
    v = v * 10 + (c - '0');
  ungetc(c, f);
  return v;
}

image *ppm_read(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { LOG_ERROR("ppm_read: cannot open %s", path); return NULL; }

  char magic[3] = {0};
  if (fread(magic, 1, 2, f) != 2) { fclose(f); return NULL; }

  int channels;
  if      (magic[0] == 'P' && magic[1] == '5') channels = 1;
  else if (magic[0] == 'P' && magic[1] == '6') channels = 3;
  else { LOG_ERROR("ppm_read: unsupported format %s", magic); fclose(f); return NULL; }

  int width  = ppm_read_int(f);
  int height = ppm_read_int(f);
  int maxval = ppm_read_int(f);
  // consume the single whitespace after maxval
  fgetc(f);

  if (width <= 0 || height <= 0 || maxval <= 0) { fclose(f); return NULL; }

  dtype_t dtype = (maxval > 255) ? DTYPE_U16 : DTYPE_U8;
  size_t data_size = (size_t)width * (size_t)height * (size_t)channels * dtype_size(dtype);

  void *data = malloc(data_size);
  if (!data) { fclose(f); return NULL; }

  size_t nr = fread(data, 1, data_size, f);
  fclose(f);
  if (nr != data_size) { free(data); return NULL; }

  image *img = calloc(1, sizeof(image));
  if (!img) { free(data); return NULL; }
  img->width     = width;
  img->height    = height;
  img->depth     = 1;
  img->channels  = channels;
  img->dtype     = dtype;
  img->data      = data;
  img->data_size = data_size;
  return img;
}

bool ppm_write(const char *path, const image *img) {
  if (!img || img->dtype != DTYPE_U8) {
    LOG_ERROR("ppm_write: only uint8 images supported"); return false;
  }
  if (img->channels != 1 && img->channels != 3) {
    LOG_ERROR("ppm_write: only 1 or 3 channel images supported"); return false;
  }

  FILE *f = fopen(path, "wb");
  if (!f) { LOG_ERROR("ppm_write: cannot open %s: %s", path, strerror(errno)); return false; }

  const char *magic = (img->channels == 1) ? "P5" : "P6";
  fprintf(f, "%s\n%d %d\n255\n", magic, img->width, img->height);
  size_t nw = fwrite(img->data, 1, img->data_size, f);
  fclose(f);
  return nw == img->data_size;
}

bool pgm_write(const char *path, const uint8_t *data, int width, int height) {
  if (!data || width <= 0 || height <= 0) return false;
  FILE *f = fopen(path, "wb");
  if (!f) { LOG_ERROR("pgm_write: cannot open %s: %s", path, strerror(errno)); return false; }
  fprintf(f, "P5\n%d %d\n255\n", width, height);
  size_t total = (size_t)width * (size_t)height;
  size_t nw = fwrite(data, 1, total, f);
  fclose(f);
  return nw == total;
}

// ---------------------------------------------------------------------------
// OBJ mesh reader
// ---------------------------------------------------------------------------

// dynamic array helpers (local, not exported)
typedef struct { float *data; int count; int cap; } float_arr;
typedef struct { int   *data; int count; int cap; } int_arr;

static bool farr_push3(float_arr *a, float x, float y, float z) {
  if (a->count + 3 > a->cap) {
    int new_cap = a->cap ? a->cap * 2 : 64;
    float *p = realloc(a->data, (size_t)new_cap * sizeof(float));
    if (!p) return false;
    a->data = p; a->cap = new_cap;
  }
  a->data[a->count++] = x;
  a->data[a->count++] = y;
  a->data[a->count++] = z;
  return true;
}

static bool iarr_push(int_arr *a, int v) {
  if (a->count >= a->cap) {
    int new_cap = a->cap ? a->cap * 2 : 64;
    int *p = realloc(a->data, (size_t)new_cap * sizeof(int));
    if (!p) return false;
    a->data = p; a->cap = new_cap;
  }
  a->data[a->count++] = v;
  return true;
}

// parse "v[/[vt]/vn]" face token, return 0-based vertex index
static int parse_face_token(const char *tok) {
  // vertex index is before the first '/'
  return atoi(tok) - 1;
}

obj_mesh *obj_read(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { LOG_ERROR("obj_read: cannot open %s: %s", path, strerror(errno)); return NULL; }

  float_arr verts   = {0};
  float_arr normals = {0};
  int_arr   indices = {0};
  int_arr   nidxs   = {0};  // normal indices (parallel to indices)
  bool      has_normals = false;

  char line[512];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == 'v' && line[1] == ' ') {
      float x, y, z;
      if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3)
        farr_push3(&verts, x, y, z);
    } else if (line[0] == 'v' && line[1] == 'n' && line[2] == ' ') {
      float x, y, z;
      if (sscanf(line + 3, "%f %f %f", &x, &y, &z) == 3) {
        farr_push3(&normals, x, y, z);
        has_normals = true;
      }
    } else if (line[0] == 'f' && line[1] == ' ') {
      // parse up to 4 face tokens (triangulate quads)
      char toks[4][64] = {{0}};
      int ntoks = sscanf(line + 2, "%63s %63s %63s %63s",
                         toks[0], toks[1], toks[2], toks[3]);
      if (ntoks < 3) continue;

      // triangle: 0,1,2; quad: 0,1,2 + 0,2,3
      int vi[4], ni[4];
      for (int i = 0; i < ntoks; i++) {
        vi[i] = parse_face_token(toks[i]);
        // find normal index after "//" or second "/"
        ni[i] = -1;
        char *p = strchr(toks[i], '/');
        if (p) {
          p++;
          if (*p == '/') p++;  // skip "//"
          else { p = strchr(p, '/'); if (p) p++; }
          if (p && *p) ni[i] = atoi(p) - 1;
        }
      }

      // emit triangle(s)
      int tri_vi[6], tri_ni[6], nout = 0;
      tri_vi[nout] = vi[0]; tri_ni[nout++] = ni[0];
      tri_vi[nout] = vi[1]; tri_ni[nout++] = ni[1];
      tri_vi[nout] = vi[2]; tri_ni[nout++] = ni[2];
      if (ntoks == 4) {
        tri_vi[nout] = vi[0]; tri_ni[nout++] = ni[0];
        tri_vi[nout] = vi[2]; tri_ni[nout++] = ni[2];
        tri_vi[nout] = vi[3]; tri_ni[nout++] = ni[3];
      }
      for (int i = 0; i < nout; i++) {
        iarr_push(&indices, tri_vi[i]);
        iarr_push(&nidxs,   tri_ni[i]);
      }
    }
  }
  fclose(f);

  int vertex_count = verts.count / 3;
  if (vertex_count == 0) {
    free(verts.data); free(normals.data); free(indices.data); free(nidxs.data);
    return NULL;
  }

  // if normals are indexed separately, expand them to per-vertex array
  float *out_normals = NULL;
  if (has_normals && normals.count > 0 && nidxs.count == indices.count) {
    // build per-face-vertex normals (not per-vertex; just store flat like vertices)
    // NOTE: for simplicity we re-index normals to match vertex indices when possible
    out_normals = calloc((size_t)vertex_count * 3, sizeof(float));
    if (out_normals) {
      for (int i = 0; i < indices.count; i++) {
        int vi = indices.data[i];
        int ni = nidxs.data[i];
        if (vi >= 0 && vi < vertex_count && ni >= 0 && ni * 3 + 2 < normals.count) {
          out_normals[vi*3+0] = normals.data[ni*3+0];
          out_normals[vi*3+1] = normals.data[ni*3+1];
          out_normals[vi*3+2] = normals.data[ni*3+2];
        }
      }
    }
  }

  free(normals.data);
  free(nidxs.data);

  obj_mesh *m = calloc(1, sizeof(obj_mesh));
  if (!m) { free(verts.data); free(indices.data); free(out_normals); return NULL; }
  m->vertices     = verts.data;
  m->indices      = indices.data;
  m->normals      = out_normals;
  m->vertex_count = vertex_count;
  m->index_count  = indices.count;
  return m;
}

void obj_free(obj_mesh *m) {
  if (!m) return;
  free(m->vertices);
  free(m->indices);
  free(m->normals);
  free(m);
}

// ---------------------------------------------------------------------------
// TIFF writer (uncompressed baseline TIFF, little-endian)
// Supports: uint8 or uint16, 1 (grayscale) or 3 (RGB) channels.
// NOTE: We write a minimal IFD with the 12 tags required by the TIFF 6.0 spec
// for a baseline grayscale/RGB image.  No compression, no pyramid levels.
// ---------------------------------------------------------------------------

// Write a 4-byte little-endian uint32 at the current position.
static void w32(FILE *f, uint32_t v) {
  uint8_t b[4] = { v & 0xFF, (v>>8)&0xFF, (v>>16)&0xFF, (v>>24)&0xFF };
  fwrite(b, 1, 4, f);
}
static void w16(FILE *f, uint16_t v) {
  uint8_t b[2] = { v & 0xFF, (v>>8)&0xFF };
  fwrite(b, 1, 2, f);
}

// Write one IFD entry: tag, type, count, value/offset.
// type: 3=SHORT(2B), 4=LONG(4B).  For SHORT with count==1, value is right-padded.
static void ifd_entry(FILE *f, uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
  w16(f, tag);
  w16(f, type);
  w32(f, count);
  if (type == 3 && count == 1) { w16(f, (uint16_t)value); w16(f, 0); }
  else w32(f, value);
}

bool tiff_write(const char *path, const image *img) {
  if (!img || !img->data || img->width <= 0 || img->height <= 0) return false;
  if (img->channels != 1 && img->channels != 3) return false;
  if (img->dtype != DTYPE_U8 && img->dtype != DTYPE_U16) return false;

  FILE *f = fopen(path, "wb");
  if (!f) return false;

  int      w        = img->width;
  int      h        = img->height;
  int      ch       = img->channels;
  uint16_t bps      = (uint16_t)(img->dtype == DTYPE_U8 ? 8 : 16);
  uint32_t row_bytes = (uint32_t)(w * ch * (bps / 8));
  uint32_t data_size = row_bytes * (uint32_t)h;

  // TIFF header: byte order "II" (little-endian), magic 42, IFD offset.
  // Image data starts at offset 8; IFD follows immediately after data.
  uint32_t data_offset = 8;
  uint32_t ifd_offset  = data_offset + data_size;

  fwrite("II", 1, 2, f);  // little-endian
  w16(f, 42);              // TIFF magic
  w32(f, ifd_offset);      // offset to first IFD

  // Write raw image data (rows top-to-bottom, samples interleaved).
  fwrite(img->data, 1, data_size, f);

  // IFD: count then entries then next-IFD offset (0 = last).
  // Tags (must be in ascending numeric order):
  //  256=ImageWidth, 257=ImageLength, 258=BitsPerSample,
  //  259=Compression(1=none), 262=PhotometricInterpretation,
  //  273=StripOffsets, 278=RowsPerStrip, 279=StripByteCounts,
  //  280=MinSampleValue, 281=MaxSampleValue, 282=XResolution(*), 283=YResolution(*),
  //  284=PlanarConfig(1=contig), 296=ResolutionUnit(1=no units)
  // (*) XResolution/YResolution are RATIONAL (type 5) — we point them to a small
  //     data area appended after the IFD.

  // We store XRes and YRes as 1/1 rationals appended right after the IFD.
  uint16_t n_entries = 14;
  uint32_t rat_offset = ifd_offset + 2 + (uint32_t)n_entries * 12 + 4;

  w16(f, n_entries);
  ifd_entry(f, 256, 4, 1, (uint32_t)w);
  ifd_entry(f, 257, 4, 1, (uint32_t)h);
  // BitsPerSample: for RGB we need 3 SHORTs; store them after rational data.
  if (ch == 1) {
    ifd_entry(f, 258, 3, 1, bps);
  } else {
    // 3 shorts packed in the offset field (little-endian): bps,bps,bps
    // Actually they don't fit in 4 bytes as separate shorts — use offset to extra data.
    uint32_t bps_offset = rat_offset + 16; // 16 bytes for two rationals
    ifd_entry(f, 258, 3, 3, bps_offset);
  }
  ifd_entry(f, 259, 3, 1, 1);   // Compression = None
  ifd_entry(f, 262, 3, 1, (uint32_t)(ch == 1 ? 1 : 2)); // 1=BlackIsZero, 2=RGB
  ifd_entry(f, 273, 4, 1, data_offset);   // StripOffsets (single strip)
  ifd_entry(f, 278, 4, 1, (uint32_t)h);   // RowsPerStrip
  ifd_entry(f, 279, 4, 1, data_size);     // StripByteCounts
  ifd_entry(f, 280, 3, 1, 0);             // MinSampleValue
  ifd_entry(f, 281, 3, 1, (uint32_t)(bps == 8 ? 255 : 65535));
  ifd_entry(f, 282, 5, 1, rat_offset);    // XResolution RATIONAL
  ifd_entry(f, 283, 5, 1, rat_offset+8);  // YResolution RATIONAL
  ifd_entry(f, 284, 3, 1, 1);             // PlanarConfig = Contiguous
  ifd_entry(f, 296, 3, 1, 1);             // ResolutionUnit = No absolute unit
  w32(f, 0);                              // next IFD offset = 0

  // Append rational data: two 1/1 rationals (XRes, YRes).
  w32(f, 1); w32(f, 1);   // XResolution = 1/1
  w32(f, 1); w32(f, 1);   // YResolution = 1/1

  // Append BitsPerSample data for RGB (3 × uint16).
  if (ch == 3) { w16(f, bps); w16(f, bps); w16(f, bps); }

  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// tiff_write_multipage — multi-page TIFF for 3D stacks
//
// Layout: header (8 bytes), then for each page: image data then IFD.
// Each IFD's next-IFD pointer links to the next page's IFD (0 for last).
//
// Supports: uint8, uint16, float32 (SampleFormat=3); 1 or 3 channels.
// ---------------------------------------------------------------------------

bool tiff_write_multipage(const char *path, const void **pages, int depth,
                          int width, int height, dtype_t dtype, int channels) {
  if (!path || !pages || depth < 1 || width < 1 || height < 1) return false;
  if (channels != 1 && channels != 3) return false;
  if (dtype != DTYPE_U8 && dtype != DTYPE_U16 && dtype != DTYPE_F32) return false;

  FILE *f = fopen(path, "wb");
  if (!f) return false;

  uint16_t bps        = (uint16_t)(dtype_size(dtype) * 8);
  uint16_t sample_fmt = (dtype == DTYPE_F32) ? 3 : 1;  // 1=uint, 3=IEEE float
  uint32_t row_bytes  = (uint32_t)(width * channels) * (bps / 8);
  uint32_t img_bytes  = row_bytes * (uint32_t)height;

  // Per-page layout (relative to start of page block):
  //   data: img_bytes
  //   IFD:  2 + n_tags*12 + 4 bytes
  //   extra: rationals (16 bytes) + optional bps_data (6 bytes for RGB)
  uint16_t n_tags     = (uint16_t)(channels == 3 ? 16 : 15);  // +SamplesPerPixel, +BitsPerSample offset, +SampleFormat
  uint32_t ifd_bytes  = 2u + (uint32_t)n_tags * 12u + 4u;
  uint32_t extra_base = ifd_bytes;                              // rationals start here (relative to IFD start)
  uint32_t bps_off_rel= extra_base + 16u;                       // bps data after 2 rationals

  // Pre-compute absolute offsets for all pages.
  // Page i: data_off[i] = 8 + sum_{j<i}(img_bytes + ifd_bytes + 16 + [6 if RGB])
  uint32_t extra_per_page = 16u + (uint32_t)(channels == 3 ? 6u : 0u);
  uint32_t page_stride    = img_bytes + ifd_bytes + extra_per_page;

  fwrite("II", 1, 2, f);           // little-endian byte order
  w16(f, 42);                       // TIFF magic
  w32(f, 8u + img_bytes);           // first IFD offset = after first page's data

  for (int pg = 0; pg < depth; pg++) {
    uint32_t data_off = 8u + (uint32_t)pg * page_stride;
    uint32_t ifd_off  = data_off + img_bytes;
    uint32_t rat_off  = ifd_off + extra_base;
    uint32_t bps_off  = ifd_off + bps_off_rel;
    uint32_t next_ifd = (pg + 1 < depth) ? (ifd_off + page_stride) : 0u;

    // Write image data for this page.
    fwrite(pages[pg], 1, img_bytes, f);

    // IFD
    w16(f, n_tags);
    ifd_entry(f, 256, 4, 1, (uint32_t)width);
    ifd_entry(f, 257, 4, 1, (uint32_t)height);
    if (channels == 1) {
      ifd_entry(f, 258, 3, 1, bps);
    } else {
      ifd_entry(f, 258, 3, 3, bps_off);
    }
    ifd_entry(f, 259, 3, 1, 1);                                     // Compression=none
    ifd_entry(f, 262, 3, 1, (uint32_t)(channels == 1 ? 1u : 2u));  // PhotometricInterp
    ifd_entry(f, 273, 4, 1, data_off);                              // StripOffsets
    ifd_entry(f, 278, 4, 1, (uint32_t)height);                     // RowsPerStrip
    ifd_entry(f, 279, 4, 1, img_bytes);                             // StripByteCounts
    ifd_entry(f, 280, 3, 1, 0);                                     // MinSampleValue
    ifd_entry(f, 281, 3, 1, (bps == 8u ? 255u : (bps == 16u ? 65535u : 0u)));
    ifd_entry(f, 282, 5, 1, rat_off);                               // XResolution
    ifd_entry(f, 283, 5, 1, rat_off + 8u);                         // YResolution
    ifd_entry(f, 284, 3, 1, 1);                                     // PlanarConfig=contiguous
    ifd_entry(f, 296, 3, 1, 1);                                     // ResolutionUnit=none
    ifd_entry(f, 339, 3, 1, sample_fmt);                            // SampleFormat
    if (channels == 3)
      ifd_entry(f, 277, 3, 1, 3);                                   // SamplesPerPixel=3
    w32(f, next_ifd);

    // Rational extras: XRes=1/1, YRes=1/1
    w32(f, 1); w32(f, 1);
    w32(f, 1); w32(f, 1);
    if (channels == 3) { w16(f, bps); w16(f, bps); w16(f, bps); }
  }

  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// tiff_write_xyz — 3-channel float32 TIFF (tifxyz format)
//
// xyz: float[height * width * 3], channels=RGB=XYZ interleaved.
// Uses tiff_write_multipage with a single page, float32, 3 channels.
// ---------------------------------------------------------------------------

bool tiff_write_xyz(const char *path, const float *xyz, int width, int height) {
  if (!path || !xyz || width < 1 || height < 1) return false;
  const void *pages[1] = { xyz };
  return tiff_write_multipage(path, pages, 1, width, height, DTYPE_F32, 3);
}
