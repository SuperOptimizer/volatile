#define _POSIX_C_SOURCE 200809L

#include "gui/review_report.h"
#include "server/db.h"
#include "server/review.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char *status_label(review_status s) {
  switch (s) {
    case REVIEW_APPROVED:      return "Approved";
    case REVIEW_REJECTED:      return "Rejected";
    case REVIEW_NEEDS_CHANGES: return "Needs changes";
    case REVIEW_PENDING:
    default:                   return "Pending";
  }
}

static void fmt_time(char *buf, int bufsz, int64_t ts) {
  if (ts <= 0) { snprintf(buf, (size_t)bufsz, "—"); return; }
  time_t t = (time_t)ts;
  struct tm tm_buf;
  gmtime_r(&t, &tm_buf);
  strftime(buf, (size_t)bufsz, "%Y-%m-%d %H:%M UTC", &tm_buf);
}

// HTML-escape a string into buf. Returns buf.
static char *html_escape(const char *src, char *buf, int bufsz) {
  int out = 0;
  for (const char *p = src; *p && out < bufsz - 6; p++) {
    switch (*p) {
      case '<': memcpy(buf + out, "&lt;",  4); out += 4; break;
      case '>': memcpy(buf + out, "&gt;",  4); out += 4; break;
      case '&': memcpy(buf + out, "&amp;", 5); out += 5; break;
      case '"': memcpy(buf + out, "&quot;", 6); out += 6; break;
      default:  buf[out++] = *p; break;
    }
  }
  buf[out] = '\0';
  return buf;
}

// ---------------------------------------------------------------------------
// Per-row callback context
// ---------------------------------------------------------------------------

typedef struct {
  FILE          *f;
  bool           html;
  review_system *reviews;
  const char    *thumb_dir;
  int            count;
} report_ctx;

static bool write_segment_row(const segment_row *row, void *userdata) {
  report_ctx *ctx = userdata;
  ctx->count++;

  // Get review status
  review_status status = REVIEW_PENDING;
  if (ctx->reviews)
    status = review_get_status(ctx->reviews, row->id);

  char updated[32];
  fmt_time(updated, sizeof(updated), row->updated_at);

  if (ctx->html) {
    // Thumbnail path: <thumb_dir>/<id>.png  (optional)
    char thumb_cell[768] = "<td>—</td>";
    if (ctx->thumb_dir) {
      char thumb_path[512];
      snprintf(thumb_path, sizeof(thumb_path), "%s/%lld.png",
               ctx->thumb_dir, (long long)row->id);
      FILE *probe = fopen(thumb_path, "r");
      if (probe) {
        fclose(probe);
        char esc[512];
        html_escape(thumb_path, esc, sizeof(esc));
        snprintf(thumb_cell, sizeof(thumb_cell),
                 "<td><img src=\"%s\" width=\"64\" height=\"64\" alt=\"thumb\"></td>",
                 esc);
      }
    }

    const char *status_class =
      status == REVIEW_APPROVED      ? "approved"  :
      status == REVIEW_REJECTED      ? "rejected"  :
      status == REVIEW_NEEDS_CHANGES ? "changes"   : "pending";

    char name_esc[512], path_esc[512], vid_esc[256];
    html_escape(row->name,       name_esc, sizeof(name_esc));
    html_escape(row->surface_path, path_esc, sizeof(path_esc));
    html_escape(row->volume_id,  vid_esc,  sizeof(vid_esc));

    fprintf(ctx->f,
      "    <tr>\n"
      "      <td>%lld</td>\n"
      "      <td>%s</td>\n"
      "      <td>%s</td>\n"
      "      <td>%s</td>\n"
      "      <td class=\"%s\">%s</td>\n"
      "      <td>%s</td>\n"
      "      %s\n"
      "    </tr>\n",
      (long long)row->id,
      name_esc,
      vid_esc,
      path_esc,
      status_class, status_label(status),
      updated,
      thumb_cell);
  } else {
    fprintf(ctx->f, "  %-8lld  %-32s  %-24s  %-16s  %s\n",
            (long long)row->id,
            row->name,
            row->volume_id,
            status_label(status),
            updated);
  }
  return true;  // continue iteration
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool review_report_generate(const project *proj, const char *output_path, bool html) {
  if (!proj || !proj->db || !output_path) return false;

  FILE *f = fopen(output_path, "w");
  if (!f) {
    LOG_WARN("review_report_generate: cannot open %s for writing", output_path);
    return false;
  }

  report_ctx ctx = {
    .f         = f,
    .html      = html,
    .reviews   = proj->reviews,
    .thumb_dir = proj->thumb_dir,
    .count     = 0,
  };

  const char *proj_name = proj->name ? proj->name : "Volatile Project";

  if (html) {
    char esc_name[256];
    html_escape(proj_name, esc_name, sizeof(esc_name));
    fprintf(f,
      "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
      "  <meta charset=\"UTF-8\">\n"
      "  <title>Review Report — %s</title>\n"
      "  <style>\n"
      "    body { font-family: monospace; margin: 2em; }\n"
      "    h1   { font-size: 1.4em; }\n"
      "    table { border-collapse: collapse; width: 100%%; }\n"
      "    th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }\n"
      "    th { background: #f0f0f0; }\n"
      "    tr:nth-child(even) { background: #fafafa; }\n"
      "    .approved  { color: #186f1b; font-weight: bold; }\n"
      "    .rejected  { color: #b00000; font-weight: bold; }\n"
      "    .changes   { color: #9a6500; font-weight: bold; }\n"
      "    .pending   { color: #555; }\n"
      "  </style>\n"
      "</head>\n<body>\n"
      "<h1>Review Report — %s</h1>\n"
      "<table>\n"
      "  <thead><tr>\n"
      "    <th>ID</th><th>Name</th><th>Volume</th>"
      "<th>Surface path</th><th>Status</th><th>Last modified</th>",
      esc_name, esc_name);
    if (proj->thumb_dir) fputs("<th>Thumbnail</th>", f);
    fputs("\n  </tr></thead>\n  <tbody>\n", f);
  } else {
    char datebuf[32];
    time_t now = time(NULL);
    struct tm tm_buf;
    gmtime_r(&now, &tm_buf);
    strftime(datebuf, sizeof(datebuf), "%Y-%m-%d %H:%M UTC", &tm_buf);

    fprintf(f, "Review Report — %s\nGenerated: %s\n\n", proj_name, datebuf);
    fprintf(f, "  %-8s  %-32s  %-24s  %-16s  %s\n",
            "ID", "Name", "Volume", "Status", "Last modified");
    fprintf(f, "  %s\n", "-----------------------------------------------------------------------"
                          "----------------------------");
  }

  seg_db_list_all_segments(proj->db, write_segment_row, &ctx);

  if (html) {
    fprintf(f, "  </tbody>\n</table>\n<p>%d segment(s) total.</p>\n</body>\n</html>\n",
            ctx.count);
  } else {
    fprintf(f, "\n%d segment(s) total.\n", ctx.count);
  }

  fclose(f);
  LOG_INFO("review_report_generate: wrote %d segments to %s", ctx.count, output_path);
  return true;
}
