#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "gui/neural_trace.h"
#include "core/log.h"
#include "core/json.h"

#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define SOCK_PATH_MAX    108
#define RECV_BUF        (1 << 20)   // 1 MiB
#define CONNECT_RETRIES  20
#define CONNECT_DELAY_US 50000      // 50 ms per retry

struct neural_tracer {
  char  *model_path;
  char   sock_path[SOCK_PATH_MAX];
  pid_t  child_pid;
  int    sock_fd;
};

neural_tracer *neural_tracer_new(const char *model_path) {
  neural_tracer *t = calloc(1, sizeof(*t));
  if (!t) return NULL;
  t->model_path = strdup(model_path ? model_path : "");
  t->sock_fd    = -1;
  snprintf(t->sock_path, sizeof(t->sock_path),
           "/tmp/volatile_neural_%d.sock", (int)getpid());
  return t;
}

void neural_tracer_free(neural_tracer *t) {
  if (!t) return;
  if (neural_tracer_is_running(t)) neural_tracer_stop(t);
  free(t->model_path);
  free(t);
}

bool neural_tracer_is_running(const neural_tracer *t) {
  if (!t || t->child_pid == 0) return false;
  return waitpid(t->child_pid, NULL, WNOHANG) == 0;
}

bool neural_tracer_start(neural_tracer *t) {
  if (!t) return false;
  if (neural_tracer_is_running(t)) return true;

  unlink(t->sock_path);

  pid_t pid = fork();
  if (pid < 0) { LOG_ERROR("neural_tracer_start: fork: %s", strerror(errno)); return false; }
  if (pid == 0) {
    execl("/usr/bin/python3", "python3", "-m", "volatile.ml.service",
          "--model", t->model_path, "--socket", t->sock_path, (char *)NULL);
    _exit(127);
  }
  t->child_pid = pid;
  LOG_INFO("neural_tracer_start: pid=%d socket=%s", (int)pid, t->sock_path);

  int fd = -1;
  for (int i = 0; i < CONNECT_RETRIES; i++) {
    usleep(CONNECT_DELAY_US);
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) continue;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, t->sock_path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) break;
    close(fd); fd = -1;
  }
  if (fd < 0) {
    LOG_ERROR("neural_tracer_start: could not connect to service");
    kill(t->child_pid, SIGTERM);
    waitpid(t->child_pid, NULL, 0);
    t->child_pid = 0;
    return false;
  }
  t->sock_fd = fd;
  return true;
}

bool neural_tracer_stop(neural_tracer *t) {
  if (!t || t->child_pid == 0) return true;
  if (t->sock_fd >= 0) {
    const char *m = "{\"type\":\"shutdown\"}\n";
    write(t->sock_fd, m, strlen(m));
    close(t->sock_fd);
    t->sock_fd = -1;
  }
  int st = 0;
  if (waitpid(t->child_pid, &st, WNOHANG) == 0) {
    usleep(100000);
    if (waitpid(t->child_pid, &st, WNOHANG) == 0) {
      kill(t->child_pid, SIGTERM);
      waitpid(t->child_pid, &st, 0);
    }
  }
  t->child_pid = 0;
  unlink(t->sock_path);
  return true;
}

// Read one newline-terminated JSON response from fd into a heap buffer.
static char *recv_line(int fd) {
  char *buf = malloc(RECV_BUF);
  if (!buf) return NULL;
  ssize_t total = 0;
  while (total < (ssize_t)RECV_BUF - 1) {
    ssize_t n = read(fd, buf + total, (size_t)(RECV_BUF - 1 - total));
    if (n <= 0) break;
    total += n;
    if (memchr(buf, '\n', (size_t)total)) break;
  }
  buf[total] = '\0';
  return buf;
}

trace_result *neural_tracer_predict(neural_tracer *t,
                                    const float *patch, int d, int h, int w,
                                    const vec3f *current_points, int npoints) {
  if (!t || t->sock_fd < 0 || !patch || !current_points) return NULL;

  // Protocol: JSON header line, then raw patch bytes, then raw point bytes.
  char hdr[256];
  int hl = snprintf(hdr, sizeof(hdr),
    "{\"type\":\"predict\",\"shape\":[%d,%d,%d],\"npoints\":%d}\n", d, h, w, npoints);
  if (write(t->sock_fd, hdr, (size_t)hl) < 0) return NULL;
  if (write(t->sock_fd, patch, (size_t)(d * h * w) * sizeof(float)) < 0) return NULL;
  if (write(t->sock_fd, current_points, (size_t)npoints * sizeof(vec3f)) < 0) return NULL;

  char *resp = recv_line(t->sock_fd);
  if (!resp) return NULL;
  json_value *root = json_parse(resp);
  free(resp);
  if (!root) { LOG_ERROR("neural_tracer_predict: bad JSON"); return NULL; }

  const json_value *arr = json_object_get(root, "displacements");
  if (!arr || json_typeof(arr) != JSON_ARRAY) { json_free(root); return NULL; }

  int n = (int)json_array_len(arr);
  trace_result *r = calloc(1, sizeof(*r));
  r->displacements = calloc((size_t)(n > 0 ? n : 1), sizeof(vec3f));
  if (!r || !r->displacements) { free(r); json_free(root); return NULL; }
  r->count = n;
  for (int i = 0; i < n; i++) {
    const json_value *v = json_array_get(arr, (size_t)i);
    if (!v || json_typeof(v) != JSON_ARRAY || json_array_len(v) < 3) continue;
    r->displacements[i] = (vec3f){
      (float)json_get_number(json_array_get(v, 0), 0.0),
      (float)json_get_number(json_array_get(v, 1), 0.0),
      (float)json_get_number(json_array_get(v, 2), 0.0),
    };
  }
  json_free(root);
  return r;
}

void trace_result_free(trace_result *r) {
  if (!r) return;
  free(r->displacements);
  free(r);
}

struct opt_service {
  int  sock_fd;
  char status[64];
};

opt_service *opt_service_connect(const char *host, int port) {
  if (!host || port <= 0) return NULL;
  char ps[16];
  snprintf(ps, sizeof(ps), "%d", port);
  struct addrinfo hints = {0}, *res = NULL;
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host, ps, &hints, &res) != 0 || !res) {
    LOG_ERROR("opt_service_connect: getaddrinfo %s:%d failed", host, port);
    return NULL;
  }
  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) { freeaddrinfo(res); return NULL; }
  if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
    LOG_ERROR("opt_service_connect: %s:%d: %s", host, port, strerror(errno));
    close(fd); freeaddrinfo(res); return NULL;
  }
  freeaddrinfo(res);
  opt_service *s = calloc(1, sizeof(*s));
  if (!s) { close(fd); return NULL; }
  s->sock_fd = fd;
  strncpy(s->status, "connected", sizeof(s->status) - 1);
  return s;
}

void opt_service_free(opt_service *s) {
  if (!s) return;
  if (s->sock_fd >= 0) close(s->sock_fd);
  free(s);
}

bool opt_service_submit(opt_service *s, quad_surface *surface,
                        const char *params_json) {
  if (!s || s->sock_fd < 0 || !surface) return false;
  const char *p = params_json ? params_json : "{}";
  char hdr[512];
  int hl = snprintf(hdr, sizeof(hdr),
    "{\"type\":\"submit\",\"rows\":%d,\"cols\":%d,\"params\":%s}\n",
    surface->rows, surface->cols, p);
  if (write(s->sock_fd, hdr, (size_t)hl) < 0) return false;
  size_t pts = (size_t)(surface->rows * surface->cols) * sizeof(vec3f);
  if (write(s->sock_fd, surface->points, pts) < 0) return false;
  strncpy(s->status, "pending", sizeof(s->status) - 1);
  return true;
}

bool opt_service_poll_status(opt_service *s, char *status_out, int maxlen) {
  if (!s || s->sock_fd < 0 || !status_out || maxlen <= 0) return false;
  const char *req = "{\"type\":\"status\"}\n";
  if (write(s->sock_fd, req, strlen(req)) < 0) return false;
  char *resp = recv_line(s->sock_fd);
  if (!resp) return false;
  json_value *root = json_parse(resp);
  free(resp);
  if (!root) return false;
  const json_value *sv = json_object_get(root, "status");
  const char *str = sv ? json_get_str(sv) : NULL;
  if (str) {
    strncpy(s->status, str, sizeof(s->status) - 1);
    strncpy(status_out, str, (size_t)maxlen - 1);
    status_out[maxlen - 1] = '\0';
  }
  json_free(root);
  return str != NULL;
}

quad_surface *opt_service_get_result(opt_service *s) {
  if (!s || s->sock_fd < 0 || strcmp(s->status, "done") != 0) return NULL;
  const char *req = "{\"type\":\"result\"}\n";
  if (write(s->sock_fd, req, strlen(req)) < 0) return NULL;
  char *resp = recv_line(s->sock_fd);
  if (!resp) return NULL;
  json_value *root = json_parse(resp);
  free(resp);
  if (!root) return NULL;

  const json_value *rv = json_object_get(root, "rows");
  const json_value *cv = json_object_get(root, "cols");
  const json_value *pv = json_object_get(root, "points");
  int rows = rv ? (int)json_get_int(rv, 0) : 0;
  int cols = cv ? (int)json_get_int(cv, 0) : 0;
  if (rows <= 0 || cols <= 0 || !pv || json_typeof(pv) != JSON_ARRAY) {
    json_free(root); return NULL;
  }

  quad_surface *surf = quad_surface_new(rows, cols);
  if (!surf) { json_free(root); return NULL; }

  size_t n = json_array_len(pv);
  for (size_t i = 0; i < n && (int)i < rows * cols; i++) {
    const json_value *pt = json_array_get(pv, i);
    if (!pt || json_array_len(pt) < 3) continue;
    surf->points[i] = (vec3f){
      (float)json_get_number(json_array_get(pt, 0), 0.0),
      (float)json_get_number(json_array_get(pt, 1), 0.0),
      (float)json_get_number(json_array_get(pt, 2), 0.0),
    };
  }
  json_free(root);
  return surf;
}
