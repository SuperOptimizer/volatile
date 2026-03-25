#include "greatest.h"
#include "server/db.h"

#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Open a fresh in-memory database for each test.
static seg_db *open_mem(void) {
  return seg_db_open(":memory:");
}

// ---------------------------------------------------------------------------
// Open / close
// ---------------------------------------------------------------------------

TEST test_open_close(void) {
  seg_db *db = open_mem();
  ASSERT(db != NULL);
  seg_db_close(db);
  PASS();
}

TEST test_close_null(void) {
  seg_db_close(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Segment CRUD
// ---------------------------------------------------------------------------

TEST test_insert_segment_returns_id(void) {
  seg_db *db = open_mem();
  ASSERT(db != NULL);
  int64_t id = seg_db_insert_segment(db, "vol-1", "seg-alpha", "/surfaces/a.obj");
  ASSERT(id > 0);
  seg_db_close(db);
  PASS();
}

TEST test_insert_multiple_segments_unique_ids(void) {
  seg_db *db = open_mem();
  int64_t id1 = seg_db_insert_segment(db, "vol-1", "seg-a", "");
  int64_t id2 = seg_db_insert_segment(db, "vol-1", "seg-b", "");
  ASSERT(id1 > 0);
  ASSERT(id2 > 0);
  ASSERT(id1 != id2);
  seg_db_close(db);
  PASS();
}

TEST test_get_segment_roundtrip(void) {
  seg_db *db = open_mem();
  int64_t id = seg_db_insert_segment(db, "vol-xyz", "my-seg", "/path/to/surf.obj");
  ASSERT(id > 0);

  segment_row row = {0};
  ASSERT(seg_db_get_segment(db, id, &row));
  ASSERT_EQ(id, row.id);
  ASSERT_STR_EQ("vol-xyz",           row.volume_id);
  ASSERT_STR_EQ("my-seg",            row.name);
  ASSERT_STR_EQ("/path/to/surf.obj", row.surface_path);
  ASSERT(row.created_at > 0);
  ASSERT(row.updated_at > 0);

  seg_db_close(db);
  PASS();
}

TEST test_get_segment_not_found(void) {
  seg_db *db = open_mem();
  segment_row row = {0};
  ASSERT(!seg_db_get_segment(db, 9999, &row));
  seg_db_close(db);
  PASS();
}

// Accumulate ids from list callback.
typedef struct { int64_t ids[16]; int count; } id_list;

static bool collect_seg_ids(const segment_row *row, void *ud) {
  id_list *l = ud;
  if (l->count < 16) l->ids[l->count++] = row->id;
  return true;
}

TEST test_list_segments_by_volume(void) {
  seg_db *db = open_mem();
  int64_t a = seg_db_insert_segment(db, "vol-A", "seg-1", "");
  int64_t b = seg_db_insert_segment(db, "vol-A", "seg-2", "");
  /*int64_t c =*/ seg_db_insert_segment(db, "vol-B", "seg-3", "");

  id_list found = {0};
  ASSERT(seg_db_list_segments(db, "vol-A", collect_seg_ids, &found));
  ASSERT_EQ(2, found.count);
  // IDs should be a and b (order: ascending by id)
  ASSERT((found.ids[0] == a && found.ids[1] == b) ||
         (found.ids[0] == b && found.ids[1] == a));

  id_list found_b = {0};
  ASSERT(seg_db_list_segments(db, "vol-B", collect_seg_ids, &found_b));
  ASSERT_EQ(1, found_b.count);

  seg_db_close(db);
  PASS();
}

TEST test_list_segments_empty_volume(void) {
  seg_db *db = open_mem();
  id_list found = {0};
  ASSERT(seg_db_list_segments(db, "no-such-vol", collect_seg_ids, &found));
  ASSERT_EQ(0, found.count);
  seg_db_close(db);
  PASS();
}

TEST test_delete_segment(void) {
  seg_db *db = open_mem();
  int64_t id = seg_db_insert_segment(db, "vol-1", "to-delete", "");
  ASSERT(id > 0);
  ASSERT(seg_db_delete_segment(db, id));

  segment_row row = {0};
  ASSERT(!seg_db_get_segment(db, id, &row));
  seg_db_close(db);
  PASS();
}

TEST test_delete_segment_nonexistent(void) {
  seg_db *db = open_mem();
  ASSERT(!seg_db_delete_segment(db, 9999));
  seg_db_close(db);
  PASS();
}

// ---------------------------------------------------------------------------
// Annotation CRUD
// ---------------------------------------------------------------------------

TEST test_insert_annotation_returns_id(void) {
  seg_db *db = open_mem();
  int64_t seg = seg_db_insert_segment(db, "vol-1", "seg", "");
  int64_t ann = seg_db_insert_annotation(db, seg, "comment", "{\"text\":\"hello\"}");
  ASSERT(ann > 0);
  seg_db_close(db);
  PASS();
}

typedef struct { int count; char last_type[64]; char *last_json; } ann_list;

static bool collect_ann(const annotation_row *row, void *ud) {
  ann_list *l = ud;
  l->count++;
  strncpy(l->last_type, row->type, sizeof(l->last_type) - 1);
  free(l->last_json);
  l->last_json = strdup(row->data_json ? row->data_json : "");
  return true;
}

TEST test_list_annotations_roundtrip(void) {
  seg_db *db  = open_mem();
  int64_t seg = seg_db_insert_segment(db, "vol-1", "seg", "");
  seg_db_insert_annotation(db, seg, "note",    "{\"msg\":\"first\"}");
  seg_db_insert_annotation(db, seg, "flag",    "{\"priority\":1}");

  ann_list found = {0};
  ASSERT(seg_db_list_annotations(db, seg, collect_ann, &found));
  ASSERT_EQ(2, found.count);
  free(found.last_json);
  seg_db_close(db);
  PASS();
}

TEST test_list_annotations_empty(void) {
  seg_db *db  = open_mem();
  int64_t seg = seg_db_insert_segment(db, "vol-1", "seg", "");

  ann_list found = {0};
  ASSERT(seg_db_list_annotations(db, seg, collect_ann, &found));
  ASSERT_EQ(0, found.count);
  seg_db_close(db);
  PASS();
}

TEST test_delete_annotation(void) {
  seg_db *db  = open_mem();
  int64_t seg = seg_db_insert_segment(db, "vol-1", "seg", "");
  int64_t ann = seg_db_insert_annotation(db, seg, "note", "{}");
  ASSERT(seg_db_delete_annotation(db, ann));

  ann_list found = {0};
  seg_db_list_annotations(db, seg, collect_ann, &found);
  ASSERT_EQ(0, found.count);
  seg_db_close(db);
  PASS();
}

TEST test_delete_annotation_nonexistent(void) {
  seg_db *db = open_mem();
  ASSERT(!seg_db_delete_annotation(db, 9999));
  seg_db_close(db);
  PASS();
}

TEST test_annotation_row_free_null(void) {
  annotation_row_free(NULL);  // must not crash
  PASS();
}

TEST test_annotation_null_json_defaults(void) {
  seg_db *db  = open_mem();
  int64_t seg = seg_db_insert_segment(db, "vol-1", "seg", "");
  int64_t ann = seg_db_insert_annotation(db, seg, "note", NULL);
  ASSERT(ann > 0);

  ann_list found = {0};
  ASSERT(seg_db_list_annotations(db, seg, collect_ann, &found));
  ASSERT_EQ(1, found.count);
  // NULL data_json should default to "{}"
  ASSERT_STR_EQ("{}", found.last_json);
  free(found.last_json);
  seg_db_close(db);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(db_suite) {
  RUN_TEST(test_open_close);
  RUN_TEST(test_close_null);

  RUN_TEST(test_insert_segment_returns_id);
  RUN_TEST(test_insert_multiple_segments_unique_ids);
  RUN_TEST(test_get_segment_roundtrip);
  RUN_TEST(test_get_segment_not_found);
  RUN_TEST(test_list_segments_by_volume);
  RUN_TEST(test_list_segments_empty_volume);
  RUN_TEST(test_delete_segment);
  RUN_TEST(test_delete_segment_nonexistent);

  RUN_TEST(test_insert_annotation_returns_id);
  RUN_TEST(test_list_annotations_roundtrip);
  RUN_TEST(test_list_annotations_empty);
  RUN_TEST(test_delete_annotation);
  RUN_TEST(test_delete_annotation_nonexistent);
  RUN_TEST(test_annotation_row_free_null);
  RUN_TEST(test_annotation_null_json_defaults);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(db_suite);
  GREATEST_MAIN_END();
}
