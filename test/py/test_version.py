from __future__ import annotations

def test_version():
  import volatile
  assert volatile.__version__ == "0.1.0"

def test_log_level():
  import volatile
  volatile.log_set_level(2)  # WARN
  assert volatile.log_get_level() == 2
  volatile.log_set_level(1)  # reset to INFO
