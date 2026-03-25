import os
from setuptools import setup, Extension

# Paths to the cmake build outputs. Default to ../build; override with
# VOLATILE_BUILD_DIR env var so CI can point at a different tree.
_build = os.environ.get("VOLATILE_BUILD_DIR",
                        os.path.join(os.path.dirname(__file__), "..", "build"))
_build = os.path.abspath(_build)

_src        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_blosc2_inc = os.path.join(_build, "_deps", "blosc2-src", "include")
_blosc2_lib = os.path.join(_build, "_deps", "blosc2-build", "blosc")
_core_lib   = os.path.join(_build, "src", "core")

volatile_core = Extension(
  "volatile._core",
  sources=["volatile/_core.c"],
  include_dirs=[_src, _blosc2_inc],
  extra_link_args=[
    "-Wl,--whole-archive",
    os.path.join(_core_lib, "libvolatile_core.a"),
    os.path.join(_blosc2_lib, "libblosc2.a"),
    "-Wl,--no-whole-archive",
    "-lm", "-lcurl",
  ],
)

setup(ext_modules=[volatile_core])
