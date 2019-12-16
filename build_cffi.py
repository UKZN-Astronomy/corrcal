import os

from cffi import FFI


ffi = FFI()
LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, "corrcal", "src")
include_dirs = [CLOC]

# =================================================================
# Set compilation arguments dependent on environment... a bit buggy
# Not really how to set this up properly yet ...
# in macOS, gcc is symlink to clang, which does not have built-in
# openMP support. Users have to install the LLVM openMP library
# to get openMP to work with clang. See https://openmp.llvm.org/,
# (although installing through homebrew is most convenient,
#  see https://iscinumpy.gitlab.io/post/omp-on-high-sierra/).
#
# Is there a way to check an automate this?
# =================================================================
# if "DEBUG" in os.environ:
#     extra_compile_args = ["-fopenmp", "-w", "-g", "-O0"]
# else:
#     extra_compile_args = ["-fopenmp", "-Ofast", "-w"]
extra_compile_args = ["-std=c99"]


# # Set the C-code logging level.
# # If DEBUG is set, we default to the highest level, but if not,
# # we set it to the level just above no logging at all.
# log_level = os.environ.get("LOG_LEVEL", 3 if "DEBUG" in os.environ else 1)
# available_levels = [
#     "NONE",
#     "ERROR",
#     "WARNING",
#     "INFO",
#     "DEBUG",
#     "SUPER_DEBUG",
#     "ULTRA_DEBUG",
# ]
#
#
# if isinstance(log_level, str) and log_level.upper() in available_levels:
#     log_level = available_levels.index(log_level.upper())
#
# try:
#     log_level = int(log_level)
# except ValueError:
#     # note: for py35 support, can't use f strings.
#     raise ValueError(
#         "LOG_LEVEL must be specified as a positive integer, or one "
#         "of {}".format(available_levels)
#     )
# =================================================================

library_dirs = []
for k, v in os.environ.items():
    if "inc" in k.lower():
        include_dirs += [v]
    elif "lib" in k.lower():
        library_dirs += [v]

# =================================================================

# This is the Header file
with open(os.path.join(CLOC, "corrcal_c_funcs.h")) as f:
    ffi.cdef(f.read())

# This is the overall C code.
ffi.set_source(
    "corrcal.c_corrcal",  # Name/Location of shared library module
    r"""
        #include "corrcal_c_funcs.c"
    """,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["m"],
    extra_compile_args=extra_compile_args,
    extra_link_args=["-fopenmp"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
