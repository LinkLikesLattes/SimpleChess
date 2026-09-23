# =============================================================================
# Simple Chess — UCI engine build
#
# Primary build path (no CMake required). Tuned for Apple Silicon by default.
#
#   make            # optimized native build -> ./simplechess
#   make debug      # -O0 -g with assertions and sanitizers
#   make run        # build, then launch the engine (UCI on stdin)
#   make clean      # remove build artifacts
#   make version    # print the current version string
#   make save/promote/restore/versions   # version-archive wrappers (dev tree only)
#
# Override the compiler or flags from the command line, e.g.:
#   make CXX=g++-14
#   make ARCH="-mcpu=apple-m2"
# =============================================================================

CXX      ?= clang++
EXE      ?= simplechess

SRC_DIR   := src
# Debug and release keep separate object dirs so switching targets never links
# stale objects built with the other configuration's flags.
BUILD_DIR ?= build
LIB_INC   := external/chess-library/include

# Vendored Fathom (Syzygy probing): header-only from the build's view — src/syzygy.cpp
# is the single TU that #includes tbprobe.c (which pulls in tbchess.c). We only add
# its include path here. See external/Fathom/src/LICENSE (MIT).
FATHOM_INC := external/Fathom/src

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)

# ---- Version identity -------------------------------------------------------
# The version lives in the top-level VERSION file (single source of truth) and
# is compiled into the binary so it self-reports via UCI `id name`. Injected
# via a dedicated variable (not CXXFLAGS) so it survives the `debug` target's
# full CXXFLAGS override too.
VERSION      := $(shell cat VERSION 2>/dev/null || echo dev)
BUILD_DATE   := $(shell date +%Y-%m-%d)
# Filesystem spelling of the version (0.1 -> 0-1), used for asset filenames such
# as the NNUE net (SCNNUEv0-1.scn). Keeping the version's dot out of the name
# leaves exactly one dot in the file, so ".scn" is unambiguously the extension
# no matter how a tool splits it. Derived from VERSION, never written by hand.
VERSION_FS   := $(subst .,-,$(VERSION))
# NNUE nets are keyed by MAJOR only (SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5): a MAJOR bump
# is an architecture/format change, while retraining does NOT bump the version, so a
# MAJOR owns many dated nets and the engine discovers the newest one at startup
# (src/uci.cpp). The build only needs the MAJOR; the date lives in the filename.
NET_MAJOR := $(word 1,$(subst ., ,$(VERSION)))
VERSION_DEFS := -DSC_VERSION='"$(VERSION)"' -DSC_VERSION_FS='"$(VERSION_FS)"' \
                -DSC_NET_MAJOR=$(NET_MAJOR) \
                -DSC_BUILD_DATE='"$(BUILD_DATE)"'

# Apple Silicon: -mcpu=native lets clang target this exact core (M1/M2/M3/M4).
# On x86 clang, -mcpu= is only an -mtune alias (no ISA level!), so a bare `make`
# there would silently emit an SSE2 binary; pick -march=native on x86 hosts.
ARCH     ?= $(if $(filter x86_64 amd64,$(shell uname -m)),-march=native,-mcpu=native)

STD       := -std=c++20
# -Wshadow is intentionally omitted: the vendored chess-library header trips it.
WARN      := -Wall -Wextra
INCLUDES  := -I$(SRC_DIR) -I$(LIB_INC) -I$(FATHOM_INC)

# EXTRA hooks in ad-hoc defines without replacing the whole flag set, e.g.
#   make EXTRA="-DSC_KATT_SCALE=50" BUILD_DIR=... EXE=...   (weight experiments)
# EXTRA also survives profile-build: the PGO phases append their flags to it,
# so `make profile-build EXTRA="-DSC_X=1" ...` really compiles with SC_X=1.
EXTRA     ?=
# EMBED_NET=<path/to/SCNNUEv3-<date>.scn5> bakes that net into the executable (src/embed_net.cpp,
# an .incbin in read-only data) so a single file runs with no net beside it -- the form phone
# front-ends and tournament harnesses need. A newer dated net on disk still wins at startup and
# EvalFile still overrides. Absolute path: under LTO the .incbin is resolved at link time.
EMBED_NET ?=
# The byte count is taken from the file here rather than from an end label in the assembly: a
# Mach-O linker may reorder a second label's atom, and a size symbol must not depend on layout.
EMBED_DEFS := $(if $(EMBED_NET),-DSC_EMBED_NET_PATH='"$(abspath $(EMBED_NET))"' -DSC_EMBED_NET_NAME='"$(notdir $(EMBED_NET))"' -DSC_EMBED_NET_SIZE=$(strip $(shell wc -c < $(EMBED_NET))),)
CXXFLAGS  ?= $(STD) -O3 -DNDEBUG -flto $(ARCH) -funroll-loops $(WARN) $(INCLUDES) $(EXTRA)
LDFLAGS   ?= -flto -pthread

# ---- Debug build overrides --------------------------------------------------
DBGFLAGS  := $(STD) -O0 -g $(WARN) $(INCLUDES) -fsanitize=address,undefined
DBGLD     := -pthread -fsanitize=address,undefined

.PHONY: all debug run clean version save versions profile-build

all: $(EXE)

$(EXE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(VERSION_DEFS) $(EMBED_DEFS) -MMD -MP -c $< -o $@

# Rebuild when the version string or the build recipe changes so the baked-in
# identity never goes stale.
$(OBJECTS): VERSION Makefile
ifneq ($(EMBED_NET),)
$(BUILD_DIR)/embed_net.o: $(EMBED_NET)
endif

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

debug:
	$(MAKE) BUILD_DIR=build-debug EXE=simplechess-debug \
	        CXXFLAGS="$(DBGFLAGS)" LDFLAGS="$(DBGLD)" all

run: all
	./$(EXE)

# ---- Profile-guided optimization (PGO) --------------------------------------
# Keeps the LTO baseline and layers profile feedback on top (3 phases:
# instrument -> run workload -> rebuild with -fprofile-use). Measured at or above
# the plain LTO build, so it's the kept build for the engine. Provide a net so the
# NNUE paths get profiled; weights don't matter, only executed code paths.
#   make profile-build PGO_NET=nets/SCNNUEv3-2026-09-12.scn5
LLVM_PROFDATA ?= $(shell xcrun --find llvm-profdata 2>/dev/null || echo llvm-profdata)
# Sibling of BUILD_DIR (NOT nested) so the phase-3 "rm -rf BUILD_DIR" cannot wipe
# the profile we just merged.
PGO_DIR       ?= $(BUILD_DIR)-pgo
PGO_NET       ?=
# The workload may sit in tools/ or tools/speed/; take whichever exists.
PGO_WORKLOAD  ?= $(firstword $(wildcard tools/speed/pgo_workload.py tools/pgo_workload.py) tools/pgo_workload.py)
PGO_DEPTH     ?=
PGO_FENS      ?=
# The workload needs python-chess; override when `python3` on PATH lacks it
# (e.g. MSYS2's own python): make profile-build PYTHON=/path/to/python.exe ...
PYTHON        ?= python3
# Native (non-MSYS) clang/llvm-profdata need a Windows-style path for the
# profile dir; inside an MSYS2 shell $(MSYSTEM) is set, so convert there.
PGO_ABS       := $(if $(MSYSTEM),$(shell cygpath -m $(abspath $(PGO_DIR))),$(abspath $(PGO_DIR)))

profile-build:
	@test -n "$(PGO_NET)" || { echo "ERROR: profile-build needs PGO_NET=path/to/net"; exit 1; }
	@rm -rf $(BUILD_DIR) $(EXE) $(PGO_DIR)
	@mkdir -p $(PGO_DIR)
	@echo ">> PGO 1/3: instrumented build (-fprofile-generate)"
	@$(MAKE) SRC_DIR=$(SRC_DIR) BUILD_DIR=$(BUILD_DIR) EXE=$(EXE) \
	         EXTRA="$(EXTRA) -fprofile-generate=$(PGO_ABS)" \
	         LDFLAGS="$(LDFLAGS) -fprofile-generate=$(PGO_ABS)" all
	@echo ">> PGO 2/3: training workload"
	@LLVM_PROFILE_FILE="$(PGO_ABS)/prof-%p-%m.profraw" PGO_DEPTH="$(PGO_DEPTH)" PGO_FENS="$(PGO_FENS)" \
	   $(PYTHON) $(PGO_WORKLOAD) ./$(EXE) $(PGO_NET)
	@$(LLVM_PROFDATA) merge -output=$(PGO_ABS)/prof.profdata $(PGO_ABS)/*.profraw
	@echo ">> PGO 3/3: rebuild (-fprofile-use)"
	@rm -rf $(BUILD_DIR) $(EXE)
	@$(MAKE) SRC_DIR=$(SRC_DIR) BUILD_DIR=$(BUILD_DIR) EXE=$(EXE) \
	         EXTRA="$(EXTRA) -fprofile-use=$(PGO_ABS)/prof.profdata -Wno-profile-instr-out-of-date -Wno-profile-instr-unprofiled" \
	         LDFLAGS="$(LDFLAGS) -fprofile-use=$(PGO_ABS)/prof.profdata" all
	@echo ">> PGO build complete: ./$(EXE)"

version:
	@echo "Simple Chess $(VERSION)"

clean:
	@rm -rf build build-debug simplechess simplechess-debug

# ---- Versioning convenience wrappers (dev tree only: need tools/release/version.py) ----
ifneq ($(wildcard tools/release/version.py),)
save:
	@python3 tools/release/version.py save $(NAME) $(if $(MSG),-m "$(MSG)",)

promote:
	@python3 tools/release/version.py promote $(NAME)

restore:
	@python3 tools/release/version.py restore $(NAME)

versions:
	@python3 tools/release/version.py list
endif

-include $(DEPS)
