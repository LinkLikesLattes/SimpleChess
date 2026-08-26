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
#   make save NAME=v0.2 MSG="..."   # archive this build (see tools/version.py)
#   make promote NAME=v0.2          # v0.2 becomes ./simplechess (see tools/version.py)
#   make restore NAME=v0.2          # v0.2's archived source overwrites the working tree
#   make versions   # list archived versions
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
# NNUE nets track MAJOR.MINOR only: a patch/hotfix bump (1.9 -> 1.9.1) does not
# change the net, so the default net name must drop the patch component and both
# 1.9 and 1.9.8 look for SCNNUEv1-9.scn3. Take the first two dot-separated fields.
NET_VERSION_FS := $(word 1,$(subst ., ,$(VERSION)))-$(word 2,$(subst ., ,$(VERSION)))
VERSION_DEFS := -DSC_VERSION='"$(VERSION)"' -DSC_VERSION_FS='"$(VERSION_FS)"' \
                -DSC_NET_VERSION_FS='"$(NET_VERSION_FS)"' \
                -DSC_BUILD_DATE='"$(BUILD_DATE)"'

# Apple Silicon: -mcpu=native lets clang target this exact core (M1/M2/M3/M4).
ARCH     ?= -mcpu=native

STD       := -std=c++20
# -Wshadow is intentionally omitted: the vendored chess-library header trips it.
WARN      := -Wall -Wextra
INCLUDES  := -I$(SRC_DIR) -I$(LIB_INC)

# EXTRA hooks in ad-hoc defines without replacing the whole flag set, e.g.
#   make EXTRA="-DSC_KATT_SCALE=50" BUILD_DIR=... EXE=...   (weight experiments)
EXTRA     ?=
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
	$(CXX) $(CXXFLAGS) $(VERSION_DEFS) -MMD -MP -c $< -o $@

# Rebuild when the version string or the build recipe changes so the baked-in
# identity never goes stale.
$(OBJECTS): VERSION Makefile

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
#   make profile-build PGO_NET=nets/SCNNUEv2-5.scn5
LLVM_PROFDATA ?= $(shell xcrun --find llvm-profdata 2>/dev/null || echo llvm-profdata)
# Sibling of BUILD_DIR (NOT nested) so the phase-3 "rm -rf BUILD_DIR" cannot wipe
# the profile we just merged.
PGO_DIR       ?= $(BUILD_DIR)-pgo
PGO_NET       ?=
PGO_WORKLOAD  ?= tools/pgo_workload.py
PGO_ABS       := $(abspath $(PGO_DIR))

profile-build:
	@test -n "$(PGO_NET)" || { echo "ERROR: profile-build needs PGO_NET=path/to/net"; exit 1; }
	@rm -rf $(BUILD_DIR) $(EXE) $(PGO_DIR)
	@mkdir -p $(PGO_DIR)
	@echo ">> PGO 1/3: instrumented build (-fprofile-generate)"
	@$(MAKE) SRC_DIR=$(SRC_DIR) BUILD_DIR=$(BUILD_DIR) EXE=$(EXE) \
	         EXTRA="-fprofile-generate=$(PGO_ABS)" \
	         LDFLAGS="$(LDFLAGS) -fprofile-generate=$(PGO_ABS)" all
	@echo ">> PGO 2/3: training workload"
	@LLVM_PROFILE_FILE="$(PGO_ABS)/prof-%p-%m.profraw" \
	   python3 $(PGO_WORKLOAD) ./$(EXE) $(PGO_NET)
	@$(LLVM_PROFDATA) merge -output=$(PGO_ABS)/prof.profdata $(PGO_ABS)/*.profraw
	@echo ">> PGO 3/3: rebuild (-fprofile-use)"
	@rm -rf $(BUILD_DIR) $(EXE)
	@$(MAKE) SRC_DIR=$(SRC_DIR) BUILD_DIR=$(BUILD_DIR) EXE=$(EXE) \
	         EXTRA="-fprofile-use=$(PGO_ABS)/prof.profdata -Wno-profile-instr-out-of-date -Wno-profile-instr-unprofiled" \
	         LDFLAGS="$(LDFLAGS) -fprofile-use=$(PGO_ABS)/prof.profdata" all
	@echo ">> PGO build complete: ./$(EXE)"

# ---- Versioning convenience wrappers (see tools/version.py) ------------------
version:
	@echo "Simple Chess $(VERSION)"

save:
	@python3 tools/version.py save $(NAME) $(if $(MSG),-m "$(MSG)",)

promote:
	@python3 tools/version.py promote $(NAME)

restore:
	@python3 tools/version.py restore $(NAME)

versions:
	@python3 tools/version.py list

clean:
	@rm -rf build build-debug simplechess simplechess-debug

-include $(DEPS)
