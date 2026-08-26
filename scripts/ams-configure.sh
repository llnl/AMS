#!/usr/bin/env bash
#
# ams-configure.sh — assemble and run the CMake configure step for AMS.
#
# We have two dependency-provisioning paths:
#   * On an LLNL Livermore Computing (LC) cluster it sources the repo's own
#     scripts/gitlab/setup-env.sh (internal Spack env) and maps the exported
#     AMS_*_PATH variables onto the right -D..._DIR hints.
#   * Elsewhere it relies on *_DIR / CMAKE_PREFIX_PATH you set yourself
#     (see references/manual-deps.md).
#
# The full cmake command is always printed before it runs. Use --dry-run to
# print without configuring.
#
# Run from the AMS repo root, or pass --src <path-to-AMS>.

set -euo pipefail

# ---- defaults --------------------------------------------------------------
SRC="."
BUILD="build"
BUILD_TYPE="Release"
INSTALL_PREFIX="./install"
SHARED="On"
DRY_RUN=0
FORCE_MODE=""            # "lc" | "nolc" | "" (auto)

# feature toggles (all off by default -> minimal CPU build)
ENABLE_MPI="Off"
ENABLE_CUDA="Off"
ENABLE_HIP="Off"
ENABLE_CALIPER="Off"
ENABLE_RMQ="Off"
ENABLE_PERFFLOWASPECT="Off"
ENABLE_WORKFLOW="Off"
INSTALL_FLUX_PYTHON="Off"
ENABLE_DEBUG="Off"
ENABLE_TESTS="Off"
CUDA_ARCH=""            # override; else AMS_CUDA_ARCH after setup-env

usage() {
  cat <<'EOF'
Usage: ams-configure.sh [feature flags] [options]

Feature flags (compose freely):
  --mpi                 enable MPI            (-DENABLE_MPI=On)
  --cuda                enable CUDA (NVIDIA)  (-DENABLE_CUDA=On)
  --hip                 enable HIP  (AMD)     (-DENABLE_HIP=On)     [mutually exclusive with --cuda]
  --caliper             Caliper profiling     (-DENABLE_CALIPER=On)
  --rmq                 RabbitMQ back end     (-DENABLE_RMQ=On)
  --perfflowaspect      PerfFlowAspect        (-DENABLE_PERFFLOWASPECT=On)
  --workflow            Python drivers        (-DENABLE_WORKFLOW=On)
  --install-flux-python install flux-python    (-DAMS_INSTALL_FLUX_PYTHON=On)
  --debug               verbose logging       (-DAMS_ENABLE_DEBUG=On)
  --tests               build tests           (-DENABLE_TESTS=On)

Options:
  --src PATH            AMS repo root (default: .)
  --build DIR           build directory (default: build)
  --build-type TYPE     Release|Debug|RelWithDebInfo (default: Release)
  --install-prefix PATH install prefix (default: ./install)
  --static              build static libs (default: shared)
  --cuda-arch ARCH      CUDA arch, e.g. 80,90 (default: $AMS_CUDA_ARCH on LC)
  --lc | --no-lc        force LC / non-LC dependency path (default: auto-detect)
  --dry-run             print the cmake command but do not run it
  -h, --help            this help

Examples:
  ams-configure.sh                          # minimal CPU build
  ams-configure.sh --rmq --mpi              # RabbitMQ + MPI
  ams-configure.sh --cuda --caliper         # GPU + profiling
  ams-configure.sh --rmq --dry-run          # show the command only
EOF
}

# ---- parse args ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mpi) ENABLE_MPI="On";;
    --cuda) ENABLE_CUDA="On";;
    --hip) ENABLE_HIP="On";;
    --caliper) ENABLE_CALIPER="On";;
    --rmq|--rabbitmq) ENABLE_RMQ="On";;
    --perfflowaspect|--pfa) ENABLE_PERFFLOWASPECT="On";;
    --workflow) ENABLE_WORKFLOW="On";;
    --install-flux-python) INSTALL_FLUX_PYTHON="On";;
    --debug) ENABLE_DEBUG="On";;
    --tests) ENABLE_TESTS="On";;
    --src) SRC="$2"; shift;;
    --build) BUILD="$2"; shift;;
    --build-type) BUILD_TYPE="$2"; shift;;
    --install-prefix) INSTALL_PREFIX="$2"; shift;;
    --static) SHARED="Off";;
    --cuda-arch) CUDA_ARCH="$2"; shift;;
    --lc) FORCE_MODE="lc";;
    --no-lc) FORCE_MODE="nolc";;
    --dry-run) DRY_RUN=1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
  shift
done

if [[ "$ENABLE_CUDA" == "On" && "$ENABLE_HIP" == "On" ]]; then
  echo "Error: --cuda and --hip are mutually exclusive." >&2
  exit 2
fi

if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
  echo "Error: '$SRC' does not look like the AMS repo root (no CMakeLists.txt)." >&2
  echo "       cd into the AMS clone or pass --src <path>." >&2
  exit 2
fi

# ---- decide dependency path ------------------------------------------------
LC_ENV_DIR="/usr/workspace/AMS/ams-spack-environments"
MODE="$FORCE_MODE"
if [[ -z "$MODE" ]]; then
  if [[ -d "$LC_ENV_DIR" ]]; then MODE="lc"; else MODE="nolc"; fi
fi

# extra -D hints accumulated from the environment
declare -a DEP_ARGS=()

if [[ "$MODE" == "lc" ]]; then
  echo ">> LC cluster detected — sourcing $SRC/scripts/gitlab/setup-env.sh"
  # shellcheck disable=SC1091
  source "$SRC/scripts/gitlab/setup-env.sh"

  [[ -n "${AMS_TORCH_PATH:-}" ]]   && DEP_ARGS+=("-DTorch_DIR=${AMS_TORCH_PATH}")
  [[ -n "${AMS_HDF5_PATH:-}" ]]    && DEP_ARGS+=("-DHDF5_DIR=${AMS_HDF5_PATH}")
  if [[ "$ENABLE_CALIPER" == "On" && -n "${AMS_CALIPER_PATH:-}" ]]; then
    DEP_ARGS+=("-Dcaliper_DIR=${AMS_CALIPER_PATH}")
  fi
  if [[ "$ENABLE_RMQ" == "On" && -n "${AMS_AMQPCPP_PATH:-}" ]]; then
    DEP_ARGS+=("-Damqpcpp_DIR=${AMS_AMQPCPP_PATH}")
  fi
  if [[ "$ENABLE_TESTS" == "On" && -n "${AMS_CATCH2_DIR:-}" ]]; then
    DEP_ARGS+=("-DAMS_CATCH2_DIR=${AMS_CATCH2_DIR}")
  fi
  if [[ "$ENABLE_CUDA" == "On" ]]; then
    ARCH="${CUDA_ARCH:-${AMS_CUDA_ARCH:-}}"
    [[ -n "$ARCH" ]] && DEP_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${ARCH}")
  fi
else
  echo ">> Non-LC cluster — using your *_DIR / CMAKE_PREFIX_PATH hints."
  echo "   (see references/manual-deps.md; a bare configure will fail if"
  echo "    Torch/HDF5/nlohmann_json can't be found)"
  # pass through anything the user already exported, if present
  [[ -n "${Torch_DIR:-}" ]]          && DEP_ARGS+=("-DTorch_DIR=${Torch_DIR}")
  [[ -n "${AMS_HDF5_DIR:-}" ]]       && DEP_ARGS+=("-DHDF5_DIR=${AMS_HDF5_DIR}")
  [[ -n "${nlohmann_json_DIR:-}" ]]  && DEP_ARGS+=("-Dnlohmann_json_DIR=${nlohmann_json_DIR}")
  [[ "$ENABLE_CALIPER" == "On" && -n "${caliper_DIR:-}" ]] && DEP_ARGS+=("-Dcaliper_DIR=${caliper_DIR}")
  [[ "$ENABLE_RMQ" == "On" && -n "${amqpcpp_DIR:-}" ]]      && DEP_ARGS+=("-Damqpcpp_DIR=${amqpcpp_DIR}")
  [[ "$ENABLE_TESTS" == "On" && -n "${AMS_CATCH2_DIR:-}" ]] && DEP_ARGS+=("-DAMS_CATCH2_DIR=${AMS_CATCH2_DIR}")
  if [[ "$ENABLE_CUDA" == "On" && -n "$CUDA_ARCH" ]]; then
    DEP_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
  fi
fi

# ---- assemble cmake command ------------------------------------------------
CMAKE_ARGS=(
  -S "$SRC" -B "$BUILD"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DBUILD_SHARED_LIBS="$SHARED"
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH="$SHARED"
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
  -DENABLE_MPI="$ENABLE_MPI"
  -DENABLE_CUDA="$ENABLE_CUDA"
  -DENABLE_HIP="$ENABLE_HIP"
  -DENABLE_CALIPER="$ENABLE_CALIPER"
  -DENABLE_RMQ="$ENABLE_RMQ"
  -DENABLE_PERFFLOWASPECT="$ENABLE_PERFFLOWASPECT"
  -DENABLE_WORKFLOW="$ENABLE_WORKFLOW"
  -DAMS_INSTALL_FLUX_PYTHON="$INSTALL_FLUX_PYTHON"
  -DAMS_ENABLE_DEBUG="$ENABLE_DEBUG"
  -DENABLE_TESTS="$ENABLE_TESTS"
  "${DEP_ARGS[@]}"
)

echo
echo ">> cmake command:"
printf '   cmake'
for a in "${CMAKE_ARGS[@]}"; do printf ' \\\n     %q' "$a"; done
printf '\n\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo ">> --dry-run set; not configuring."
  exit 0
fi

cmake "${CMAKE_ARGS[@]}"

echo
echo ">> Configured. Next:"
echo "   cmake --build $BUILD -j \"\$(nproc)\""
echo "   cmake --install $BUILD"
