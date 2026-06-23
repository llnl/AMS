#!/usr/bin/env bash
# 
# Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Developer utility for AMS code-quality checks and explicit fixes.
#
# Examples:
#   scripts/run-code-quality.sh
#   scripts/run-code-quality.sh --fix
#   scripts/run-code-quality.sh --ruff --fix
#   scripts/run-code-quality.sh --clang-format --clang-tidy src tests
#   scripts/run-code-quality.sh --check --staged

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-code-quality.sh [options] [paths...]

Run code-quality tools across the repository or a selected subset of files.
By default, this runs check-only mode on tracked files under:
  src tests examples tools scripts .githooks

Options:
  --check              Run non-mutating checks only (default)
  --fix                Apply clang-format and ruff fixes explicitly
  --staged             Operate on staged files instead of the full codebase
  --fail-on-partial    Refuse staged runs when relevant files also have unstaged changes
  --ruff               Run ruff format/check on Python files
  --clang-format       Run clang-format on C/C++ files
  --clang-tidy         Run clang-tidy on C/C++ translation units
  --build-dir DIR      Build directory containing compile_commands.json
  --help               Show this help text

Notes:
  - If no tool is selected, the script runs ruff, clang-format, and clang-tidy.
  - --fix does not apply clang-tidy fixes; clang-tidy remains check-only.
  - AMS_SKIP_TIDY=1 skips clang-tidy.
  - AMS_BUILD_DIR can be used instead of --build-dir.
EOF
}

find_tool() {
  local base="$1"
  local candidate=""

  for candidate in "${base}-18" "${base}-17" "${base}-16" "${base}-15" "${base}-14" "${base}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

warn() {
  printf 'WARNING: %s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

append_if_exists() {
  local path="$1"

  if [[ -e "$path" ]]; then
    RELEVANT_FILES+=("$path")
  fi
}

is_cpp_file() {
  local path="$1"
  [[ "$path" =~ \.(cpp|hpp|cc|hh|h|c|cxx|hxx)$ ]]
}

is_tidy_file() {
  local path="$1"
  [[ "$path" =~ \.(cpp|cc|c|cxx)$ ]]
}

is_python_file() {
  local path="$1"
  [[ "$path" =~ \.py$ ]]
}

collect_files() {
  local target=""

  if [[ "$USE_STAGED" -eq 1 ]]; then
    while IFS= read -r -d '' target; do
      append_if_exists "$target"
    done < <(git diff --cached --name-only -z --diff-filter=ACM -- "${TARGETS[@]}")
    return
  fi

  while IFS= read -r -d '' target; do
    append_if_exists "$target"
  done < <(git ls-files -z -- "${TARGETS[@]}")
}

detect_partial_staging() {
  local file=""
  local partial_files=()

  [[ "$USE_STAGED" -eq 1 && "$FAIL_ON_PARTIAL" -eq 1 ]] || return 0

  for file in "${RELEVANT_FILES[@]}"; do
    if ! git diff --quiet -- "$file"; then
      partial_files+=("$file")
    fi
  done

  if [[ "${#partial_files[@]}" -eq 0 ]]; then
    return 0
  fi

  printf 'ERROR: refusing to run on partially staged files.\n' >&2
  printf 'Stage or stash the remaining edits first:\n' >&2
  for file in "${partial_files[@]}"; do
    printf '  %s\n' "$file" >&2
  done
  return 1
}

find_compile_db() {
  local candidate=""
  local dir=""

  if [[ -n "$BUILD_DIR" && -f "$BUILD_DIR/compile_commands.json" ]]; then
    printf '%s\n' "$BUILD_DIR/compile_commands.json"
    return 0
  fi

  for dir in build build-* cmake-build-* out; do
    while IFS= read -r candidate; do
      if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done < <(find "$REPO_ROOT" -maxdepth 2 -path "$REPO_ROOT/$dir/compile_commands.json" -print 2>/dev/null)
  done

  return 1
}

run_clang_format() {
  local file=""
  local failed=0

  if [[ "${#CPP_FILES[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ -z "$CLANG_FORMAT" ]]; then
    warn "clang-format not found, skipping C/C++ formatting."
    return 0
  fi

  if [[ "$APPLY_FIXES" -eq 1 ]]; then
    for file in "${CPP_FILES[@]}"; do
      "$CLANG_FORMAT" -i -style=file "$file"
      printf 'clang-format: updated %s\n' "$file"
    done
    return 0
  fi

  for file in "${CPP_FILES[@]}"; do
    if ! "$CLANG_FORMAT" --dry-run -Werror -style=file "$file"; then
      failed=1
    fi
  done

  if [[ "$failed" -eq 1 ]]; then
    printf '\nclang-format: formatting issues found.\n'
    printf 'Apply them explicitly with: scripts/run-code-quality.sh --clang-format --fix\n'
    return 1
  fi

  return 0
}

run_ruff() {
  local failed=0

  if [[ "${#PY_FILES[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ -z "$RUFF" ]]; then
    warn "ruff not found, skipping Python checks."
    warn "Install with: pip install ruff"
    return 0
  fi

  if [[ "$APPLY_FIXES" -eq 1 ]]; then
    "$RUFF" format "${PY_FILES[@]}"
    "$RUFF" check --fix "${PY_FILES[@]}" || true
  else
    if ! "$RUFF" format --check "${PY_FILES[@]}"; then
      failed=1
    fi
  fi

  if ! "$RUFF" check "${PY_FILES[@]}"; then
    failed=1
  fi

  if [[ "$failed" -eq 1 ]]; then
    printf '\nruff: Python formatting or lint issues found.\n'
    printf 'Apply fixes explicitly with: scripts/run-code-quality.sh --ruff --fix\n'
    return 1
  fi

  return 0
}

run_clang_tidy() {
  local compile_db=""
  local compile_db_dir=""
  local file=""
  local failed=0
  local rel_file=""
  local abs_file=""

  if [[ "$SKIP_TIDY" -eq 1 || "${#TIDY_FILES[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ -z "$CLANG_TIDY" ]]; then
    warn "clang-tidy not found, skipping C/C++ lint checks."
    return 0
  fi

  if ! compile_db=$(find_compile_db); then
    warn "compile_commands.json not found, skipping clang-tidy."
    warn "Run cmake first, or set AMS_BUILD_DIR / --build-dir."
    return 0
  fi

  compile_db_dir=$(dirname "$compile_db")

  for file in "${TIDY_FILES[@]}"; do
    rel_file="$file"
    abs_file="$REPO_ROOT/$file"

    if ! grep -Fq "\"$rel_file\"" "$compile_db" && ! grep -Fq "\"$abs_file\"" "$compile_db"; then
      continue
    fi

    printf 'clang-tidy: checking %s\n' "$file"
    if ! "$CLANG_TIDY" -p "$compile_db_dir" --quiet "$file"; then
      failed=1
    fi
  done

  if [[ "$failed" -eq 1 ]]; then
    printf '\nclang-tidy: errors found.\n'
    printf 'Fix the reported issues, or skip tidy with: AMS_SKIP_TIDY=1 ...\n'
    return 1
  fi

  return 0
}

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

DEFAULT_TARGETS=(src tests examples tools scripts .githooks)
TARGETS=()
RELEVANT_FILES=()
CPP_FILES=()
PY_FILES=()
TIDY_FILES=()

USE_STAGED=0
FAIL_ON_PARTIAL=0
APPLY_FIXES=0
RUN_RUFF=0
RUN_CLANG_FORMAT=0
RUN_CLANG_TIDY=0
BUILD_DIR="${AMS_BUILD_DIR:-}"
SKIP_TIDY=0

if [[ "${AMS_SKIP_TIDY:-0}" == "1" ]]; then
  SKIP_TIDY=1
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      APPLY_FIXES=0
      ;;
    --fix)
      APPLY_FIXES=1
      ;;
    --staged)
      USE_STAGED=1
      ;;
    --fail-on-partial)
      FAIL_ON_PARTIAL=1
      ;;
    --ruff)
      RUN_RUFF=1
      ;;
    --clang-format)
      RUN_CLANG_FORMAT=1
      ;;
    --clang-tidy)
      RUN_CLANG_TIDY=1
      ;;
    --build-dir)
      shift
      [[ "$#" -gt 0 ]] || die "--build-dir requires a value"
      BUILD_DIR="$1"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ "$#" -gt 0 ]]; do
        TARGETS+=("$1")
        shift
      done
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      TARGETS+=("$1")
      ;;
  esac
  shift
done

if [[ "$RUN_RUFF" -eq 0 && "$RUN_CLANG_FORMAT" -eq 0 && "$RUN_CLANG_TIDY" -eq 0 ]]; then
  RUN_RUFF=1
  RUN_CLANG_FORMAT=1
  RUN_CLANG_TIDY=1
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

CLANG_FORMAT=$(find_tool clang-format || true)
CLANG_TIDY=$(find_tool clang-tidy || true)
RUFF=$(command -v ruff 2>/dev/null || true)

collect_files

for file in "${RELEVANT_FILES[@]}"; do
  if [[ "$RUN_CLANG_FORMAT" -eq 1 ]] && is_cpp_file "$file"; then
    CPP_FILES+=("$file")
  fi

  if [[ "$RUN_RUFF" -eq 1 ]] && is_python_file "$file"; then
    PY_FILES+=("$file")
  fi

  if [[ "$RUN_CLANG_TIDY" -eq 1 ]] && is_tidy_file "$file"; then
    TIDY_FILES+=("$file")
  fi
done

if [[ "${#CPP_FILES[@]}" -eq 0 && "${#PY_FILES[@]}" -eq 0 && "${#TIDY_FILES[@]}" -eq 0 ]]; then
  exit 0
fi

detect_partial_staging

status=0

if [[ "$RUN_CLANG_FORMAT" -eq 1 ]] && ! run_clang_format; then
  status=1
fi

if [[ "$RUN_RUFF" -eq 1 ]] && ! run_ruff; then
  status=1
fi

if [[ "$RUN_CLANG_TIDY" -eq 1 ]] && ! run_clang_tidy; then
  status=1
fi

exit "$status"
