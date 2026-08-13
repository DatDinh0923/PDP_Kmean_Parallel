#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_IMAGE="$SCRIPT_DIR/data/test_1024.png"
DEFAULT_K=16
DEFAULT_SEED=42
DEFAULT_MAX_ITER=30
CUDA_ARCH=${CUDA_ARCH:-sm_75}

usage() {
    cat <<'EOF'
Usage:
  ./run.sh seq [image] [K] [seed] [max_iter] [output_prefix]
  ./run.sh cuda <v1|v2|v3> [image] [K] [seed] [max_iter] [output_prefix]

The selected backend is built automatically when its binary is missing or its
source files are newer. Override the default CUDA architecture with, for
example, CUDA_ARCH=sm_86 ./run.sh cuda v3 image.jpg.
EOF
}

select_backend() {
    echo "Select a backend:" >&2
    echo "  1) Sequential (-O3)" >&2
    echo "  2) CUDA v1" >&2
    echo "  3) CUDA v2" >&2
    echo "  4) CUDA v3" >&2
    read -r -p "Choice [1-4]: " choice

    case "$choice" in
        1) printf '%s\n' "seq" ;;
        2) printf '%s\n' "v1" ;;
        3) printf '%s\n' "v2" ;;
        4) printf '%s\n' "v3" ;;
        *) echo "Invalid choice: $choice" >&2; exit 2 ;;
    esac
}

resolve_image() {
    local requested=${1:-}

    if [[ -z "$requested" ]]; then
        printf '%s\n' "$DEFAULT_IMAGE"
    elif [[ "$requested" = /* || -f "$requested" ]]; then
        printf '%s\n' "$requested"
    else
        printf '%s\n' "$SCRIPT_DIR/$requested"
    fi
}

require_file() {
    local path=$1
    local description=$2

    if [[ ! -f "$path" ]]; then
        echo "Error: $description not found: $path" >&2
        exit 1
    fi
}

require_command() {
    local command_name=$1

    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required compiler not found: $command_name" >&2
        exit 1
    fi
}

needs_rebuild() {
    local binary=$1
    shift

    if [[ ! -x "$binary" ]]; then
        return 0
    fi

    local source
    for source in "$@"; do
        if [[ "$source" -nt "$binary" ]]; then
            return 0
        fi
    done

    return 1
}

build_seq_if_needed() {
    local binary="$SCRIPT_DIR/bin/seq_O3"
    local sources=(
        "$SCRIPT_DIR/seq/kmeans_seq.cpp"
        "$SCRIPT_DIR/common/kmeans_common.h"
        "$SCRIPT_DIR/common/stb_image.h"
        "$SCRIPT_DIR/common/stb_image_write.h"
    )

    if ! needs_rebuild "$binary" "${sources[@]}"; then
        return
    fi

    require_command g++
    echo "Building sequential backend..."
    g++ -O3 -ffp-contract=off -std=c++17 '-DKM_TAG="O3"' \
        -I"$SCRIPT_DIR/common" "$SCRIPT_DIR/seq/kmeans_seq.cpp" \
        -o "$binary" -lm
}

build_cuda_if_needed() {
    local binary="$SCRIPT_DIR/bin/kmeans_cuda"
    local sources=(
        "$SCRIPT_DIR/cuda/kmeans_cuda.cu"
        "$SCRIPT_DIR/common/kmeans_common.h"
        "$SCRIPT_DIR/common/stb_image.h"
        "$SCRIPT_DIR/common/stb_image_write.h"
    )

    if ! needs_rebuild "$binary" "${sources[@]}"; then
        return
    fi

    require_command nvcc
    echo "Building CUDA backend for $CUDA_ARCH..."
    nvcc -O3 -std=c++17 -arch="$CUDA_ARCH" --fmad=false \
        -Xcompiler "-O3 -ffp-contract=off" -I"$SCRIPT_DIR/common" \
        "$SCRIPT_DIR/cuda/kmeans_cuda.cu" -o "$binary"
}

run_seq() {
    local image=$1
    local k=$2
    local seed=$3
    local max_iter=$4
    local output_prefix=$5
    local binary="$SCRIPT_DIR/bin/seq_O3"

    build_seq_if_needed
    echo "Running sequential backend with image: $image"
    "$binary" "$image" "$k" "$seed" "$max_iter" "$output_prefix"
}

run_cuda() {
    local version=$1
    local image=$2
    local k=$3
    local seed=$4
    local max_iter=$5
    local output_prefix=$6
    local binary="$SCRIPT_DIR/bin/kmeans_cuda"

    build_cuda_if_needed
    echo "Running CUDA v$version with image: $image"
    "$binary" "$image" "$k" "$seed" "$max_iter" "$output_prefix" \
        --version "$version"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -eq 0 ]]; then
    backend=$(select_backend)
else
    backend=$1
    shift
fi

if [[ "$backend" == "cuda" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "Error: CUDA requires a version: v1, v2, or v3." >&2
        usage >&2
        exit 2
    fi
    backend=$1
    shift
fi

case "$backend" in
    seq|v1|v2|v3|all) ;;
    *)
        echo "Error: unknown backend '$backend'." >&2
        usage >&2
        exit 2
        ;;
esac

image=$(resolve_image "${1:-}")
[[ $# -gt 0 ]] && shift
k=${1:-$DEFAULT_K}
[[ $# -gt 0 ]] && shift
seed=${1:-$DEFAULT_SEED}
[[ $# -gt 0 ]] && shift
max_iter=${1:-$DEFAULT_MAX_ITER}
[[ $# -gt 0 ]] && shift

require_file "$image" "input image"
mkdir -p "$SCRIPT_DIR/bin"

case "$backend" in
    seq)
        output_prefix=${1:-$SCRIPT_DIR/bin/run_seq}
        run_seq "$image" "$k" "$seed" "$max_iter" "$output_prefix"
        ;;
    v1|v2|v3)
        version=${backend#v}
        output_prefix=${1:-$SCRIPT_DIR/bin/run_v$version}
        run_cuda "$version" "$image" "$k" "$seed" "$max_iter" "$output_prefix"
        ;;
    all)
        if [[ $# -gt 0 ]]; then
            echo "Error: the 'all' option does not accept an output prefix." >&2
            exit 2
        fi
        run_seq "$image" "$k" "$seed" "$max_iter" "$SCRIPT_DIR/bin/run_seq"
        for version in 1 2 3; do
            run_cuda "$version" "$image" "$k" "$seed" "$max_iter" \
                "$SCRIPT_DIR/bin/run_v$version"
        done
        ;;
esac
