#!/usr/bin/env bash
#
# Run all model variant + device combinations and report results.
#
# Usage:
#   ./scripts/test-variants.sh                  # auto-detect devices, run all
#   ./scripts/test-variants.sh --device cpu      # CPU only
#   ./scripts/test-variants.sh --device cuda     # CUDA only
#   ./scripts/test-variants.sh --serve           # start HTTP server after tests
#   ./scripts/test-variants.sh --build           # build release binary first
#   ./scripts/test-variants.sh --hostname mymachine  # override hostname for URLs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

BIN="$REPO_ROOT/target/release/generate_audio"
MODELS_DIR="$REPO_ROOT/test_data/models"
REF_AUDIO="$REPO_ROOT/examples/data/apollo11_one_small_step.wav"
REF_TEXT="That's one small step for man, one giant leap for mankind."
TEXT="Hello world, this is a test."
OUTPUT_BASE="$PROJECT_ROOT/test_data/variant_tests"
SEED=42
DURATION=3.0

# Parse arguments
DEVICES=()
SERVE=false
BUILD=false
HOSTNAME="${HOSTNAME:-$(hostname)}"
HTTP_PORT=8765

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)   DEVICES+=("$2"); shift 2 ;;
        --serve)    SERVE=true; shift ;;
        --build)    BUILD=true; shift ;;
        --hostname) HOSTNAME="$2"; shift 2 ;;
        --port)     HTTP_PORT="$2"; shift 2 ;;
        --text)     TEXT="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--device cpu|cuda|metal] [--serve] [--build] [--hostname HOST]"
            echo ""
            echo "Options:"
            echo "  --device DEV    Test specific device(s). Repeat for multiple. Default: auto-detect."
            echo "  --serve         Start HTTP server after tests complete."
            echo "  --build         Build release binary before testing."
            echo "  --hostname H    Hostname for HTTP URLs (default: \$HOSTNAME)."
            echo "  --port P        HTTP server port (default: 8765)."
            echo "  --text TEXT     Text to synthesize (default: \"Hello world, this is a test.\")."
            echo "  --seed N        Random seed (default: 42)."
            echo "  --duration S    Duration in seconds (default: 3.0)."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Build ─────────────────────────────────────────────────────────────
if $BUILD; then
    echo "Building release binary..."
    # Check for CUDA toolkit
    if command -v nvcc &>/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        FEATURES="cuda,cli"
    else
        FEATURES="cli"
    fi
    cargo build --release --features "$FEATURES" --manifest-path "$REPO_ROOT/Cargo.toml"
    echo ""
fi

# ── Verify binary ────────────────────────────────────────────────────
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Binary not found: $BIN"
    echo "Run with --build or: cargo build --release --features cli --manifest-path $REPO_ROOT/Cargo.toml"
    exit 1
fi

# ── Auto-detect devices ──────────────────────────────────────────────
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    DEVICES=(cpu)
    # Check if binary was compiled with CUDA support
    if $BIN --device cuda --help &>/dev/null 2>&1; then
        # Check if CUDA device is actually available
        if $BIN --device cuda --text "x" --duration 0.1 --model-dir /nonexistent 2>&1 | grep -q "CUDA"; then
            DEVICES+=(cuda)
        fi
    fi
    echo "Auto-detected devices: ${DEVICES[*]}"
fi

# ── Discover models ──────────────────────────────────────────────────
declare -a MODEL_NAMES=()
declare -A MODEL_PATHS=()
declare -A MODEL_TYPES=()  # "base" or "customvoice"

for model_dir in "$MODELS_DIR"/*/; do
    [[ -d "$model_dir" ]] || continue
    name="$(basename "$model_dir")"

    # Skip if no model.safetensors
    [[ -f "$model_dir/model.safetensors" ]] || continue

    MODEL_NAMES+=("$name")
    MODEL_PATHS["$name"]="$model_dir"

    # Detect type from config.json
    if [[ -f "$model_dir/config.json" ]]; then
        if grep -qi '"model_type".*"base"' "$model_dir/config.json" 2>/dev/null || \
           grep -qi 'speaker_encoder' "$model_dir/config.json" 2>/dev/null; then
            MODEL_TYPES["$name"]="base"
        else
            MODEL_TYPES["$name"]="customvoice"
        fi
    else
        # Guess from directory name
        if [[ "$name" == *base* ]]; then
            MODEL_TYPES["$name"]="base"
        else
            MODEL_TYPES["$name"]="customvoice"
        fi
    fi
done

if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    echo "ERROR: No models found in $MODELS_DIR"
    echo "Run: ./scripts/download_test_data.sh"
    exit 1
fi

echo "Found ${#MODEL_NAMES[@]} models: ${MODEL_NAMES[*]}"
echo ""

# ── Define test cases ────────────────────────────────────────────────
# Each test: label + array of arguments (indexed by TEST_<N>_ARGS)
declare -a TEST_LABELS=()
test_count=0

add_test() {
    local label="$1"; shift
    TEST_LABELS+=("$label")
    # Store args as a declare-p'd array for safe retrieval
    local -a args=("$@")
    eval "TEST_${test_count}_ARGS=(\"\${args[@]}\")"
    test_count=$((test_count + 1))
}

for name in "${MODEL_NAMES[@]}"; do
    model_dir="${MODEL_PATHS[$name]}"
    model_type="${MODEL_TYPES[$name]}"

    if [[ "$model_type" == "base" ]]; then
        # Base models: x_vector_only and ICL
        if [[ -f "$REF_AUDIO" ]]; then
            add_test "${name}-xvector" \
                --model-dir "$model_dir" --ref-audio "$REF_AUDIO" --x-vector-only
            add_test "${name}-icl" \
                --model-dir "$model_dir" --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT"
        else
            echo "WARN: Skipping base model $name (no reference audio: $REF_AUDIO)"
        fi
    else
        # CustomVoice / VoiceDesign: preset speakers
        tokenizer_args=()
        if [[ -f "$model_dir/tokenizer.json" ]]; then
            tokenizer_args=(--tokenizer-dir "$model_dir")
        fi
        add_test "${name}-ryan" \
            --model-dir "$model_dir" "${tokenizer_args[@]}" --speaker ryan
        add_test "${name}-serena" \
            --model-dir "$model_dir" "${tokenizer_args[@]}" --speaker serena
    fi
done

echo "Test matrix: ${#TEST_LABELS[@]} tests x ${#DEVICES[@]} devices = $(( ${#TEST_LABELS[@]} * ${#DEVICES[@]} )) runs"
echo ""

# ── Run tests ────────────────────────────────────────────────────────
# Results arrays
declare -a RESULT_LABELS=()
declare -a RESULT_DEVICES=()
declare -a RESULT_TIMES=()
declare -a RESULT_STATUSES=()
declare -a RESULT_SIZES=()
declare -a RESULT_FILES=()

total_runs=$(( ${#TEST_LABELS[@]} * ${#DEVICES[@]} ))
run_num=0

for device in "${DEVICES[@]}"; do
    device_dir="$OUTPUT_BASE/$device"
    mkdir -p "$device_dir"

    for i in "${!TEST_LABELS[@]}"; do
        label="${TEST_LABELS[$i]}"
        # Retrieve the args array for this test
        eval "test_args=(\"\${TEST_${i}_ARGS[@]}\")"
        run_num=$((run_num + 1))
        outfile="$device_dir/${label}.wav"

        printf "[%d/%d] %-12s %-30s " "$run_num" "$total_runs" "$device" "$label"

        # Run and capture time
        start_time=$(date +%s%N)
        if "$BIN" --device "$device" "${test_args[@]}" \
            --text "$TEXT" --duration "$DURATION" --seed "$SEED" \
            --output "$outfile" >/dev/null 2>&1; then
            status="PASS"
        else
            status="FAIL"
        fi
        end_time=$(date +%s%N)
        elapsed_ms=$(( (end_time - start_time) / 1000000 ))
        elapsed_s=$(awk "BEGIN{printf \"%.1f\", $elapsed_ms/1000}")

        # File size
        if [[ -f "$outfile" ]]; then
            size=$(du -h "$outfile" | cut -f1)
        else
            size="-"
        fi

        RESULT_LABELS+=("$label")
        RESULT_DEVICES+=("$device")
        RESULT_TIMES+=("$elapsed_s")
        RESULT_STATUSES+=("$status")
        RESULT_SIZES+=("$size")
        RESULT_FILES+=("$outfile")

        if [[ "$status" == "PASS" ]]; then
            printf "%-6s %6ss  %s\n" "$status" "$elapsed_s" "$size"
        else
            printf "%-6s %6ss  (failed)\n" "$status" "$elapsed_s"
        fi
    done
done

# ── Summary table ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build comparison table if multiple devices
if [[ ${#DEVICES[@]} -gt 1 ]]; then
    # Header
    printf "%-30s" "Test"
    for d in "${DEVICES[@]}"; do
        printf "  %10s" "$d"
    done
    printf "  %10s\n" "Speedup"

    printf "%-30s" "$(printf '%.0s─' {1..30})"
    for d in "${DEVICES[@]}"; do
        printf "  %10s" "──────────"
    done
    printf "  %10s\n" "──────────"

    # Collect unique labels
    declare -A seen_labels=()
    declare -a unique_labels=()
    for label in "${RESULT_LABELS[@]}"; do
        if [[ -z "${seen_labels[$label]:-}" ]]; then
            seen_labels["$label"]=1
            unique_labels+=("$label")
        fi
    done

    for label in "${unique_labels[@]}"; do
        printf "%-30s" "$label"
        declare -A device_times=()
        for i in "${!RESULT_LABELS[@]}"; do
            if [[ "${RESULT_LABELS[$i]}" == "$label" ]]; then
                d="${RESULT_DEVICES[$i]}"
                t="${RESULT_TIMES[$i]}"
                s="${RESULT_STATUSES[$i]}"
                if [[ "$s" == "PASS" ]]; then
                    printf "  %9ss" "$t"
                    device_times["$d"]="$t"
                else
                    printf "  %10s" "FAIL"
                fi
            fi
        done

        # Calculate speedup (cpu / fastest-gpu)
        if [[ -n "${device_times[cpu]:-}" ]]; then
            cpu_t="${device_times[cpu]}"
            best_gpu=""
            for d in "${DEVICES[@]}"; do
                [[ "$d" == "cpu" ]] && continue
                if [[ -n "${device_times[$d]:-}" ]]; then
                    if [[ -z "$best_gpu" ]] || awk "BEGIN{exit !($best_gpu > ${device_times[$d]})}" 2>/dev/null; then
                        best_gpu="${device_times[$d]}"
                    fi
                fi
            done
            if [[ -n "$best_gpu" ]]; then
                speedup=$(awk "BEGIN{printf \"%.1fx\", $cpu_t/$best_gpu}")
                printf "  %10s" "$speedup"
            fi
        fi
        echo ""
    done
else
    printf "%-30s  %10s  %6s  %s\n" "Test" "Device" "Time" "Size"
    printf "%-30s  %10s  %6s  %s\n" "$(printf '%.0s─' {1..30})" "──────────" "──────" "────"
    for i in "${!RESULT_LABELS[@]}"; do
        printf "%-30s  %10s  %5ss  %s\n" \
            "${RESULT_LABELS[$i]}" "${RESULT_DEVICES[$i]}" \
            "${RESULT_TIMES[$i]}" "${RESULT_SIZES[$i]}"
    done
fi

# Pass/fail summary
pass_count=0
fail_count=0
for s in "${RESULT_STATUSES[@]}"; do
    if [[ "$s" == "PASS" ]]; then
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "Results: $pass_count passed, $fail_count failed out of $total_runs total"

# ── HTTP server ──────────────────────────────────────────────────────
if $SERVE; then
    echo ""
    echo "Starting HTTP server on port $HTTP_PORT..."

    # Kill existing server
    pkill -f "python3 -m http.server $HTTP_PORT" 2>/dev/null || true
    sleep 0.5

    cd "$OUTPUT_BASE"
    python3 -m http.server "$HTTP_PORT" &>/dev/null &
    server_pid=$!
    echo "Server PID: $server_pid"
    echo ""
    echo "Listen to results:"

    for i in "${!RESULT_FILES[@]}"; do
        if [[ "${RESULT_STATUSES[$i]}" == "PASS" ]]; then
            relpath="${RESULT_FILES[$i]#$OUTPUT_BASE/}"
            echo "  http://${HOSTNAME}:${HTTP_PORT}/${relpath}"
        fi
    done

    echo ""
    echo "Stop server: kill $server_pid"
fi

# Exit with failure if any test failed
[[ $fail_count -eq 0 ]]
