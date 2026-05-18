#!/usr/bin/env bash
#
# run_uc_validation_retry.sh - Retry pass for failed UIDs across all 6 UC
# institutions. Reads per-institution retry sample files from /tmp, runs
# scoring with extended timeout (1800s) and lower workers (2) for slow UIDs.
#
# Resume-safe: the underlying validation script skips UIDs already in
# scores.json. If a UID succeeds during retry, it's added to scores.json
# and won't be re-attempted.
#
# Prereqs: same as scripts/run_uc_validation_all.sh (patched ReCiter JAR
# with aws.s3.use=false, -Xmx12g start script, scoring service available).
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/external_validation/uc_system/data"
CONFIG_DIR="${REPO_ROOT}/external_validation/uc_system"
LOG_DIR="${REPO_ROOT}/external_validation/uc_system/logs"
MARKER_DIR="${REPO_ROOT}/external_validation/uc_system/.markers"

START_SCRIPT="${REPO_ROOT}/scripts/start_local_reciter.sh"
STOP_SCRIPT="${REPO_ROOT}/scripts/stop_local_reciter.sh"
VALIDATION_SCRIPT="${REPO_ROOT}/scripts/run_external_validation.py"

API_KEY="local"
API_URL="http://localhost:8081"
WORKERS=2

export RECITER_MAX_FAIL_PCT="0.50"
export RECITER_PER_UID_TIMEOUT="1800"

INSTITUTIONS=(
    "uci|UCI|University of California, Irvine|uci.edu,hs.uci.edu"
    "ucla|UCLA|University of California, Los Angeles|ucla.edu,mednet.ucla.edu"
    "usc|USC|University of Southern California|usc.edu,med.usc.edu"
    "ucsd|UCSD|University of California, San Diego|ucsd.edu,health.ucsd.edu"
    "ucdavis|UC Davis|University of California, Davis|ucdavis.edu,health.ucdavis.edu"
    "ucsf|UCSF|University of California, San Francisco|ucsf.edu"
)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/retry_orchestrator.log"
}

log "=========================================="
log "UC-System External Validation RETRY PASS"
log "=========================================="
log "Settings: workers=$WORKERS, timeout=${RECITER_PER_UID_TIMEOUT}s, fail_threshold=${RECITER_MAX_FAIL_PCT}"

for entry in "${INSTITUTIONS[@]}"; do
    IFS='|' read -r SHORT DISPLAY LABEL DOMAINS <<< "$entry"
    SAMPLE_FILE="/tmp/${SHORT}_retry_uids.json"
    DATA_CSV="$DATA_DIR/${SHORT}_data.csv"
    SPRING_CONFIG="$CONFIG_DIR/${SHORT}_spring_config.json"

    if [[ ! -f "$SAMPLE_FILE" ]]; then
        log "SKIP $SHORT — no retry sample file at $SAMPLE_FILE"
        continue
    fi
    COUNT=$(python3 -c "import json; print(len(json.load(open('$SAMPLE_FILE'))))")
    if [[ "$COUNT" -eq 0 ]]; then
        log "SKIP $SHORT — zero failed UIDs"
        continue
    fi

    log "==== START retry $SHORT ($DISPLAY) — $COUNT UIDs ===="
    T_START=$(date +%s)

    log "[$SHORT] Stopping existing stack..."
    bash "$STOP_SCRIPT" >> "$LOG_DIR/${SHORT}_retry_stack_stop.log" 2>&1 || true
    sleep 2

    log "[$SHORT] Starting stack..."
    if ! bash "$START_SCRIPT" --config-file "$SPRING_CONFIG" \
            > "$LOG_DIR/${SHORT}_retry_stack_start.log" 2>&1; then
        log "FAILED $SHORT — stack start failed"
        exit 1
    fi

    log "[$SHORT] Loading identities + gold standard (needed because container was wiped)..."
    if ! RECITER_API_KEY="$API_KEY" RECITER_API_URL="$API_URL" \
        python3 "$VALIDATION_SCRIPT" \
            --institution "$SHORT" \
            --data-file "$DATA_CSV" \
            --config "$SPRING_CONFIG" \
            --uid-prefix "${SHORT}_" \
            --email-domains "$DOMAINS" \
            --institution-label "$LABEL" \
            --base-url "$API_URL" \
            --non-interactive \
            --load-only > "$LOG_DIR/${SHORT}_retry_load.log" 2>&1; then
        log "FAILED $SHORT — load failed"
        exit 1
    fi

    # Remove these UIDs from the scores file so the resume-safe logic
    # re-tries them this pass (otherwise --score-only would skip them
    # because they're already in the file with score=0).
    # Actually they're NOT in scores.json — they were failures, written to
    # scoring_errors.json only. So no removal needed.

    log "[$SHORT] Scoring $COUNT retry UIDs (workers=$WORKERS, timeout=${RECITER_PER_UID_TIMEOUT}s)..."
    RECITER_API_KEY="$API_KEY" RECITER_API_URL="$API_URL" \
        python3 "$VALIDATION_SCRIPT" \
            --institution "$SHORT" \
            --data-file "$DATA_CSV" \
            --config "$SPRING_CONFIG" \
            --uid-prefix "${SHORT}_" \
            --base-url "$API_URL" \
            --non-interactive \
            --score-only \
            --workers "$WORKERS" \
            --sample "$SAMPLE_FILE" \
            > "$LOG_DIR/${SHORT}_retry.log" 2>&1
    RETRY_EXIT=$?

    T_END=$(date +%s)
    DUR=$((T_END - T_START))

    SCORED=$(grep -oE '[0-9]+ scored' "$LOG_DIR/${SHORT}_retry.log" | tail -1 || echo "? scored")
    FAILED=$(grep -oE '[0-9]+ failed' "$LOG_DIR/${SHORT}_retry.log" | tail -1 || echo "? failed")
    log "[$SHORT] Retry complete in ${DUR}s: $SCORED, $FAILED (exit $RETRY_EXIT)"
    log "==== END retry $SHORT ===="

    # Write retry marker
    {
        echo "institution=$SHORT"
        echo "retry_input_count=$COUNT"
        echo "retry_duration_seconds=$DUR"
        echo "$SCORED"
        echo "$FAILED"
        echo "retry_exit=$RETRY_EXIT"
        echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$MARKER_DIR/${SHORT}.retry.done"
done

log "All retries complete. Stopping stack..."
bash "$STOP_SCRIPT" >> "$LOG_DIR/retry_final_stack_stop.log" 2>&1 || true

log "=========================================="
log "Retry pass complete."
log "=========================================="
