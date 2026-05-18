#!/usr/bin/env bash
#
# run_uc_validation_all.sh - Sequentially run external validation for all 6
# UC-system institutions against the patched standalone-local ReCiter stack.
#
# For each institution:
#   1. Stop local stack (kills Docker container too -> wipes prior DynamoDB)
#   2. Start stack with institution-specific spring config
#   3. Load identities + gold standard via run_external_validation.py --load-only
#   4. Score all UIDs via run_external_validation.py --score-only (no --sample)
#   5. Mark institution complete via .planning marker
#
# Resume-safe: if interrupted, re-running skips institutions with marker files.
# Per-UID score resume is handled by run_external_validation.py's built-in
# skip-existing behavior.
#
# Prerequisites:
#   * Patched ReCiter JAR (aws.s3.use=false in application.properties)
#   * Per-institution data CSVs at external_validation/uc_system/data/<inst>_data.csv
#   * Per-institution spring configs at external_validation/uc_system/<inst>_spring_config.json
#
# Usage:
#   ./run_uc_validation_all.sh                  # run all institutions, in order
#   ./run_uc_validation_all.sh ucla usc        # run only the listed institutions
#   ./run_uc_validation_all.sh --force ucla    # run ucla even if .done marker exists
#
set -uo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/external_validation/uc_system/data"
CONFIG_DIR="${REPO_ROOT}/external_validation/uc_system"
LOG_DIR="${REPO_ROOT}/external_validation/uc_system/logs"
MARKER_DIR="${REPO_ROOT}/external_validation/uc_system/.markers"
RESULTS_DIR="${REPO_ROOT}/external_validation/results"
SUMMARY_FILE="${REPO_ROOT}/external_validation/uc_system/RUN_SUMMARY.md"
# Per-institution persistent DynamoDB Local data dirs, kept outside Dropbox to
# avoid sync churn. Retained after the run so a future re-score skips the
# multi-day PubMed retrieval.
DYNAMO_BASE="${HOME}/reciter-uc-dynamodb"

START_SCRIPT="${REPO_ROOT}/scripts/start_local_reciter.sh"
STOP_SCRIPT="${REPO_ROOT}/scripts/stop_local_reciter.sh"
VALIDATION_SCRIPT="${REPO_ROOT}/scripts/run_external_validation.py"

API_KEY="local"
API_URL="http://localhost:8081"
WORKERS=2

# filterByFeedback for the score phase. Default matches the original run;
# override with FILTER_BY_FEEDBACK=ALL to also capture PENDING (new-match)
# articles, which then write to <inst>_scores_all.json.
FILTER_BY_FEEDBACK="${FILTER_BY_FEEDBACK:-ACCEPTED_AND_REJECTED}"

# Allow higher failure rate and longer per-UID timeout. The script's defaults
# (5%, 300s) are tuned for the WCM ~874-UID scale. For UC institutions of
# 131-4021 persons with cold PubMed cache per institution, slow UIDs and
# transient 500s are expected. These env vars override scripts/run_external_validation.py
# defaults via os.environ lookups (added 2026-05-14).
export RECITER_MAX_FAIL_PCT="0.25"
export RECITER_PER_UID_TIMEOUT="600"

# Institutions in processing order (smallest -> largest).
# Format: short_name|display_name|institution_label|email_domains
INSTITUTIONS=(
    "uci|UCI|University of California, Irvine|uci.edu,hs.uci.edu"
    "ucla|UCLA|University of California, Los Angeles|ucla.edu,mednet.ucla.edu"
    "usc|USC|University of Southern California|usc.edu,med.usc.edu"
    "ucsd|UCSD|University of California, San Diego|ucsd.edu,health.ucsd.edu"
    "ucdavis|UC Davis|University of California, Davis|ucdavis.edu,health.ucdavis.edu"
    "ucsf|UCSF|University of California, San Francisco|ucsf.edu"
)

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
FORCE_FLAG=0
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE_FLAG=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) TARGETS+=("$1"); shift ;;
    esac
done

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$MARKER_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/orchestrator.log"
}

fail_institution() {
    local short="$1"
    local reason="$2"
    log "FAILED: $short — $reason"
    log "Pausing. Inspect logs at $LOG_DIR/${short}_*.log and re-run when ready."
    exit 1
}

run_institution() {
    local entry="$1"
    IFS='|' read -r SHORT DISPLAY LABEL DOMAINS <<< "$entry"

    local marker="$MARKER_DIR/${SHORT}.done"
    local data_csv="$DATA_DIR/${SHORT}_data.csv"
    local spring_config="$CONFIG_DIR/${SHORT}_spring_config.json"

    if [[ -f "$marker" && $FORCE_FLAG -eq 0 ]]; then
        log "SKIP $SHORT ($DISPLAY) — .done marker exists at $marker"
        return 0
    fi

    if [[ ! -f "$data_csv" ]]; then
        fail_institution "$SHORT" "data CSV missing at $data_csv (run scripts/build_uc_validation_data.py)"
    fi
    if [[ ! -f "$spring_config" ]]; then
        fail_institution "$SHORT" "spring config missing at $spring_config"
    fi

    log "==== START $SHORT ($DISPLAY) ===="
    local t_start=$(date +%s)

    # 1. Stop existing stack (wipes Docker container)
    log "[$SHORT] Stopping existing stack..."
    bash "$STOP_SCRIPT" >> "$LOG_DIR/${SHORT}_stack_stop.log" 2>&1 || true
    sleep 2

    # 2. Start fresh stack with this institution's spring config
    log "[$SHORT] Starting stack with $spring_config..."
    if ! bash "$START_SCRIPT" --config-file "$spring_config" \
            --dynamodb-data-dir "$DYNAMO_BASE/$SHORT" \
            > "$LOG_DIR/${SHORT}_stack_start.log" 2>&1; then
        fail_institution "$SHORT" "stack start failed; see $LOG_DIR/${SHORT}_stack_start.log"
    fi

    # 3. Verify API is reachable
    if ! curl -sf "$API_URL/reciter/ping" > /dev/null; then
        fail_institution "$SHORT" "API did not respond at $API_URL after stack start"
    fi

    # 4. Load identities + gold standard
    log "[$SHORT] Loading identities + gold standard..."
    local uid_prefix="${SHORT}_"
    if ! RECITER_API_KEY="$API_KEY" RECITER_API_URL="$API_URL" \
        python3 "$VALIDATION_SCRIPT" \
            --institution "$SHORT" \
            --data-file "$data_csv" \
            --config "$spring_config" \
            --uid-prefix "$uid_prefix" \
            --email-domains "$DOMAINS" \
            --institution-label "$LABEL" \
            --base-url "$API_URL" \
            --non-interactive \
            --load-only \
            > "$LOG_DIR/${SHORT}_load.log" 2>&1; then
        fail_institution "$SHORT" "load failed; see $LOG_DIR/${SHORT}_load.log"
    fi

    local load_count=$(grep -oE '[0-9]+ loaded' "$LOG_DIR/${SHORT}_load.log" | head -1)
    log "[$SHORT] Load complete: $load_count"

    # 5. Score all UIDs (resume-safe via skip-existing)
    log "[$SHORT] Scoring all UIDs (workers=$WORKERS)..."
    local score_start=$(date +%s)
    if ! RECITER_API_KEY="$API_KEY" RECITER_API_URL="$API_URL" \
        python3 "$VALIDATION_SCRIPT" \
            --institution "$SHORT" \
            --data-file "$data_csv" \
            --config "$spring_config" \
            --uid-prefix "$uid_prefix" \
            --base-url "$API_URL" \
            --non-interactive \
            --score-only \
            --workers "$WORKERS" \
            --filter-by-feedback "$FILTER_BY_FEEDBACK" \
            > "$LOG_DIR/${SHORT}_score.log" 2>&1; then
        fail_institution "$SHORT" "score failed; see $LOG_DIR/${SHORT}_score.log"
    fi

    local score_end=$(date +%s)
    local score_dur=$((score_end - score_start))
    local total_dur=$((score_end - t_start))

    # Extract summary stats
    local scored=$(grep -oE '[0-9]+ scored' "$LOG_DIR/${SHORT}_score.log" | tail -1)
    local failed=$(grep -oE '[0-9]+ failed' "$LOG_DIR/${SHORT}_score.log" | tail -1)
    log "[$SHORT] Score complete in ${score_dur}s: $scored, $failed"
    log "[$SHORT] Total duration ${total_dur}s"

    # Mark complete
    {
        echo "institution=$SHORT"
        echo "display=$DISPLAY"
        echo "duration_seconds=$total_dur"
        echo "score_duration_seconds=$score_dur"
        echo "$scored"
        echo "$failed"
        echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$marker"
    log "==== END $SHORT (marker written) ===="
}

write_summary() {
    {
        echo "# UC-System External Validation Run Summary"
        echo
        echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        echo "| Inst | Persons | Duration | Scored | Failed |"
        echo "|---|---|---|---|---|"
        for entry in "${INSTITUTIONS[@]}"; do
            IFS='|' read -r SHORT DISPLAY _ _ <<< "$entry"
            local marker="$MARKER_DIR/${SHORT}.done"
            if [[ -f "$marker" ]]; then
                local dur=$(grep '^duration_seconds=' "$marker" | cut -d= -f2)
                local scored=$(grep 'scored' "$marker" || echo '?')
                local failed=$(grep 'failed' "$marker" || echo '?')
                # Person count from CSV (subtract header)
                local persons=$(( $(wc -l < "$DATA_DIR/${SHORT}_data.csv") - 1 ))
                echo "| $DISPLAY | $persons rows | ${dur}s | $scored | $failed |"
            else
                echo "| $DISPLAY | (not run) | - | - | - |"
            fi
        done
    } > "$SUMMARY_FILE"
    log "Summary written to $SUMMARY_FILE"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
log "=========================================="
log "UC-System External Validation Orchestrator"
log "=========================================="

# Select institutions to run
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    SELECTED=("${INSTITUTIONS[@]}")
else
    SELECTED=()
    for target in "${TARGETS[@]}"; do
        local_found=0
        for entry in "${INSTITUTIONS[@]}"; do
            short="${entry%%|*}"
            if [[ "$short" == "$target" ]]; then
                SELECTED+=("$entry")
                local_found=1
                break
            fi
        done
        if [[ $local_found -eq 0 ]]; then
            log "ERROR: unknown institution '$target' (expected one of: $(echo "${INSTITUTIONS[@]}" | tr ' ' '\n' | cut -d'|' -f1 | tr '\n' ' '))"
            exit 1
        fi
    done
fi

log "Plan: run ${#SELECTED[@]} institution(s)"
for entry in "${SELECTED[@]}"; do
    log "  - $(echo "$entry" | cut -d'|' -f2)"
done

# Run each
for entry in "${SELECTED[@]}"; do
    run_institution "$entry"
done

# Final summary
write_summary

# Clean shutdown
log "All institutions complete. Stopping stack..."
bash "$STOP_SCRIPT" >> "$LOG_DIR/final_stack_stop.log" 2>&1 || true

log "=========================================="
log "Orchestrator complete."
log "Summary: $SUMMARY_FILE"
log "=========================================="
