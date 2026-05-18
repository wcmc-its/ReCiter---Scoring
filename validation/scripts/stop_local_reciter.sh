#!/usr/bin/env bash
#
# stop_local_reciter.sh - Stop all local ReCiter services.
#
# Reads PIDs from /tmp/reciter-local-pids.txt, kills processes,
# and removes the DynamoDB Local Docker container.
#
# Usage:
#   ./stop_local_reciter.sh [OPTIONS]
#
# Options:
#   --clean    Also remove scoring data files from /tmp/reciter-scoring-data
#   -h, --help Show this help message
#
set -euo pipefail

PID_FILE="/tmp/reciter-local-pids.txt"
DYNAMODB_CONTAINER="reciter-dynamodb-local"
CLEAN=false

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            head -15 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------
# Stop services from PID file
# ------------------------------------------------------------------
if [[ -f "$PID_FILE" ]]; then
    echo "Reading PIDs from $PID_FILE"
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

        if [[ "$line" == docker:* ]]; then
            container="${line#docker:}"
            echo -n "Stopping Docker container: $container ... "
            docker stop "$container" > /dev/null 2>&1 && echo "stopped" || echo "already stopped"
            docker rm -f "$container" > /dev/null 2>&1 || true
        elif [[ "$line" == pid:* ]]; then
            # Format: pid:<PID>:<name>
            IFS=':' read -r _ pid name <<< "$line"
            echo -n "Stopping $name (PID $pid) ... "
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
                # Wait up to 5 seconds for graceful shutdown
                for i in {1..5}; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
                echo "stopped"
            else
                echo "already stopped"
            fi
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "Removed PID file"
else
    echo "No PID file found at $PID_FILE"
    echo "Attempting to stop known services..."

    # Fallback: stop known container
    echo -n "Stopping Docker container: $DYNAMODB_CONTAINER ... "
    docker stop "$DYNAMODB_CONTAINER" > /dev/null 2>&1 && echo "stopped" || echo "not running"
    docker rm -f "$DYNAMODB_CONTAINER" > /dev/null 2>&1 || true
fi

# ------------------------------------------------------------------
# Safety net: kill orphaned ReCiter JVMs whose PID never made it into the
# PID file (start_local_reciter.sh captures the subshell PID, not the JVM,
# so the real JVM can be orphaned).
# ------------------------------------------------------------------
ORPHAN_PIDS=$(pgrep -f "java .*reciter-[0-9.]+\.jar.*--server.port=8081" 2>/dev/null || true)
if [[ -n "$ORPHAN_PIDS" ]]; then
    for pid in $ORPHAN_PIDS; do
        echo -n "Killing orphaned ReCiter JVM (PID $pid) ... "
        kill "$pid" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            echo "force-killed"
        else
            echo "stopped"
        fi
    done
fi

# Similar safety net for orphaned PubMed and scoring services
for pattern in "java .*reciter-pubmed-retrieval-tool.*\.jar" "python.* local_scoring_service.py"; do
    ORPHAN=$(pgrep -f "$pattern" 2>/dev/null || true)
    for pid in $ORPHAN; do
        echo -n "Killing orphaned ($pattern, PID $pid) ... "
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        echo "stopped"
    done
done

# ------------------------------------------------------------------
# Clean up scoring data if requested
# ------------------------------------------------------------------
if [[ "$CLEAN" == "true" ]]; then
    SCORING_DATA_DIR="/tmp/reciter-scoring-data"
    if [[ -d "$SCORING_DATA_DIR" ]]; then
        echo "Cleaning scoring data directory: $SCORING_DATA_DIR"
        rm -rf "$SCORING_DATA_DIR"
    fi
fi

# ------------------------------------------------------------------
# Clean up log files
# ------------------------------------------------------------------
echo ""
echo "Log files preserved at:"
for logfile in /tmp/reciter-pubmed.log /tmp/reciter-scoring.log /tmp/reciter-java.log; do
    if [[ -f "$logfile" ]]; then
        echo "  $logfile"
    fi
done

echo ""
echo "All local ReCiter services stopped."
