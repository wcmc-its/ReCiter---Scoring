#!/usr/bin/env bash
#
# start_local_reciter.sh - Start all local ReCiter services in dependency order.
#
# Starts:
#   1. DynamoDB Local (Docker, port 8000)
#   2. PubMed Retrieval Tool (JAR, port 8083)
#   3. Local Scoring Service (Python/Flask, port 9000)
#   4. ReCiter Java (JAR, port 8081)
#
# Automatically builds missing JARs and checks all prerequisites.
#
# Usage:
#   ./start_local_reciter.sh [OPTIONS]
#
# Options:
#   --config-file PATH       Path to JSON file for SPRING_APPLICATION_JSON
#   --scoring-data-dir PATH  Directory for scoring input files (default: auto-detected)
#   --dynamodb-data-dir PATH Persist DynamoDB Local to this host dir (default: -inMemory)
#   --skip-pubmed            Skip starting PubMed Retrieval Tool
#   --skip-dynamodb          Skip starting DynamoDB Local (assume already running)
#   -h, --help               Show this help message
#
set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
# ReCiter and PubMed-tool are external repos; override via env vars if not at these defaults.
RECITER_DIR="${RECITER_DIR:-${HOME}/Dropbox/GitHub/ReCiter}"
PUBMED_DIR="${PUBMED_DIR:-${HOME}/Dropbox/GitHub/ReCiter-PubMed-Retrieval-Tool}"
# This harness lives inside the ReCiter---Scoring repo; derive its root from the script path.
SCORING_REPO_DIR="${SCORING_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PID_FILE="/tmp/reciter-local-pids.txt"
DYNAMODB_CONTAINER="reciter-dynamodb-local"
LOG_DIR="/tmp"

CONFIG_FILE=""
SCORING_DATA_DIR=""
DYNAMODB_DATA_DIR=""
SKIP_PUBMED=false
SKIP_DYNAMODB=false

# Java 17 required — Lombok doesn't work with newer JDKs
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
export PATH="${JAVA_HOME}/bin:${PATH}"

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --scoring-data-dir)
            SCORING_DATA_DIR="$2"
            shift 2
            ;;
        --dynamodb-data-dir)
            DYNAMODB_DATA_DIR="$2"
            shift 2
            ;;
        --skip-pubmed)
            SKIP_PUBMED=true
            shift
            ;;
        --skip-dynamodb)
            SKIP_DYNAMODB=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
log_status() {
    local service="$1"
    local status="$2"
    case "$status" in
        OK)      echo "[  OK  ] $service" ;;
        STARTING) echo -n "[START ] $service ... " ;;
        FAILED)  echo "[FAILED] $service" ;;
        BUILD)   echo "[BUILD ] $service" ;;
        SKIP)    echo "[ SKIP ] $service" ;;
        CHECK)   echo -n "[CHECK ] $service ... " ;;
    esac
}

wait_for_health() {
    local url="$1"
    local timeout="$2"
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

cleanup_on_error() {
    echo ""
    echo "[ERROR] Startup failed. Cleaning up..."
    if [[ -f "$PID_FILE" ]]; then
        bash "$(dirname "$0")/stop_local_reciter.sh" 2>/dev/null || true
    fi
    echo ""
    echo "Log files for debugging:"
    for f in "$LOG_DIR"/reciter-pubmed.log "$LOG_DIR"/reciter-scoring.log "$LOG_DIR"/reciter-java.log; do
        if [[ -f "$f" && -s "$f" ]]; then
            echo "  $f (last 5 lines):"
            tail -5 "$f" | sed 's/^/    /'
        fi
    done
    exit 1
}

trap cleanup_on_error ERR

# ------------------------------------------------------------------
# Prerequisites check
# ------------------------------------------------------------------
echo ""
echo "========================================="
echo "  ReCiter Standalone Local Stack"
echo "========================================="
echo ""

PREREQ_FAIL=false

# Check Java 17
log_status "Java 17" "CHECK"
if ! java -version 2>&1 | grep -q '"17\.'; then
    echo "MISSING"
    echo "        Install: brew install openjdk@17"
    echo "        Then: export JAVA_HOME=/opt/homebrew/opt/openjdk@17"
    PREREQ_FAIL=true
else
    echo "OK ($(java -version 2>&1 | head -1 | cut -d'"' -f2))"
fi

# Check Docker
if [[ "$SKIP_DYNAMODB" == "false" ]]; then
    log_status "Docker" "CHECK"
    if ! docker info > /dev/null 2>&1; then
        echo "NOT RUNNING"
        echo "        Start Docker Desktop first, or use --skip-dynamodb"
        PREREQ_FAIL=true
    else
        echo "OK"
    fi
fi

# Check Python + Flask
log_status "Python + Flask" "CHECK"
if ! python3 -c "import flask" 2>/dev/null; then
    echo "MISSING"
    echo "        Install: pip3 install flask"
    PREREQ_FAIL=true
else
    echo "OK"
fi

# Check scoring models
log_status "Scoring models" "CHECK"
if [[ ! -f "${SCORING_REPO_DIR}/app/models/feedbackIdentityModel.joblib" ]]; then
    echo "MISSING"
    echo "        Expected at: ${SCORING_REPO_DIR}/app/models/"
    PREREQ_FAIL=true
else
    echo "OK"
fi

# Check config file if specified
if [[ -n "$CONFIG_FILE" ]]; then
    log_status "Config file" "CHECK"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "NOT FOUND: $CONFIG_FILE"
        PREREQ_FAIL=true
    else
        echo "OK ($CONFIG_FILE)"
    fi
fi

# Check for port conflicts
for port_info in "8000:DynamoDB Local" "8081:ReCiter Java" "8083:PubMed Tool" "9000:Scoring Service"; do
    port="${port_info%%:*}"
    svc="${port_info#*:}"
    if lsof -i ":$port" -sTCP:LISTEN > /dev/null 2>&1; then
        existing=$(lsof -i ":$port" -sTCP:LISTEN -t 2>/dev/null | head -1)
        existing_name=$(ps -p "$existing" -o comm= 2>/dev/null || echo "unknown")
        # Allow our own DynamoDB container
        if [[ "$port" == "8000" && "$existing_name" == *"docker"* ]]; then
            continue
        fi
        echo "[WARN ] Port $port ($svc) already in use by PID $existing ($existing_name)"
    fi
done

if [[ "$PREREQ_FAIL" == "true" ]]; then
    echo ""
    echo "[ABORT] Fix the issues above and retry."
    exit 1
fi

# ------------------------------------------------------------------
# Auto-build missing JARs
# ------------------------------------------------------------------

# ReCiter JAR
RECITER_JAR=$(ls "${RECITER_DIR}/target/reciter-"*.jar 2>/dev/null | head -1)
if [[ -z "$RECITER_JAR" ]]; then
    log_status "Building ReCiter JAR (first time, ~2 min)" "BUILD"
    if (cd "$RECITER_DIR" && JAVA_HOME=/opt/homebrew/opt/openjdk@17 mvn clean package -DskipTests -q 2>&1); then
        RECITER_JAR=$(ls "${RECITER_DIR}/target/reciter-"*.jar 2>/dev/null | head -1)
        echo "[  OK  ] ReCiter JAR built: $(basename "$RECITER_JAR")"
    else
        echo "[FAILED] ReCiter JAR build failed."
        echo "        Try manually: cd ${RECITER_DIR} && JAVA_HOME=/opt/homebrew/opt/openjdk@17 mvn clean package -DskipTests"
        exit 1
    fi
fi

# PubMed Retrieval Tool JAR
if [[ "$SKIP_PUBMED" == "false" ]]; then
    PUBMED_JAR=$(ls "${PUBMED_DIR}/target/reciter-pubmed-retrieval-tool-"*.jar 2>/dev/null | head -1)
    if [[ -z "$PUBMED_JAR" ]]; then
        log_status "Building PubMed Tool JAR (first time, ~10s)" "BUILD"
        if (cd "$PUBMED_DIR" && JAVA_HOME=/opt/homebrew/opt/openjdk@17 mvn clean package -DskipTests -q 2>&1); then
            PUBMED_JAR=$(ls "${PUBMED_DIR}/target/reciter-pubmed-retrieval-tool-"*.jar 2>/dev/null | head -1)
            echo "[  OK  ] PubMed JAR built: $(basename "$PUBMED_JAR")"
        else
            echo "[FAILED] PubMed JAR build failed."
            echo "        Try manually: cd ${PUBMED_DIR} && JAVA_HOME=/opt/homebrew/opt/openjdk@17 mvn clean package -DskipTests"
            exit 1
        fi
    fi
fi

# ------------------------------------------------------------------
# Detect SCORING_DATA_DIR
# ------------------------------------------------------------------
if [[ -z "$SCORING_DATA_DIR" ]]; then
    SCORING_DATA_DIR="${RECITER_DIR}/src/main/resources/scripts"
fi
mkdir -p "$SCORING_DATA_DIR"

# ------------------------------------------------------------------
# Spring config
# ------------------------------------------------------------------
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    CONFIG_JSON=$(cat "$CONFIG_FILE")
else
    CONFIG_JSON='{
        "aws.dynamoDb.local": "false",
        "aws.s3.use": "false",
        "aws.s3.use.cached.identityAll": "false",
        "aws.dynamodb.settings.table.create": "true",
        "aws.dynamodb.settings.region": "us-east-1",
        "spring.security.enabled": "false",
        "use.scopus.articles": "false",
        "strategy.scopus.common.affiliation": "false"
    }'
fi

# ------------------------------------------------------------------
# Initialize PID file
# ------------------------------------------------------------------
> "$PID_FILE"
echo "# ReCiter Local Services PIDs - $(date)" >> "$PID_FILE"

echo ""
echo "--- Starting services ---"
echo ""

# ------------------------------------------------------------------
# 1. DynamoDB Local (Docker, port 8000)
# ------------------------------------------------------------------
if [[ "$SKIP_DYNAMODB" == "false" ]]; then
    log_status "DynamoDB Local (Docker, port 8000)" "STARTING"

    docker rm -f "$DYNAMODB_CONTAINER" > /dev/null 2>&1 || true

    if [[ -n "$DYNAMODB_DATA_DIR" ]]; then
        # Persistent mode: bind-mount a host dir and use -dbPath so the
        # PubMedArticle cache + Analysis survive container restarts.
        mkdir -p "$DYNAMODB_DATA_DIR"
        chmod 777 "$DYNAMODB_DATA_DIR"
        echo -n "(persistent: $DYNAMODB_DATA_DIR) "
        docker run -d --name "$DYNAMODB_CONTAINER" \
            -p 8000:8000 \
            -v "$DYNAMODB_DATA_DIR":/home/dynamodblocal/data \
            amazon/dynamodb-local \
            -jar DynamoDBLocal.jar -sharedDb -dbPath /home/dynamodblocal/data > /dev/null 2>&1
    else
        docker run -d --name "$DYNAMODB_CONTAINER" \
            -p 8000:8000 \
            amazon/dynamodb-local \
            -jar DynamoDBLocal.jar -sharedDb -inMemory > /dev/null 2>&1
    fi

    echo "docker:${DYNAMODB_CONTAINER}" >> "$PID_FILE"

    # DynamoDB Local returns 400 to plain HTTP (missing auth token),
    # so use aws CLI for health check instead of curl
    DYNAMO_ELAPSED=0
    DYNAMO_OK=false
    while [[ $DYNAMO_ELAPSED -lt 15 ]]; do
        if aws dynamodb list-tables --endpoint-url http://localhost:8000 --region us-east-1 --no-cli-pager > /dev/null 2>&1; then
            DYNAMO_OK=true
            break
        fi
        sleep 1
        DYNAMO_ELAPSED=$((DYNAMO_ELAPSED + 1))
    done
    if [[ "$DYNAMO_OK" == "true" ]]; then
        log_status "DynamoDB Local (Docker, port 8000)" "OK"
    else
        log_status "DynamoDB Local (Docker, port 8000)" "FAILED"
        echo "        Docker logs:"
        docker logs "$DYNAMODB_CONTAINER" 2>&1 | tail -5 | sed 's/^/        /'
        exit 1
    fi
else
    log_status "DynamoDB Local (--skip-dynamodb)" "SKIP"
fi

# ------------------------------------------------------------------
# 2. PubMed Retrieval Tool (JAR, port 8083)
# ------------------------------------------------------------------
if [[ "$SKIP_PUBMED" == "false" ]]; then
    log_status "PubMed Retrieval Tool (JAR, port 8083)" "STARTING"

    # Pass PUBMED_API_KEY if available (increases NCBI rate limit from 3 to 10 req/sec)
    PUBMED_API_KEY="${PUBMED_API_KEY:-}" \
    java -jar "$PUBMED_JAR" --server.port=8083 > "$LOG_DIR/reciter-pubmed.log" 2>&1 &
    PUBMED_PID=$!
    echo "pid:${PUBMED_PID}:pubmed" >> "$PID_FILE"

    if wait_for_health "http://localhost:8083/pubmed/ping" 60 "PubMed Retrieval Tool"; then
        log_status "PubMed Retrieval Tool (JAR, port 8083)" "OK"
    else
        # Fallback: just check if port is listening
        if wait_for_health "http://localhost:8083" 10 "PubMed port check"; then
            log_status "PubMed Retrieval Tool (JAR, port 8083)" "OK"
        else
            log_status "PubMed Retrieval Tool (JAR, port 8083)" "FAILED"
            echo "        Last 5 lines of $LOG_DIR/reciter-pubmed.log:"
            tail -5 "$LOG_DIR/reciter-pubmed.log" 2>/dev/null | sed 's/^/        /'
            exit 1
        fi
    fi
else
    log_status "PubMed Retrieval Tool (--skip-pubmed)" "SKIP"
fi

# ------------------------------------------------------------------
# 3. Local Scoring Service (Python/Flask, port 9000)
# ------------------------------------------------------------------
log_status "Local Scoring Service (Python, port 9000)" "STARTING"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SCORING_APP_DIR="${SCORING_REPO_DIR}/app" \
SCORING_DATA_DIR="$SCORING_DATA_DIR" \
python3 "${SCRIPT_DIR}/local_scoring_service.py" > "$LOG_DIR/reciter-scoring.log" 2>&1 &
SCORING_PID=$!
echo "pid:${SCORING_PID}:scoring" >> "$PID_FILE"

if wait_for_health "http://localhost:9000/health" 30 "Local Scoring Service"; then
    log_status "Local Scoring Service (Python, port 9000)" "OK"
else
    log_status "Local Scoring Service (Python, port 9000)" "FAILED"
    echo "        Last 5 lines of $LOG_DIR/reciter-scoring.log:"
    tail -5 "$LOG_DIR/reciter-scoring.log" 2>/dev/null | sed 's/^/        /'
    exit 1
fi

# ------------------------------------------------------------------
# 4. ReCiter Java (JAR, port 8081)
# ------------------------------------------------------------------
log_status "ReCiter Java (JAR, port 8081)" "STARTING"

# Merge standalone-required properties into CONFIG_JSON
# Write configs to temp files to avoid shell quoting issues
STANDALONE_TMP=$(mktemp)
CONFIG_TMP=$(mktemp)
MERGED_TMP=$(mktemp)

cat > "$STANDALONE_TMP" <<'STANDALONEEOF'
{
    "aws.s3.use": "false",
    "aws.dynamoDb.local": "false",
    "aws.s3.use.cached.identityAll": "false",
    "aws.dynamodb.settings.table.create": "true",
    "aws.dynamodb.settings.region": "us-east-1",
    "spring.security.enabled": "false",
    "use.scopus.articles": "false",
    "strategy.scopus.common.affiliation": "false"
}
STANDALONEEOF

echo "$CONFIG_JSON" > "$CONFIG_TMP"

python3 -c "
import json
base = json.load(open('$STANDALONE_TMP'))
user = json.load(open('$CONFIG_TMP'))
base.update(user)
json.dump(base, open('$MERGED_TMP', 'w'))
"
MERGED_JSON=$(cat "$MERGED_TMP")
rm -f "$STANDALONE_TMP" "$CONFIG_TMP" "$MERGED_TMP"

# Java writes scoring input to src/main/resources/scripts/ relative to CWD,
# so we must launch from the ReCiter repo directory
(cd "$RECITER_DIR" && \
AMAZON_DYNAMODB_ENDPOINT=http://localhost:8000 \
AWS_REGION=us-east-1 \
ADMIN_API_KEY=local \
CONSUMER_API_KEY=local \
PUBMED_SERVICE=http://localhost:8083 \
SCOPUS_SERVICE=http://localhost:8082 \
RECITERSCORING_SERVICE_URL=http://localhost:9000 \
SPRING_APPLICATION_JSON="$MERGED_JSON" \
java -Xmx4g \
    -Daws.s3.use=false \
    -Daws.dynamoDb.local=false \
    -Dspring.security.enabled=false \
    -jar "$RECITER_JAR" \
    --server.port=8081 > "$LOG_DIR/reciter-java.log" 2>&1) &
RECITER_PID=$!
echo "pid:${RECITER_PID}:reciter" >> "$PID_FILE"

if wait_for_health "http://localhost:8081/reciter/ping" 90 "ReCiter Java"; then
    log_status "ReCiter Java (JAR, port 8081)" "OK"
else
    log_status "ReCiter Java (JAR, port 8081)" "FAILED"
    echo "        Last 5 lines of $LOG_DIR/reciter-java.log:"
    tail -5 "$LOG_DIR/reciter-java.log" 2>/dev/null | sed 's/^/        /'
    exit 1
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "========================================="
echo "  All local ReCiter services are running"
echo "========================================="
echo ""
echo "  DynamoDB Local:     http://localhost:8000"
if [[ "$SKIP_PUBMED" == "false" ]]; then
    echo "  PubMed Tool:        http://localhost:8083"
fi
echo "  Scoring Service:    http://localhost:9000/health"
echo "  ReCiter API:        http://localhost:8081/reciter/ping"
echo ""
echo "  Scoring data dir:   $SCORING_DATA_DIR"
echo "  PID file:           $PID_FILE"
echo "  Logs:               $LOG_DIR/reciter-{pubmed,scoring,java}.log"
echo ""
echo "  Stop all services:  $(dirname "$0")/stop_local_reciter.sh"
echo ""
echo "  Score a user:       RECITER_API_KEY=local RECITER_API_URL=http://localhost:8081 \\"
echo "                      python3 scripts/run_external_validation.py \\"
echo "                      --institution fredhutch --score-only --adaptive --base-url http://localhost:8081 \\"
echo "                      --data-file \"external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx\""
echo ""
