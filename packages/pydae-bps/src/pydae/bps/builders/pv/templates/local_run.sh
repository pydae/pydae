#!/usr/bin/env bash
# Direct (no-Mininet) plant: pydae_sim + pydae_ppc, talking over plain HTTP.
#
# Brings up the bare pydae model and the controller in one terminal; the PPC
# management API is then reachable at http://localhost:<API_PORT>/ for the test
# drivers (Terminal B).  Ctrl-C stops the controller; the simulator is torn
# down on exit.
#
#   ./run.sh                 # defaults below
#   PYDAE_PORT=8001 API_PORT=5500 ./run.sh
set -euo pipefail

_HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3}"
JSON="$_HERE/__JSON_FILENAME__"
PYDAE_PORT="${PYDAE_PORT:-8001}"
API_PORT="${API_PORT:-5500}"

echo "[local] launching pydae_sim on :$PYDAE_PORT (building model) …"
"$PY" -m colinker.pydae_sim --json "$JSON" --port "$PYDAE_PORT" &
SIM_PID=$!
trap 'echo; echo "[local] stopping pydae_sim ($SIM_PID)"; kill "$SIM_PID" 2>/dev/null || true' EXIT

echo "[local] waiting for pydae_sim …"
until curl -s -o /dev/null --max-time 2 "http://localhost:$PYDAE_PORT/cosim/measurements"; do
    kill -0 "$SIM_PID" 2>/dev/null || { echo "[local] ERROR: pydae_sim died"; exit 1; }
    sleep 1
done

echo "[local] pydae_ppc -> http://localhost:$PYDAE_PORT   PPC API on :$API_PORT"
echo "[local] (Terminal B:  PPC_URL=http://localhost:$API_PORT \$PY tests/frequency_response/run.py --mode p)"
exec "$PY" -m colinker.pydae_ppc --json "$JSON" \
    --pydae-url "http://localhost:$PYDAE_PORT" \
    --api-host 0.0.0.0 --api-port "$API_PORT"
