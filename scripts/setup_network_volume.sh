#!/usr/bin/env bash
# =============================================================================
# setup_network_volume.sh
#
# One-time setup: creates a RunPod Network Volume and downloads all model
# weights onto it using a temporary GPU pod.
#
# Prerequisites:
#   - runpodctl installed  (brew install runpod/runpodctl/runpodctl)
#   - RUNPOD_API_KEY exported (or configured via runpodctl doctor)
#   - At least $5 RunPod account balance (required to create network volumes)
#   - HF_TOKEN exported (for gated models)
#   - SSH key registered: runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
#     (or let this script create ~/.runpod/ssh/RunPod-Key-Go via runpodctl ssh add-key)
# =============================================================================

set -euo pipefail

# --- Config ------------------------------------------------------------------
VOLUME_NAME="${VOLUME_NAME:-ai-model-storage}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-200}"
DATACENTER="${DATACENTER:-EU-RO-1}"   # EU-RO-1 has RTX 4090; volume datacenter overrides when reusing
GPU_ID="${GPU_ID:-}"                  # leave empty to auto-pick in the target datacenter
POD_NAME="${POD_NAME:-model-downloader}"
POD_IMAGE="${POD_IMAGE:-runpod/base:0.6.2-cuda12.1.0}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"   # SECURE or COMMUNITY
AUTO_GPU="${AUTO_GPU:-1}"             # 1 = auto-pick GPU when requested type unavailable
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-600}"
MIN_BALANCE_USD="${MIN_BALANCE_USD:-5}"
# -----------------------------------------------------------------------------

die_runpod_error() {
  local context="$1"
  local output="$2"

  if echo "$output" | grep -qi "at least \$5"; then
    echo ""
    echo "ERROR: RunPod requires at least \$${MIN_BALANCE_USD} in your account to create a network volume."
    echo "       Add credits at: https://www.runpod.io/console/user/billing"
    echo "       Check balance:  runpodctl user"
    exit 1
  fi

  echo ""
  echo "ERROR: $context"
  echo "$output" | head -5
  exit 1
}

check_account_balance() {
  local user_json balance
  user_json=$(runpodctl user -o json 2>&1) || die_runpod_error "Could not fetch RunPod account info" "$user_json"

  balance=$(USER_JSON="$user_json" MIN_BALANCE="$MIN_BALANCE_USD" python3 -c "
import json, os, sys
data = json.loads(os.environ['USER_JSON'])
for key in ('balance', 'credit', 'clientBalance', 'currentSpend', 'spendLimit'):
    if key in data and data[key] is not None:
        print(data[key])
        break
else:
    # nested user object
    user = data.get('user', data)
    for key in ('balance', 'credit', 'clientBalance'):
        if key in user and user[key] is not None:
            print(user[key])
            break
" 2>/dev/null || true)

  if [[ -n "${balance:-}" ]]; then
    echo "==> RunPod account balance: \$${balance}"
    if python3 -c "import sys; sys.exit(0 if float('${balance}') >= float('${MIN_BALANCE_USD}') else 1)" 2>/dev/null; then
      return 0
    fi
    echo ""
    echo "ERROR: Insufficient RunPod balance (\$${balance}). At least \$${MIN_BALANCE_USD} is required."
    echo "       Add credits at: https://www.runpod.io/console/user/billing"
    exit 1
  fi

  echo "==> Could not read balance automatically — continuing (RunPod will reject if below \$${MIN_BALANCE_USD})"
}

json_field() {
  local json="$1"
  local field="$2"
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['$field'])" <<< "$json"
}

# runpodctl may return exit code 1 even on success; trust JSON with an "id" field.
extract_id_or_die() {
  local context="$1"
  local json="$2"
  local resource_id

  resource_id=$(python3 -c "
import json, sys

raw = sys.stdin.read().strip()
if not raw:
    sys.exit(1)

def parse(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return None

d = parse(raw)
if not d:
    sys.exit(1)
if d.get('error'):
    print('ERROR:' + (d['error'] if isinstance(d['error'], str) else json.dumps(d['error'])), file=sys.stderr)
    sys.exit(1)
rid = d.get('id')
if not rid:
    sys.exit(1)
print(rid)
" <<< "$json" 2>/dev/null) || die_runpod_error "$context" "$json"

  echo "$resource_id"
}

get_volume_datacenter() {
  local volume_id="$1"
  runpodctl network-volume get "$volume_id" -o json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('dataCenterId') or d.get('dataCenter') or '')
" 2>/dev/null || true
}

# Network volumes are datacenter-bound — pods must use the same region.
# Auto-picks a GPU with stock when the requested type is unavailable.
resolve_gpu_for_datacenter() {
  local dc_json
  dc_json=$(runpodctl datacenter list -o json 2>&1) || die_runpod_error "Could not list datacenters" "$dc_json"

  echo "==> Resolving GPU for $DATACENTER..."
  GPU_ID=$(DC_JSON="$dc_json" DATACENTER="$DATACENTER" GPU_ID="$GPU_ID" AUTO_GPU="$AUTO_GPU" python3 <<'PY'
import json, os, sys

data = json.loads(os.environ["DC_JSON"])
dcs = data if isinstance(data, list) else data.get("datacenters", [])
target = next((d for d in dcs if d.get("id") == os.environ["DATACENTER"]), None)
if not target:
    print(f"ERROR: datacenter {os.environ['DATACENTER']} not found", file=sys.stderr)
    sys.exit(1)

gpus = target.get("gpuAvailability") or []
if not gpus:
    print(f"ERROR: no GPUs listed for {os.environ['DATACENTER']}", file=sys.stderr)
    print("       Pick another DATACENTER or create the volume there first.", file=sys.stderr)
    sys.exit(1)

want = (os.environ.get("GPU_ID") or "").strip()
auto = os.environ.get("AUTO_GPU", "1") == "1"

def stock_score(status: str) -> int:
    s = (status or "").lower()
    if s in ("none", "unavailable", "out", "out of stock"):
        return -1
    return {"high": 4, "medium": 3, "low": 2}.get(s, 1)

def matches(g, needle: str) -> bool:
    n = needle.lower()
    gid = (g.get("gpuId") or "").lower()
    disp = (g.get("displayName") or "").lower()
    return n in gid or n in disp or gid in n

if want:
    match = next((g for g in gpus if matches(g, want)), None)
    if match and stock_score(match.get("stockStatus")) >= 0:
        print(f"    Using requested GPU: {match['gpuId']} (stock: {match.get('stockStatus', '?')})", file=sys.stderr)
        print(match["gpuId"])
        sys.exit(0)
    if not auto:
        print(f"ERROR: {want} is not available in {os.environ['DATACENTER']}", file=sys.stderr)
        for g in gpus:
            print(f"       - {g.get('gpuId')} (stock: {g.get('stockStatus', '?')})", file=sys.stderr)
        sys.exit(1)
    if want:
        print(f"    Requested GPU not available in {os.environ['DATACENTER']}: {want}", file=sys.stderr)

# Prefer cheaper GPUs for a one-time model download; fall back to whatever has stock.
preferred = [
    "rtx 4090", "rtx 3090", "l4", "rtx a5000", "rtx a6000", "a40",
    "rtx a4000", "rtx 5090", "h100", "h200",
]

def preference_rank(gpu_id: str) -> int:
    gid = gpu_id.lower()
    for i, token in enumerate(preferred):
        if token in gid:
            return i
    return len(preferred)

candidates = [g for g in gpus if stock_score(g.get("stockStatus")) > 0]
if not candidates:
    candidates = gpus  # last resort: try anything listed

best = max(
    candidates,
    key=lambda g: (stock_score(g.get("stockStatus")), -preference_rank(g.get("gpuId", ""))),
)

print(f"    Auto-selected GPU: {best['gpuId']} (stock: {best.get('stockStatus', '?')})", file=sys.stderr)
print(best["gpuId"])
PY
) || die_runpod_error "Could not resolve GPU for $DATACENTER" "$GPU_ID"

  echo "    Selected: $GPU_ID"
}

ensure_ssh_key() {
  local key_count
  key_count=$(runpodctl ssh list-keys -o json | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('keys', [])))")

  if [[ "$key_count" -gt 0 ]]; then
    return 0
  fi

  echo "==> No SSH keys on your RunPod account — registering one..."
  local pub_key="${HOME}/.ssh/id_ed25519.pub"
  local runpod_key="${HOME}/.runpod/ssh/RunPod-Key-Go.pub"

  if [[ -f "$pub_key" ]]; then
    runpodctl ssh add-key --key-file "$pub_key"
  elif [[ -f "$runpod_key" ]]; then
    runpodctl ssh add-key --key-file "$runpod_key"
  else
    echo "ERROR: No SSH public key found."
    echo "       Create one with:  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519"
    echo "       Then run:         runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub"
    exit 1
  fi
}

_ssh_key_path() {
  for key in "${HOME}/.ssh/id_ed25519" "${HOME}/.runpod/ssh/RunPod-Key-Go"; do
    if [[ -f "$key" ]]; then
      echo "$key"
      return 0
    fi
  done
  return 1
}

_ssh_opts() {
  # accept-new: auto-accept first connection to ssh.runpod.io (BatchMode blocks "yes/no" prompt)
  echo "-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=20"
}

# Build: ssh {podHostId}@ssh.runpod.io -i ~/.ssh/id_ed25519
# RunPod console uses machine.podHostId; runpodctl ssh info often returns "pod not ready".
build_proxy_ssh_from_pod() {
  local pod_id="$1"
  local pod_json key_path opts
  pod_json=$(runpodctl pod get "$pod_id" --include-machine -o json 2>/dev/null) || return 1
  key_path=$(_ssh_key_path) || return 1
  opts=$(_ssh_opts)

  python3 -c "
import json, os, sys
d = json.load(sys.stdin)
machine = d.get('machine') or {}
pod_host = machine.get('podHostId') or ''
if not pod_host:
    sys.exit(1)
key = os.environ['KEY_PATH']
opts = os.environ['SSH_OPTS']
print(f'ssh {pod_host}@ssh.runpod.io -i {key} {opts}')
" KEY_PATH="$key_path" SSH_OPTS="$opts" <<< "$pod_json" 2>/dev/null
}

# Parse runpodctl ssh info JSON — field name varies: sshCommand, command, etc.
get_ssh_cmd() {
  local pod_id="$1"
  local ssh_json ssh_cmd key_path opts

  key_path=$(_ssh_key_path) || true
  opts=$(_ssh_opts)

  ssh_json=$(runpodctl ssh info "$pod_id" -o json 2>/dev/null) || true
  if [[ -n "$ssh_json" ]]; then
    ssh_cmd=$(KEY_PATH="$key_path" SSH_OPTS="$opts" python3 -c "
import json, os, sys
d = json.load(sys.stdin)
if d.get('error'):
    sys.exit(1)
for field in ('sshCommand', 'command', 'ssh_command'):
    if d.get(field):
        cmd = d[field]
        if 'StrictHostKeyChecking' not in cmd:
            cmd += ' ' + os.environ.get('SSH_OPTS', '')
        print(cmd)
        sys.exit(0)
if d.get('host'):
    key = d.get('privateKeyPath') or d.get('keyPath') or os.environ.get('KEY_PATH', '')
    user = d.get('user', 'root')
    port = d.get('port')
    if port and key:
        opts = os.environ.get('SSH_OPTS', '')
        print(f'ssh {user}@{d[\"host\"]} -p {port} -i {key} {opts}')
        sys.exit(0)
sys.exit(1)
" <<< "$ssh_json" 2>/dev/null) || true
    if [[ -n "$ssh_cmd" ]]; then
      echo "$ssh_cmd"
      return 0
    fi
  fi

  # Proxy SSH via ssh.runpod.io (matches RunPod console Connect tab)
  ssh_cmd=$(build_proxy_ssh_from_pod "$pod_id") || true
  if [[ -n "$ssh_cmd" ]]; then
    echo "$ssh_cmd"
    return 0
  fi

  # Text fallback: runpodctl may print the ssh line directly
  ssh_cmd=$(runpodctl ssh info "$pod_id" 2>/dev/null | grep -E '^ssh ' | head -1) || true
  if [[ -n "$ssh_cmd" ]]; then
    echo "$ssh_cmd $opts"
    return 0
  fi

  return 1
}

ssh_can_connect() {
  local pod_id="$1"
  local ssh_cmd out
  ssh_cmd=$(get_ssh_cmd "$pod_id") || return 1
  # shellcheck disable=SC2086
  out=$(eval "$ssh_cmd $(printf '%q' 'echo runpod-ssh-ok')" 2>/dev/null) || return 1
  [[ "$out" == *runpod-ssh-ok* ]]
}

wait_for_pod_running() {
  local pod_id="$1"
  local elapsed=0

  while (( elapsed < WAIT_TIMEOUT_SECONDS )); do
    local pod_json status
    pod_json=$(runpodctl pod get "$pod_id" -o json)
    status=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('desiredStatus') or d.get('status') or '')" <<< "$pod_json")

    if [[ "$status" == "RUNNING" ]] && ssh_can_connect "$pod_id"; then
      echo "    … SSH ready (${elapsed}s)"
      return 0
    fi

    sleep 5
    ((elapsed += 5))
    if [[ "$status" == "RUNNING" ]]; then
      echo "    … pod RUNNING, waiting for SSH (${elapsed}s / ${WAIT_TIMEOUT_SECONDS}s)"
    else
      echo "    … waiting for pod ($status, ${elapsed}s / ${WAIT_TIMEOUT_SECONDS}s)"
    fi
  done

  echo "ERROR: Pod $pod_id did not become SSH-ready within ${WAIT_TIMEOUT_SECONDS}s"
  echo "       Try manually: runpodctl ssh info $pod_id"
  exit 1
}

run_on_pod() {
  local pod_id="$1"
  local remote_cmd="$2"
  local ssh_cmd

  ssh_cmd=$(get_ssh_cmd "$pod_id") || {
    echo "ERROR: Could not resolve SSH command for pod $pod_id"
    echo "       Run: runpodctl ssh info $pod_id"
    exit 1
  }

  # shellcheck disable=SC2086
  eval "$ssh_cmd $(printf '%q' "bash") $(printf '%q' "-lc") $(printf '%q' "$remote_cmd")"
}

find_volume_id_by_name() {
  local name="$1"
  # Pass name via argv — env vars set before a pipe only apply to the first command.
  runpodctl network-volume list -o json | python3 -c "
import json, sys
name = sys.argv[1]
data = json.load(sys.stdin)
volumes = data if isinstance(data, list) else data.get('networkVolumes', data.get('volumes', []))
for vol in volumes:
    if vol.get('name') == name:
        print(vol['id'])
        break
" "$name"
}

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Gated models (FLUX.1-dev) may fail."
fi

ensure_ssh_key
check_account_balance

echo "==> Creating Network Volume: $VOLUME_NAME (${VOLUME_SIZE_GB} GB) in $DATACENTER"
VOLUME_ID=$(find_volume_id_by_name "$VOLUME_NAME" || true)

if [[ -n "$VOLUME_ID" ]]; then
  echo "==> Volume already exists: $VOLUME_ID (reusing)"
  VOLUME_DC=$(get_volume_datacenter "$VOLUME_ID")
  if [[ -n "$VOLUME_DC" && "$VOLUME_DC" != "$DATACENTER" ]]; then
    echo "==> Volume datacenter is $VOLUME_DC (overriding DATACENTER=$DATACENTER)"
    DATACENTER="$VOLUME_DC"
  fi
else
  VOLUME_JSON=$(runpodctl network-volume create \
    --name "$VOLUME_NAME" \
    --size "$VOLUME_SIZE_GB" \
    --data-center-id "$DATACENTER" 2>&1) || true

  VOLUME_ID=$(extract_id_or_die "Failed to create network volume" "$VOLUME_JSON")
  echo "==> Volume created: $VOLUME_ID"
fi

resolve_gpu_for_datacenter

DOWNLOAD_POD_NAME="${POD_NAME}-$(date +%s)"
echo "==> Spinning up a temporary pod to download model weights..."
echo "    GPU: $GPU_ID | Datacenter: $DATACENTER | Pod: $DOWNLOAD_POD_NAME"
echo "    Provisioning can take 1–5 minutes — please wait..."

ENV_JSON=$(python3 -c "import json, os; print(json.dumps({'HF_TOKEN': os.environ.get('HF_TOKEN', '')}))")

POD_JSON=$(runpodctl pod create \
  --name "$DOWNLOAD_POD_NAME" \
  --image "$POD_IMAGE" \
  --gpu-id "$GPU_ID" \
  --cloud-type "$CLOUD_TYPE" \
  --data-center-ids "$DATACENTER" \
  --network-volume-id "$VOLUME_ID" \
  --volume-mount-path "/runpod-volume" \
  --env "$ENV_JSON" 2>&1) || true

POD_ID=$(extract_id_or_die "Failed to create download pod" "$POD_JSON")
echo "==> Pod started: $POD_ID — waiting for it to be ready..."
wait_for_pod_running "$POD_ID"

echo "==> Downloading models (this may take 20–40 minutes)..."
run_on_pod "$POD_ID" '
  set -euo pipefail
  PY=$(command -v python3.11 || command -v python3)
  export HF_HOME=/tmp/hf_cache
  mkdir -p "$HF_HOME"
  $PY -m pip install -q huggingface_hub
  $PY - <<'"'"'PY'"'"'
from huggingface_hub import snapshot_download
import os

tok = os.environ.get("HF_TOKEN") or None

print("[1/3] Downloading FLUX.1-schnell...")
snapshot_download(
    "black-forest-labs/FLUX.1-schnell",
    local_dir="/runpod-volume/models/flux",
    token=tok,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)

print("[2/3] Downloading LTX-Video (diffusers files only)...")
snapshot_download(
    "Lightricks/LTX-Video",
    local_dir="/runpod-volume/models/ltxvideo",
    token=tok,
    allow_patterns=[
        "model_index.json",
        "*.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
        "transformer/*",
    ],
)

print("[3/3] Downloading XTTS v2...")
snapshot_download(
    "coqui/XTTS-v2",
    local_dir="/runpod-volume/models/xtts",
    token=tok,
)

print("All models downloaded.")
PY
'

echo "==> Terminating the temporary pod..."
runpodctl pod delete "$POD_ID"

echo ""
echo "=== DONE ==================================================================="
echo "  Network Volume ID : $VOLUME_ID"
echo "  Volume Name       : $VOLUME_NAME"
echo "  Datacenter        : $DATACENTER"
echo ""
echo "  Use this volume when creating your serverless endpoints."
echo "  Attach it at mount path: /runpod-volume"
echo "============================================================================"
