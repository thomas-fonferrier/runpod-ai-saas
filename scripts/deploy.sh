#!/usr/bin/env bash
# =============================================================================
# deploy.sh
#
# Applies all Kubernetes manifests in the correct order.
# Assumes kubectl is configured to point at your cluster.
# =============================================================================

set -euo pipefail

K8S_DIR="$(cd "$(dirname "$0")/../kubernetes" && pwd)"

echo "==> Applying Kubernetes manifests from $K8S_DIR"

kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/pvc.yaml"

echo "WARNING: secrets.yaml contains placeholder values."
echo "         Edit kubernetes/secrets.yaml with your real base64 secrets before applying."
echo "         Skipping secrets.yaml — apply manually when ready:"
echo "           kubectl apply -f kubernetes/secrets.yaml"
echo ""

for model in flux ltxvideo xtts; do
  echo "--- Deploying $model ---"
  kubectl apply -f "$K8S_DIR/$model/deployment.yaml"
  kubectl apply -f "$K8S_DIR/$model/service.yaml"
  kubectl apply -f "$K8S_DIR/$model/hpa.yaml"
done

echo ""
echo "==> Waiting for rollout..."
kubectl rollout status deployment/flux-worker     -n ai-workers --timeout=300s
kubectl rollout status deployment/ltxvideo-worker -n ai-workers --timeout=300s
kubectl rollout status deployment/xtts-worker     -n ai-workers --timeout=300s

echo ""
echo "=== All workers deployed. =================================================="
kubectl get pods -n ai-workers
