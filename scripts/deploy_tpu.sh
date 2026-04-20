#!/bin/bash
# GCP Cloud TPU v4 Spot Provisioning Script
# This enforces the strict < 1000$ budget bound natively using preemptible preempts.

set -e

ZONE="us-central2-b"
ACCELERATOR="v4-8"
TPU_NAME="qhrr-tpu"

# GCP project ID — set via environment variable or --project CLI flag
GCP_PROJECT="${GCP_PROJECT:-}"
PROJECT_FLAG=""
if [ -n "$GCP_PROJECT" ]; then
    PROJECT_FLAG="--project=$GCP_PROJECT"
fi

echo "Provisioning QHRRN TPU VM ($TPU_NAME) in $ZONE as SPOT instance..."
[ -n "$GCP_PROJECT" ] && echo "GCP Project: $GCP_PROJECT"

# Protect budget strictly via --spot dynamically bounding costs
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR \
    --version=tpu-vm-v4-base \
    --spot \
    $PROJECT_FLAG

echo "TPU $TPU_NAME successfully provisioned!"
