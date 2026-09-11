#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#DOCKER_IMAGE_TAG="example_algorithm_debug"
DOCKER_IMAGE_TAG="chimera_agent_cimore_v1"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  --file "${SCRIPT_DIR}/Dockerfile_Baseline" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
