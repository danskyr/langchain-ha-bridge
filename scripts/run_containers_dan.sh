#!/bin/bash

# Start in scripts dir
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# go to services
cd ../services

docker-compose -f home-assistant/docker-compose.yml up -d
docker-compose -f whisper/docker-compose.yml up -d
docker-compose -f piper/docker-compose.yml up -d
