#!/usr/bin/env bash

# experiment 1: search performance
./scripts/performance.sh

# experiment 2: scaling to larger search spaces
./scripts/scaling.sh

# experiment 3: generalization from limited training samples
./scripts/generalization.sh