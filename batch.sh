#!/bin/bash

for filename in configs/sequence/*.yaml; do
    echo "running configuration $filename..."
    python -m main --config $filename
done