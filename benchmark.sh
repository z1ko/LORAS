#!/bin/bash

#set -x           # make script verbose
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

csv_header="modality temporal_state_dim temporal_layers_count model_dim model_size_megabytes state_size_kilobytes elapsed_ms fps"

params_model_size=( 64 128 512 1024 2048 )
params_temporal_state_dim=( 64 128 512 1024 2048 )
params_temporal_layers_count=( 1 2 3 4 5 6 )

# DESC: Main control flow
# ARGS: $@ (optional): Arguments provided to the script
# OUTS: None
function main() {

    # Activate python environment
    #source "venv/bin/activate"

    # Create output file
    output_file="${PWD}/benchmark_output.csv"
    touch "${output_file}"
    echo "${csv_header}" > "${output_file}"

    for model_size in ${params_model_size[@]}; do
        for temporal_state_dim in ${params_temporal_state_dim[@]}; do
            for temporal_layers_count in ${params_temporal_layers_count[@]}; do
                echo "benchmark with ${model_size} ${temporal_state_dim} ${temporal_layers_count}"
                python benchmark.py \
                    --chain \
                    --model_size ${model_size} \
                    --temporal_state_dim ${temporal_state_dim} \
                    --temporal_layers_count ${temporal_layers_count} \
                    2>/dev/null 1>>"${output_file}"
            done
        done
    done

    #deactivate
}

# Invoke main with args if not sourced
# Approach via: https://stackoverflow.com/a/28776166/8787985
if ! (return 0 2> /dev/null); then
    main "$@"
fi