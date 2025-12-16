
#!/bin/bash

# Get script directory, resolving symlinks
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"

# Define ANSI color codes as environment variables
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'


if [ -z "${CONDA_DEFAULT_ENV}" ]; then
    echo -e "${RED}[x]: this script must be run inside a conda environment!${RESET}"
    exit 1
fi


if [ ! $# -eq 1 ]; then
    echo -e "${RED}[x]: syntax: $0 <conda env name | path/to/conda/env>!${RESET}"
    exit 1
fi

unset CONDA_ENV_NAME
unset CONDA_ENV_DIR

if [ "${1:0:1}" != "/" ]; then
    CONDA_ENV_NAME=$1
    # Check if conda env exists
    if ! conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
        echo -e "${RED}[x]: conda environment '${CONDA_ENV_NAME}' does not exist!${RESET}"
        exit 1
    fi
else
    # Check if path exists and is a conda environment
    if [ ! -d "$1" ]; then
        echo -e "${RED}[x]: directory '$1' does not exist!${RESET}"
        exit 1
    fi
    if [ ! -d "$1/conda-meta" ]; then
        echo -e "${RED}[x]: '$1' is not a valid conda environment!${RESET}"
        exit 1
    fi
    CONDA_ENV_DIR=$1
    CONDA_ENV_NAME=$(basename $CONDA_ENV_DIR)
fi


if [ -z "${CONDA_ENV_DIR}" ]; then
    CONDA_ENV_DIR=$(conda env list | grep -w "^${CONDA_ENV_NAME}\s" | awk '{print $NF}')
fi

if [ -z "${CONDA_ENV_DIR}" ]; then
    echo -e "${RED}[x]: could not find conda environment: ${CONDA_ENV_NAME}!${RESET}"
    exit 1
fi

mkdir -p $CONDA_ENV_DIR/lib/nvvm/libdevice
cp $CONDA_ENV_DIR/lib/libdevice.10.bc $CONDA_ENV_DIR/lib/nvvm/libdevice/


mkdir -p "$CONDA_ENV_DIR/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_DIR/etc/conda/deactivate.d"
cat << EOF > "$CONDA_ENV_DIR/etc/conda/activate.d/${CONDA_ENV_NAME}.sh"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/anaconda3/envs/${CONDA_ENV_NAME}/lib/"
export TF_CPP_MIN_LOG_LEVEL=3
EOF

cat << EOF > "$CONDA_ENV_DIR/etc/conda/deactivate.d/${CONDA_ENV_NAME}.sh"
unset XLA_FLAGS
unset TF_CPP_MIN_LOG_LEVEL
EOF


echo "[i] post-install done!"
