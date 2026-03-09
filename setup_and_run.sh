#!/bin/bash
set -euo pipefail

TEST_RUN=false
RESUME_RUN_DIR=""
for arg in "$@"; do
    case "$arg" in
        --test-run)
            TEST_RUN=true
            ;;
        --resume-run-dir=*)
            RESUME_RUN_DIR="${arg#*=}"
            ;;
    esac
done

# Define el directorio del script (donde se copiarán los datos)
SCRIPT_DIR=$(pwd)

#Change branch
git checkout v2_second

echo "=========================="
echo " [NOTA] Ejecutando SETUP desde: $SCRIPT_DIR"
echo "=========================="

# --------------------------------------------------------------------------
# Bloque 1: Configuración del Sistema (Debe ejecutarse siempre)
# --------------------------------------------------------------------------

echo "=========================="
echo " 1) Eliminando repositorio APT obsoleto y actualizando sistema "
echo "=========================="
# Eliminar la línea del backports que está dando 404
sudo sed -i '/bullseye-backports/d' /etc/apt/sources.list.d/gcsfuse.list || true
sudo sed -i '/bullseye-backports/d' /etc/apt/sources.list || true
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y wget git unzip # Dependencias básicas

# --------------------------------------------------------------------------
# Bloque 2: Configuración del Entorno pyenv (Optimizado para no reinstalar)
# --------------------------------------------------------------------------

PYENV_ROOT="$HOME/.pyenv"
PYTHON_VERSION="3.11.8"
#ENV_DIR="$HOME/.venvs/train-env"
ENV_DIR="/home/fperdomo/.venvs/train-env" # ruta absoluta para evitar problemas con pyenv

echo "=========================="
echo " 2) Instalando pyenv y entorno virtual"
echo "=========================="

# 2a. Install build dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# 2b. Install pyenv if missing
if [ ! -d "$PYENV_ROOT" ]; then
    echo "Instalando pyenv..."
    git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT
fi

# 2c. Initialize pyenv
export PYENV_ROOT="$PYENV_ROOT"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# 2d. Install Python version if missing
if ! pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
    pyenv install $PYTHON_VERSION
fi

# 2e. Set local python version for this session
pyenv shell $PYTHON_VERSION

# 2f. Create virtual environment if missing
if [ ! -d "$ENV_DIR" ]; then
    echo "Creando entorno virtual en $ENV_DIR"
    python -m venv $ENV_DIR
fi

# 2g. Activate environment
source $ENV_DIR/bin/activate

echo "Entorno activado con Python:"
python --version
# --------------------------------------------------------------------------
# Bloque 3: Instalación de Dependencias (Optimizado)
# --------------------------------------------------------------------------

echo "=========================="
echo " 3) Instalando dependencias (solo si es necesario) "
echo "=========================="

# INSTALAR DEPENDENCIAS
echo "Entorno activado con Python:"
python --version

# 2h. Upgrade pip (recommended)
pip install --upgrade pip

# 2i. Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
else
    echo "No se encontró requirements.txt"
fi

# --------------------------------------------------------------------------
# Bloque 4: Copia del Dataset (Optimizado)
# --------------------------------------------------------------------------

echo "=========================="
echo " 4) Descargando dataset a disco local (¡NO se repite si ya existe!) "
echo "=========================="

# Usamos gsutil rsync, que es idempotente: solo copia los archivos nuevos/modificados.
# Como el dataset es estático, la primera vez lo copia todo, las siguientes veces no hace nada.

DATASET_DIR="$SCRIPT_DIR/dataset"

# Check if required dataset folders exist
if [ -d "$DATASET_DIR/train" ] && [ -d "$DATASET_DIR/test" ] && [ -d "$DATASET_DIR/validation" ]; then
    echo "Dataset already present (train, test, val found). Skipping download."
else
    echo "Dataset not found. Downloading from GCS..."
    #gsutil -m rsync -r gs://fire_model_dataset/ "$DATASET_DIR"
    gsutil -m rsync -r gs://fire_dataset_3/ "$DATASET_DIR"
fi

echo "Dataset copiado a: $DATASET_DIR"

# --------------------------------------------------------------------------
# Bloque 5: Ejecución
# --------------------------------------------------------------------------

echo "=========================="
echo " 5) Ejecutando entrenamiento "
echo "=========================="

#python train_vit.py
RESUME_TOTAL_EPOCHS="${RESUME_TOTAL_EPOCHS:-20}"

TRAIN_CMD=(python train_efficientnet.py)

if [ -n "$RESUME_RUN_DIR" ]; then
    echo "Reanudando run específico: $RESUME_RUN_DIR (hasta ${RESUME_TOTAL_EPOCHS} epochs totales)"
    TRAIN_CMD+=(--resume_from "$RESUME_RUN_DIR" --resume_to_total_epochs "$RESUME_TOTAL_EPOCHS")
else
    echo "Iniciando entrenamiento desde cero (nuevo run por defecto)"
fi

if [ "$TEST_RUN" = true ]; then
    echo "Modo test-run activado: usando subconjunto reducido del dataset"
    TRAIN_CMD+=(--test-run)
fi

if [ -f training_log.txt ]; then
    echo "Eliminando training_log.txt previo en $SCRIPT_DIR"
    rm -f training_log.txt
fi

"${TRAIN_CMD[@]}" 2>&1 | tee training_log.txt # muestra en terminal y guarda en archivo

# Copiar log al directorio real del run (detectado desde la salida de entrenamiento)
RUN_DIR_FROM_LOG=$(grep -oP 'Guardando resultados en:\s*\K.*' training_log.txt | tail -n1 | sed 's/[[:space:]]*$//')
if [ -n "${RUN_DIR_FROM_LOG:-}" ] && [ -d "$RUN_DIR_FROM_LOG" ]; then
    cp training_log.txt "$RUN_DIR_FROM_LOG/training_log.txt"
    rm -f training_log.txt
    echo "Log copiado a: $RUN_DIR_FROM_LOG/training_log.txt"
else
    echo "[WARN] No se pudo detectar un run_dir válido desde training_log.txt; log quedó en $SCRIPT_DIR/training_log.txt"
fi

echo "=========================="
echo " Entrenamiento finalizado "
echo "=========================="

sudo shutdown -h now