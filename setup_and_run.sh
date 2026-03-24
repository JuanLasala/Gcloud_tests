#!/bin/bash
set -e

# CLI flags
TEST_RUN=false

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --test-run   Ejecuta la pipeline en modo prueba (dataset reducido)
  -h, --help   Muestra esta ayuda
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-run)
            TEST_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Argumento no reconocido: $1"
            usage
            exit 1
            ;;
    esac
done

# Define el directorio del script (donde se copiarán los datos)
SCRIPT_DIR=$(pwd)

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
# Bloque 2: Configuración del Entorno Conda (Optimizado para no reinstalar)
# --------------------------------------------------------------------------

CONDA_PATH="$HOME/miniconda"
ENV_NAME="vit_env"

echo "=========================="
echo " 2) Instalando/Reinstalando Miniconda y Entorno"
echo "=========================="

# 2a. Reinstalar Miniconda si no existe (o si la instalación es vieja)
if [ ! -d "$CONDA_PATH" ]; then
    echo "Instalando Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_PATH
    rm miniconda.sh
fi

# 2b. Inicializar y Activar Conda (necesario en cada ejecución de script)
eval "$($CONDA_PATH/bin/conda shell.bash hook)"
conda activate $ENV_NAME || { 
    echo "Creando entorno Conda nuevo: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
    conda activate $ENV_NAME
}


# --------------------------------------------------------------------------
# Bloque 3: Instalación de Dependencias (Optimizado)
# --------------------------------------------------------------------------

echo "=========================="
echo " 3) Instalando dependencias (solo si es necesario) "
echo "=========================="
# Usamos un archivo 'sentinel' para saber si ya instalamos las dependencias
SENTINEL_FILE="$SCRIPT_DIR/.dependencies_installed"

if [ ! -f "$SENTINEL_FILE" ]; then
    echo "Instalando PyTorch, HuggingFace y utilidades..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets evaluate pillow matplotlib scikit-learn
    touch "$SENTINEL_FILE" # Crea el archivo sentinel
else
    echo "Dependencias ya instaladas (archivo sentinel encontrado)."
fi

# --------------------------------------------------------------------------
# Bloque 4: Copia del Dataset (Optimizado)
# --------------------------------------------------------------------------

echo "=========================="
echo " 4) Descargando dataset a disco local (¡NO se repite si ya existe!) "
echo "=========================="

# Usamos gsutil rsync, que es idempotente: solo copia los archivos nuevos/modificados.
# Como el dataset es estático, la primera vez lo copia todo, las siguientes veces no hace nada.

DATASET_DIR="$SCRIPT_DIR/dataset_rgb"

# Check if required dataset folders exist inside dataset_rgb
if [ -d "$DATASET_DIR/train" ] && [ -d "$DATASET_DIR/test" ] && [ -d "$DATASET_DIR/validation" ]; then
    echo "Dataset already present in $DATASET_DIR (train, test, validation found). Skipping download."
else
    echo "Dataset not found in $DATASET_DIR. Downloading from GCS..."
    #gsutil -m rsync -r gs://fire_model_dataset/ .
    #gsutil -m rsync -r gs://fire_dataset_2/ .
    gsutil -m rsync -r gs://new_rgb_dataset/dataset "$DATASET_DIR"
fi

echo "Dataset copiado a: $SCRIPT_DIR"

# --------------------------------------------------------------------------
# Bloque 5: Ejecución
# --------------------------------------------------------------------------

echo "=========================="
echo " 5) Ejecutando entrenamiento "
echo "=========================="

#python train_vit.py
RESUME_TOTAL_EPOCHS="${RESUME_TOTAL_EPOCHS:-20}"

TRAIN_CMD=(python -u train_efficientnet.py)

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
"${TRAIN_CMD[@]}" 2>&1 | tee training_log.txt # muestra en terminal y guarda en archivo

# Copiar log al directorio real del run (detectado desde la salida de entrenamiento)
RUN_DIR_FROM_LOG=$(grep -oP 'Guardando resultados en:\s*\K.*' training_log.txt | tail -n1 | sed 's/[[:space:]]*$//')
if [ -n "${RUN_DIR_FROM_LOG:-}" ] && [ -d "$RUN_DIR_FROM_LOG" ]; then
    cp training_log.txt "$RUN_DIR_FROM_LOG/training_log.txt"
    echo "Log copiado a: $RUN_DIR_FROM_LOG/training_log.txt"
else
    echo "[WARN] No se pudo detectar un run_dir válido desde training_log.txt; log quedó en $SCRIPT_DIR/training_log.txt"
fi

#python train_vit.py
python train_efficientnet.py


echo "=========================="
echo " Entrenamiento finalizado "
echo "=========================="

sudo shutdown -h now