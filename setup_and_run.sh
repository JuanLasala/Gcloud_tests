#!/bin/bash
set -e

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
gsutil -m rsync -r gs://training_data_v1_new/dataset/ .

echo "Dataset copiado a: $SCRIPT_DIR"

# --------------------------------------------------------------------------
# Bloque 5: Ejecución
# --------------------------------------------------------------------------

echo "=========================="
echo " 5) Ejecutando entrenamiento "
echo "=========================="

#python train_vit.py
python train_efficientnet.py

echo "=========================="
echo " Entrenamiento finalizado "
echo "=========================="