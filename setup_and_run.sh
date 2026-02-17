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
# Bloque 2: Configuración del Entorno pyenv (Optimizado para no reinstalar)
# --------------------------------------------------------------------------

PYENV_ROOT="$HOME/.pyenv"
ENV_NAME="train-env"
PYTHON_VERSION="3.11.8"

echo "=========================="
echo " 2) Instalando pyenv y entorno virtual"
echo "=========================="

# 2a. Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# 2b. Install pyenv if not present
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

# 2e. Create virtual environment if missing
if ! pyenv virtualenvs --bare | grep -q "$ENV_NAME"; then
    pyenv virtualenv $PYTHON_VERSION $ENV_NAME
fi

# 2f. Activate environment
pyenv activate $ENV_NAME
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
gsutil -m rsync -r gs://fire_model_dataset/ .

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

sudo shutdown -h now