set -e

ROM_DIR=$1
if [ ! -d "${ROM_DIR}" ] || [ "$(ls -1 ${ROM_DIR} | wc -l)" -lt 2 ]; then
    python setup.py download_ale --output-dir="$ROM_DIR"
fi
