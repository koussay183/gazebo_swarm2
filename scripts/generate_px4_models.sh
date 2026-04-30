#!/bin/bash
# Génère les répertoires de modèles par instance pour PX4 SITL
# Chaque instance a son propre namespace (x500_0, x500_1, x500_2, x500_3)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
NUM_DRONES=${1:-4}

echo "=== Génération des modèles x500 par instance (${NUM_DRONES} drones) ==="
for i in $(seq 0 $((NUM_DRONES - 1))); do
    echo "  → x500_$i"
    mkdir -p "$MODELS_DIR/x500_$i"
    # Remplace le namespace dans les plugins moteur
    sed "s|<robotNamespace>x500</robotNamespace>|<robotNamespace>x500_$i</robotNamespace>|g" \
        "$MODELS_DIR/x500/model.sdf" > "$MODELS_DIR/x500_$i/model.sdf"
    # Met à jour model.config
    sed "s|<name>x500</name>|<name>x500_$i</name>|g" \
        "$MODELS_DIR/x500/model.config" > "$MODELS_DIR/x500_$i/model.config"
done
echo "=== Modèles générés dans $MODELS_DIR ==="
