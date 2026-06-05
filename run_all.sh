#!/bin/bash

# 1. Lance le benchmark Rust
echo "=== [1/2] Extraction des données avec Rust ==="
cargo build --release
perf record --call-graph dwarf ./target/release/first_impl

# Le symbole && signifie : "Si la commande précédente a réussi, passe à la suivante"
if [ $? -eq 0 ]; then
    # 2. Génère les graphiques
    echo -e "\n=== [2/2] Génération des graphiques avec Python ==="
    python3 plot_benchmarks.py
else
    echo "Erreur lors de l'exécution du benchmark Rust. Script Python annulé."
fi