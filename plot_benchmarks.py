import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement des données CSV
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Error : FIle 'benchmark_results.csv' not foundd.")
    exit()

# 2. Conversion de l'unité de l'optimisé (ns -> ms) pour avoir la même échelle partout
# df['Optimized_ms'] = df['Optimized_ns'] / 1_000_000.0

# 3. Création du graphique
plt.figure(figsize=(10, 6))

# Tracé des courbes
plt.plot(df['Monomials'], df['Naive_ms'], 'o-', color='crimson', label='Naive protocol (ms)', linewidth=2)
plt.plot(df['Monomials'], df['Arkworks_ms'], 's-', color='orange', label='Arkworks protocol (ms)', linewidth=2)
plt.plot(df['Monomials'], df['Optimized_ms'], '^-', color='teal', label='Optimized / Small-Value (ms)', linewidth=2)

# Configuration de l'échelle logarithmique pour l'axe Y
plt.yscale('log')

# Personnalisation des axes et titres
plt.xlabel('Number of monomials', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Comparative benchmark of the Sumcheck Protocol', fontsize=14, fontweight='bold', pad=15)

# Ajout d'une grille propre pour la lecture
plt.grid(True, which="both", ls="--", alpha=0.5)

# Légende
plt.legend(fontsize=11, loc='upper left')

# Ajustement des marges pour que ce soit parfait
plt.tight_layout()

# 4. Sauvegarde de l'image pour ton rapport
plt.savefig('sumcheck_benchmark_curve.png', dpi=300)
print("Graphic successufly generated under the name 'sumcheck_benchmark_curve.png' !")

# ==============================================================================
# GRAPHIC 2 : THE BARS (Focus on 200 monomials)
# ==============================================================================
# Extract the row where Monomials == 200
data_200 = df[df['Monomials'] == 200]

if data_200.empty:
    print("Warning: No data found for 200 monomials. Bar chart skipped.")
else:
    plt.figure(figsize=(8, 6))
    
    protocols = ['Naive', 'Arkworks', 'Optimized\n(Small-Value)']
    # Values in ms for the correct scale on the log plot
    times_ms = [
        data_200['Naive_ms'].values[0], 
        data_200['Arkworks_ms'].values[0], 
        data_200['Optimized_ms'].values[0]
    ]
    
    # Raw values with their original units for the labels
    raw_labels = [
        f"{data_200['Naive_ms'].values[0]:.2f} ms",
        f"{data_200['Arkworks_ms'].values[0]:.2f} ms",
        f"{data_200['Optimized_ms'].values[0] :.4f} ms"
    ]
    
    colors = ['crimson', 'orange', 'teal']
    bars = plt.bar(protocols, times_ms, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    
    # Log scale is mandatory here too because of the huge performance gap
    plt.yscale('log')
    
    plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Execution Time Comparison for 200 Monomials', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, axis='y', which="both", ls="--", alpha=0.3)
    
    # Add text labels on top of the bars
    for bar, label in zip(bars, raw_labels):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height * 1.2, # Place it slightly above the bar on a log scale
            label, 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontweight='bold'
        )
        
    # Extra padding on top to prevent text clipping
    plt.ylim(top=plt.ylim()[1] * 3)
    plt.tight_layout()
    
    plt.savefig('sumcheck_bar_chart_200.png', dpi=300)
    print("Bar chart successfully generated under the name 'sumcheck_bar_chart_200.png' !")