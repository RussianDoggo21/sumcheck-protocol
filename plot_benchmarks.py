import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

csv_3d_filename = "csv/benchmark_3d_data.csv"
ratio_csv = "csv/multiplication_ratio.csv"

# ==============================================================================
# 1. GENERATION DES GRAPHES COMPARATIFS 2D (PAR DEGRE)
# ==============================================================================
if os.path.exists(csv_3d_filename):
    df_global = pd.read_csv(csv_3d_filename)
    
    # Trouver tous les degrés uniques présents dans le fichier CSV
    unique_degrees = df_global['Degree'].unique()
    
    for d in unique_degrees:
        # Filtrer les données pour le degré en cours
        df_d = df_global[df_global['Degree'] == d].sort_values(by='Variables')
        
        plt.figure(figsize=(11, 7))
        
        # Courbes principales (Lignes pleines)
        plt.plot(df_d['Variables'], df_d['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['LinearTimeSC_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['EvalProductSV_Total_ms'], 'o-', color='crimson', label='EvalProductSV (ms) - Total', linewidth=2.5)
        
        # Sous-phases de ton protocole (Lignes pointillées)
        if 'EvalProductSV_Offline_ms' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Offline_ms'], 'x--', color='purple', label='EvalProductSV - Offline (Precomp) (ms)', linewidth=1.5, alpha=0.8)
        if 'EvalProductSV_Online_ms' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Online_ms'], 'v--', color='coral', label='EvalProductSV - Online Phase (ms)', linewidth=1.5, alpha=0.8)

        plt.yscale('log')
        plt.xlabel('Number of variables ($\\ell$)', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Comparative Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        
        curve_img = f'graphs/sumcheck_benchmark_curve_d{d}.png'
        plt.savefig(curve_img, dpi=300)
        plt.close()
        print(f"[OK] Generated 2D curve plot for degree d={d}: '{curve_img}'")

# ==============================================================================
# 2. GENERATION DE LA SURFACE 3D GLOBALE
# ==============================================================================
if os.path.exists(csv_3d_filename):
    df = pd.read_csv(csv_3d_filename)
    fig = plt.figure(figsize=(16, 8))
    
    # Subplot 1 : Vanilla LinearTimeSC
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_trisurf(df['Variables'], df['Degree'], df['LinearTimeSC_ms'], cmap='viridis', edgecolor='none', alpha=0.85)
    ax1.set_title('Vanilla LinearTimeSC Execution Cost', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax1.set_ylabel('Degree ($d$)', fontweight='bold')
    ax1.set_zlabel('Execution Time (ms)', fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='ms')
    
    # Subplot 2 : Optimized EvalProductSV
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_trisurf(df['Variables'], df['Degree'], df['EvalProductSV_Total_ms'], cmap='plasma', edgecolor='none', alpha=0.85)
    ax2.set_title('Optimized EvalProductSV (1-Round SV Path)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax2.set_ylabel('Degree ($d$)', fontweight='bold')
    ax2.set_zlabel('Execution Time (ms)', fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='ms')
    
    ax1.view_init(elev=25, azim=-135)
    ax2.view_init(elev=25, azim=-135)
    
    plt.suptitle('Sumcheck Complexity Space Profiling (3D Analysis)', fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    output_3d_img = "graphs/sumcheck_3d_complexity_surface.png"
    plt.savefig(output_3d_img, dpi=300)
    plt.close()
    print(f"[OK] Generated 3D surface model: '{output_3d_img}'")

# ==============================================================================
# 3. GENERATION DU DIAGRAMME EN BARRES (SANITY CHECK 1 RATIO)
# ==============================================================================
if os.path.exists(ratio_csv):
    df_ratio = pd.read_csv(ratio_csv)
    
    plt.figure(figsize=(8, 5))
    colors = ['#334e68', '#009688']
    bars = plt.bar(df_ratio['Operation'], df_ratio['Time_ms'], color=colors, width=0.5)
    
    slow_time = df_ratio.iloc[0]['Time_ms']
    fast_time = df_ratio.iloc[1]['Time_ms']
    speedup = slow_time / fast_time
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height*0.02),
                 f"{height:.2f} ms", ha='center', va='bottom', fontweight='bold')
                 
    plt.ylabel('Execution Time (ms) for 1M operations', fontsize=11, fontweight='bold')
    plt.title(f'Sanity Check 1: Raw Arithmetic Acceleration ({speedup:.2f}x Faster)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.ylim(0, max(slow_time, fast_time) * 1.15)
    plt.tight_layout()
    
    ratio_img = "graphs/arithmetic_speedup_benchmark.png"
    plt.savefig(ratio_img, dpi=300)
    plt.close()
    print(f"[OK] Generated arithmetic bar chart: '{ratio_img}'")