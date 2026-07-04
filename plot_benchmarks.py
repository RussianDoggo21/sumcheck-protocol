import pandas as pd
import matplotlib.pyplot as plt
import os

degrees = [3, 6, 9]

for d in degrees:
    csv_filename = f"benchmark_results_d{d}.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Skipping degree d={d}: '{csv_filename}' not found.")
        continue
        
    df = pd.read_csv(csv_filename)
    
    # ==============================================================================
    # GRAPH 1 : PERFORMANCE CURVES (Offline / Online Architecture)
    # ==============================================================================
    plt.figure(figsize=(11, 7))
    
    # Main curves (Solid lines)
    plt.plot(df['Variables'], df['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
    plt.plot(df['Variables'], df['LinearTimeSC_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)
    plt.plot(df['Variables'], df['EvalProductSV_ms'], 'o-', color='crimson', label='EvalProductSV (ms) - Total', linewidth=2.5)
    
    # Sub-phase curves (Dashed lines)
    plt.plot(df['Variables'], df['EvalProductSV_Offline_ms'], 'x--', color='purple', label='EvalProductSV - Offline (Precomp) (ms)', linewidth=1.5, alpha=0.8)
    plt.plot(df['Variables'], df['EvalProductSV_Online_ms'], 'v--', color='coral', label='EvalProductSV - Online Phase (ms)', linewidth=1.5, alpha=0.8)

    plt.yscale('log')
    plt.xlabel('Number of variables ($\ell$)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'Comparative Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    
    curve_img = f'sumcheck_benchmark_curve_d{d}.png'
    plt.savefig(curve_img, dpi=300)
    plt.close()
    print(f"Generated curve plot with offline/online phases: {curve_img}")