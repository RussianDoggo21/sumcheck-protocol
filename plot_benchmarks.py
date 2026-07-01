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
    # GRAPH 1 : PERFORMANCE CURVES
    # ==============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(df['Variables'], df['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
    plt.plot(df['Variables'], df['LinearTimeSC_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)
    plt.plot(df['Variables'], df['EvalProductSV_ms'], 'o-', color='crimson', label='EvalProductSV (ms)', linewidth=2)

    plt.yscale('log')
    plt.xlabel('Number of variables ($\ell$)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'Comparative Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    curve_img = f'sumcheck_benchmark_curve_d{d}.png'
    plt.savefig(curve_img, dpi=300)
    plt.close()
    print(f"Generated curve plot: {curve_img}")

    # ==============================================================================
    # GRAPH 2 : THE BARS (Focus on max variables)
    # ==============================================================================
    max_vars = df['Variables'].max()
    data_max = df[df['Variables'] == max_vars]

    if not data_max.empty:
        plt.figure(figsize=(9, 6))
        protocols = ['Arkworks', 'LinearTime_SC', 'EvalProductSV']
        times_ms = [
            data_max['Arkworks_ms'].values[0], 
            data_max['LinearTimeSC_ms'].values[0],
            data_max['EvalProductSV_ms'].values[0]
        ]
        raw_labels = [f"{times_ms[0]:.2f} ms", f"{times_ms[1]:.4f} ms", f"{times_ms[2]:.4f} ms"]
        colors = ['orange', 'teal', 'crimson']
        
        bars = plt.bar(protocols, times_ms, color=colors, width=0.4, edgecolor='black', alpha=0.9)
        plt.yscale('log')
        plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Execution Time Comparison for {max_vars} Variables ($2^{{{max_vars}}}$ points, d={d})', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, axis='y', which="both", ls="--", alpha=0.3)
        
        for bar, label in zip(bars, raw_labels):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height * 1.2, label, ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.ylim(top=plt.ylim()[1] * 4)
        plt.tight_layout()
        
        bar_img = f'sumcheck_bar_chart_max_vars_d{d}.png'
        plt.savefig(bar_img, dpi=300)
        plt.close()
        print(f"Generated bar chart: {bar_img}")