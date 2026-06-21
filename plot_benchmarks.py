import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Loading the CSV data
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Error : File 'benchmark_results.csv' not found.")
    exit()

# ==============================================================================
# GRAPH 1 : PERFORMANCE CURVES
# ==============================================================================
plt.figure(figsize=(10, 6))

# Plotting the lines using the new column names
plt.plot(df['Variables'], df['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
plt.plot(df['Variables'], df['LinearTimeSC_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)

# Logarithmic scale for the Y axis because of the exponential nature of 2^ell
plt.yscale('log')

# Customizing axes and titles
plt.xlabel('Number of variables ($\ell$)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Comparative Benchmark: Multivariate Sumcheck Protocol', fontsize=14, fontweight='bold', pad=15)

# Adding a clean grid
plt.grid(True, which="both", ls="--", alpha=0.5)

# Legend
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()

# Save the curve plot
plt.savefig('sumcheck_benchmark_curve.png', dpi=300)
print("Graphic successfully generated under the name 'sumcheck_benchmark_curve.png' !")

# ==============================================================================
# GRAPH 2 : THE BARS (Focus on max variables analyzed, e.g., 14)
# ==============================================================================
max_vars = df['Variables'].max()
data_max = df[df['Variables'] == max_vars]

if data_max.empty:
    print("Warning: No data found for bar chart plotting. Skipped.")
else:
    plt.figure(figsize=(8, 6))
    
    protocols = ['Arkworks', 'LinearTime_SC']
    times_ms = [
        data_max['Arkworks_ms'].values[0], 
        data_max['LinearTimeSC_ms'].values[0]
    ]
    
    raw_labels = [
        f"{times_ms[0]:.2f} ms",
        f"{times_ms[1]:.4f} ms"
    ]
    
    colors = ['orange', 'teal']
    bars = plt.bar(protocols, times_ms, color=colors, width=0.4, edgecolor='black', alpha=0.9)
    
    plt.yscale('log')
    plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'Execution Time Comparison for {max_vars} Variables ($2^{{{max_vars}}}$ points)', fontsize=14, fontweight='bold', pad=20)
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
    plt.ylim(top=plt.ylim()[1] * 4)
    plt.tight_layout()
    
    # Saving using a clear updated filename
    plt.savefig('sumcheck_bar_chart_max_vars.png', dpi=300)
    print(f"Bar chart successfully generated under the name 'sumcheck_bar_chart_max_vars.png' for {max_vars} variables!")