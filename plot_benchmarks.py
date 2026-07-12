import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

csv_3d_filename = "csv/benchmark_3d_data.csv"
ratio_csv = "csv/multiplication_ratio.csv"
offline_csv = "csv/offline_seq_vs_parallel.csv"  # NEW ! TO UNDERSTAND : output of bench_offline_seq_vs_parallel
csv_3d_memory_filename = "csv/benchmark_3d_memory_data.csv"  # NEW ! TO UNDERSTAND : output of run_all_sc_memory_benchmark

# ==============================================================================
# 1. 2D COMPARATIVE GRAPH GENERATION (PER DEGREE)
# ==============================================================================
if os.path.exists(csv_3d_filename):
    df_global = pd.read_csv(csv_3d_filename)
    
    # Extract unique degree values from the CSV dataset
    unique_degrees = df_global['Degree'].unique()
    
    for d in unique_degrees:
        # Filter and sort data for the current degree configuration
        df_d = df_global[df_global['Degree'] == d].sort_values(by='Variables')
        
        # ----------------------------------------------------------------------
        # GRAPH A: Global Protocol Benchmark (Main Curves)
        # ----------------------------------------------------------------------
        plt.figure(figsize=(11, 7))
        
        plt.plot(df_d['Variables'], df_d['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['LinearTime_Vanilla_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['EvalProductSV_Total_ms'], 'o-', color='crimson', label='EvalProductSV (ms) - Total', linewidth=2.5)
        
        # Sub-phases of EvalProductSV (dashed lines)
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
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(curve_img, dpi=300)
        plt.close()
        print(f"[OK] Generated global 2D curve plot for degree d={d}: '{curve_img}'")

    # ----------------------------------------------------------------------
    # GRAPH B: SEPARATED BAR CHART (LinearTimeSC Vanilla VS SB1 Optimization)
    # NEW ! TO UNDERSTAND : instead of generating one line-plot per degree (5 files),
    # this now generates a single bar chart comparing all degrees at Variables = 14.
    # ----------------------------------------------------------------------
    fixed_l_sb1 = 14
    df_fixed_sb1 = df_global[df_global['Variables'] == fixed_l_sb1].sort_values(by='Degree')

    if not df_fixed_sb1.empty:
        sb1_degrees = df_fixed_sb1['Degree'].tolist()
        x_positions = range(len(sb1_degrees))
        bar_width = 0.35

        plt.figure(figsize=(10, 6))

        bars_vanilla = plt.bar(
            [x - bar_width / 2 for x in x_positions], df_fixed_sb1['LinearTime_Vanilla_ms'],
            width=bar_width, color='teal', label='LinearTime_SC (Vanilla)'
        )
        bars_sb1 = plt.bar(
            [x + bar_width / 2 for x in x_positions], df_fixed_sb1['LinearTime_SB1_ms'],
            width=bar_width, color='purple', label='LinearTime_SC (SB1 Optimized)'
        )

        # Dynamic label injection on top of each bar
        for bar in list(bars_vanilla) + list(bars_sb1):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                      f"{height:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.yscale('log')
        plt.xticks(list(x_positions), [f'd = {d}' for d in sb1_degrees])
        plt.xlabel('Degree', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Optimization Impact: Vanilla vs SB1 Statically Bookkept (Variables = {fixed_l_sb1})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()

        sb1_vs_vanilla_img = 'graphs/linear_time_vanilla_vs_sb1.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(sb1_vs_vanilla_img, dpi=300)
        plt.close()
        print(f"[OK] Generated Vanilla vs SB1 bar chart: '{sb1_vs_vanilla_img}'")
    else:
        print(f"[WARN] No sumcheck benchmark data found for Variables = {fixed_l_sb1}, skipping Vanilla vs SB1 bar chart.")

# ==============================================================================
# 2. GLOBAL 3D SURFACE MODEL GENERATION
# ==============================================================================
if os.path.exists(csv_3d_filename):
    df = pd.read_csv(csv_3d_filename)
    fig = plt.figure(figsize=(16, 8))
    
    # Subplot 1: Vanilla LinearTimeSC
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_trisurf(df['Variables'], df['Degree'], df['LinearTime_Vanilla_ms'], cmap='viridis', edgecolor='none', alpha=0.85)
    ax1.set_title('Vanilla LinearTimeSC Execution Cost', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax1.set_ylabel('Degree ($d$)', fontweight='bold')
    ax1.set_zlabel('Execution Time (ms)', fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='ms')
    
    # Subplot 2: Optimized EvalProductSV
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
# 3. BAR CHART GENERATION (SANITY CHECK 1 - 4-WAY CONFIGURATION MATRIX)
# ==============================================================================
if os.path.exists(ratio_csv):
    df_ratio = pd.read_csv(ratio_csv)
    
    plt.figure(figsize=(12, 6))
    # 4 distinct colors for the 4 combinations
    colors = ['#4a6572', '#34495e', '#1abc9c', '#009688']
    
    bars = plt.bar(df_ratio['Operation'], df_ratio['Time_ms'], color=colors, width=0.6)
    
    # Dynamic label injection on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                 f"{height:.2f} ms", ha='center', va='bottom', fontweight='bold')
                 
    plt.ylabel('Execution Time (ms) for 1M operations', fontsize=11, fontweight='bold')
    plt.title('Sanity Check 1: 4-Way Arithmetic Performance Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add padding to top of the graph so labels don't get cut off
    plt.ylim(0, df_ratio['Time_ms'].max() * 1.15)
    plt.tight_layout()
    
    ratio_img = "graphs/arithmetic_speedup_benchmark.png"
    plt.savefig(ratio_img, dpi=300)
    plt.close()
    print(f"[OK] Generated 4-way arithmetic bar chart: '{ratio_img}'")

# ==============================================================================
# NEW ! TO UNDERSTAND
# 4. OFFLINE PRECOMPUTATION BENCHMARK (bench_offline_seq_vs_parallel)
#    Fixed at Variables = 14, compares Sequential vs Parallel across all degrees
# ==============================================================================
if os.path.exists(offline_csv):
    df_offline = pd.read_csv(offline_csv)

    fixed_l = 14
    df_fixed = df_offline[df_offline['Variables'] == fixed_l].sort_values(by='Degree')

    if not df_fixed.empty:
        degrees = df_fixed['Degree'].tolist()
        x_positions = range(len(degrees))
        bar_width = 0.35

        plt.figure(figsize=(11, 7))

        bars_seq = plt.bar(
            [x - bar_width / 2 for x in x_positions], df_fixed['Offline_Sequential_ms'],
            width=bar_width, color='#34495e', label='Sequential'
        )
        bars_par = plt.bar(
            [x + bar_width / 2 for x in x_positions], df_fixed['Offline_Parallel_ms'],
            width=bar_width, color='#1abc9c', label='Parallel'
        )

        # Dynamic label injection on top of each bar
        for bar in list(bars_seq) + list(bars_par):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                      f"{height:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.yscale('log')
        plt.xticks(list(x_positions), [f'd = {d}' for d in degrees])
        plt.xlabel('Degree', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel(f'Offline precomputation time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'EvalProductSV: Sequential vs Parallel Offline Precomputation (Variables = {fixed_l})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()

        offline_img = 'graphs/offline_seq_vs_parallel_benchmark.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(offline_img, dpi=300)
        plt.close()
        print(f"[OK] Generated offline sequential vs parallel bar chart: '{offline_img}'")
    else:
        print(f"[WARN] No offline benchmark data found for Variables = {fixed_l}, skipping bar chart.")

# ==============================================================================
# NEW ! TO UNDERSTAND
# 5. MEMORY BENCHMARK GRAPHS (Arkworks vs LinearTimeSC vs EvalProductSV)
#    Mirrors sections 1 (2D per-degree curves) and 2 (3D surface) above, but for
#    csv/benchmark_3d_memory_data.csv (peak KB instead of ms).
# ==============================================================================
if os.path.exists(csv_3d_memory_filename):
    df_mem_global = pd.read_csv(csv_3d_memory_filename)
    unique_mem_degrees = df_mem_global['Degree'].unique()

    # ----------------------------------------------------------------------
    # GRAPH A-MEM: Global Protocol Memory Benchmark (Main Curves, per degree)
    # ----------------------------------------------------------------------
    for d in unique_mem_degrees:
        df_d = df_mem_global[df_mem_global['Degree'] == d].sort_values(by='Variables')

        plt.figure(figsize=(11, 7))

        plt.plot(df_d['Variables'], df_d['Arkworks_KB'], 's-', color='orange', label='Arkworks framework (KB)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['LinearTime_Vanilla_KB'], '^-', color='teal', label='LinearTime_SC (KB)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['EvalProductSV_Total_KB'], 'o-', color='crimson', label='EvalProductSV (KB) - Total', linewidth=2.5)

        if 'EvalProductSV_Offline_KB' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Offline_KB'], 'x--', color='purple', label='EvalProductSV - Offline (Precomp) (KB)', linewidth=1.5, alpha=0.8)
        if 'EvalProductSV_Online_KB' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Online_KB'], 'v--', color='coral', label='EvalProductSV - Online Phase (KB)', linewidth=1.5, alpha=0.8)

        plt.yscale('log')
        plt.xlabel('Number of variables ($\\ell$)', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Peak extra memory (KB) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Comparative Memory Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()

        mem_curve_img = f'graphs/sumcheck_memory_curve_d{d}.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(mem_curve_img, dpi=300)
        plt.close()
        print(f"[OK] Generated memory 2D curve plot for degree d={d}: '{mem_curve_img}'")

    # ----------------------------------------------------------------------
    # GRAPH B-MEM: GLOBAL 3D MEMORY SURFACE MODEL
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_trisurf(df_mem_global['Variables'], df_mem_global['Degree'], df_mem_global['LinearTime_Vanilla_KB'], cmap='viridis', edgecolor='none', alpha=0.85)
    ax1.set_title('Vanilla LinearTimeSC Memory Cost', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax1.set_ylabel('Degree ($d$)', fontweight='bold')
    ax1.set_zlabel('Peak Memory (KB)', fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='KB')

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_trisurf(df_mem_global['Variables'], df_mem_global['Degree'], df_mem_global['EvalProductSV_Total_KB'], cmap='plasma', edgecolor='none', alpha=0.85)
    ax2.set_title('Optimized EvalProductSV Memory Cost', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax2.set_ylabel('Degree ($d$)', fontweight='bold')
    ax2.set_zlabel('Peak Memory (KB)', fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='KB')

    ax1.view_init(elev=25, azim=-135)
    ax2.view_init(elev=25, azim=-135)

    plt.suptitle('Sumcheck Memory Complexity Space Profiling (3D Analysis)', fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout()

    output_3d_mem_img = "graphs/sumcheck_3d_memory_surface.png"
    plt.savefig(output_3d_mem_img, dpi=300)
    plt.close()
    print(f"[OK] Generated 3D memory surface model: '{output_3d_mem_img}'")
else:
    print(f"[WARN] '{csv_3d_memory_filename}' not found, skipping memory benchmark graphs.")