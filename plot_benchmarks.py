import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

csv_3d_filename = "csv/benchmark_3d_data.csv"
batch_ratio_csv = "csv/multiplication_ratio_batch.csv"
solo_ratio_csv = "csv/multiplication_ratio_solo.csv"
run_seq_vs_parallel_csv = "csv/run_seq_vs_parallel.csv"  # NEW ! TO UNDERSTAND : renamed from offline_seq_vs_parallel.csv -- output of bench_run_seq_vs_parallel (EvalProductSV has no offline/online split anymore, so this now compares the whole protocol, run_sequential vs run)
csv_3d_memory_filename = "csv/benchmark_3d_memory_data.csv"
bigint_csv = "csv/bigint_vanilla_vs_sb.csv"  # NEW ! TO UNDERSTAND : output of bench_bigint_vanilla_vs_sb (report Section 6.5) -- now swept across the SAME (Degree, Variables) grid as the main sweep, so it can be overlaid directly onto sumcheck_benchmark_curve_d{d}.png
bigint_memory_csv = "csv/bigint_memory.csv"  # NEW ! TO UNDERSTAND : memory equivalent, output of bench_bigint_memory -- overlaid onto sumcheck_memory_curve_d{d}.png

# NEW ! TO UNDERSTAND : loaded once, shared across every per-degree plot below, same
# pattern as df_global for the main sweep.
df_bigint_all = pd.read_csv(bigint_csv) if os.path.exists(bigint_csv) else None
df_bigint_mem_all = pd.read_csv(bigint_memory_csv) if os.path.exists(bigint_memory_csv) else None

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
        # NEW ! TO UNDERSTAND : EvalProductSV_Offline_ms / EvalProductSV_Online_ms no longer
        # exist in the CSV (EvalProductSV is a single `run()` call now), so these two `if`
        # guards below simply never fire anymore -- only 3 curves are plotted (Arkworks,
        # LinearTime_SC, EvalProductSV total). Left in place in case those columns ever come
        # back (e.g. if a real offline/online split is reintroduced later).
        # ----------------------------------------------------------------------
        plt.figure(figsize=(11, 7))
        
        plt.plot(df_d['Variables'], df_d['Arkworks_ms'], 's-', color='orange', label='Arkworks framework (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['LinearTime_Vanilla_ms'], '^-', color='teal', label='LinearTime_SC (ms)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['EvalProductSV_Total_ms'], 'o-', color='crimson', label='EvalProductSV (ms)', linewidth=2.5)
        
        if 'EvalProductSV_Offline_ms' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Offline_ms'], 'x--', color='purple', label='EvalProductSV - Offline (Precomp) (ms)', linewidth=1.5, alpha=0.8)
        if 'EvalProductSV_Online_ms' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Online_ms'], 'v--', color='coral', label='EvalProductSV - Online Phase (ms)', linewidth=1.5, alpha=0.8)

        # NEW ! TO UNDERSTAND : BigInt (StdFr2) curves overlaid on the same plot, when data
        # exists for this degree. Only Vanilla and sb-all are shown (not 1-sb) to keep the
        # plot legible -- the two extremes bracket the technique's full effect within the
        # SAME (Montgomery-free) field, layered on top of the arkworks-based comparison.
        if df_bigint_all is not None:
            df_bigint_d = df_bigint_all[df_bigint_all['Degree'] == d].sort_values(by='Variables')
            if not df_bigint_d.empty:
                plt.plot(df_bigint_d['Variables'], df_bigint_d['Vanilla_ms'], 'D--', color='dimgray', label='BigInt (StdFr2) - Vanilla (ms)', linewidth=1.5, alpha=0.85)
                plt.plot(df_bigint_d['Variables'], df_bigint_d['SBAll_ms'], 'D-', color='#8e44ad', label='BigInt (StdFr2) - sb-all (ms)', linewidth=1.5, alpha=0.85)

        plt.yscale('log')
        plt.xlabel('Number of variables ($\\ell$)', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Comparative Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        plt.tight_layout()
        
        curve_img = f'graphs/sumcheck_benchmark_curve_d{d}.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(curve_img, dpi=300)
        plt.close()
        print(f"[OK] Generated global 2D curve plot for degree d={d}: '{curve_img}'")

    # ----------------------------------------------------------------------
    # GRAPH B: SEPARATED BAR CHART (LinearTimeSC Vanilla VS SB1 Optimization)
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
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_trisurf(df['Variables'], df['Degree'], df['LinearTime_Vanilla_ms'], cmap='viridis', edgecolor='none', alpha=0.85)
    ax1.set_title('Vanilla LinearTimeSC Execution Cost', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Variables ($\\ell$)', fontweight='bold')
    ax1.set_ylabel('Degree ($d$)', fontweight='bold')
    ax1.set_zlabel('Execution Time (ms)', fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='ms')
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_trisurf(df['Variables'], df['Degree'], df['EvalProductSV_Total_ms'], cmap='plasma', edgecolor='none', alpha=0.85)
    ax2.set_title('EvalProductSV Execution Cost', fontsize=12, fontweight='bold', pad=10)
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
# 3a. BAR CHART GENERATION (SANITY CHECK 1, BATCH)
#     NEW ! TO UNDERSTAND : now 5 bars instead of 4 (added "Extrapolate (Small bigints
#     precomputed)", the theoretical-peak-throughput variant) -- one more color added.
# ==============================================================================
if os.path.exists(batch_ratio_csv):
    df_ratio = pd.read_csv(batch_ratio_csv)
    
    plt.figure(figsize=(13, 6))
    colors = ['#4a6572', '#34495e', '#1abc9c', '#009688', '#f39c12']
    
    bars = plt.bar(df_ratio['Operation'], df_ratio['Time_ms'], color=colors[:len(df_ratio)], width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                 f"{height:.2f} ms", ha='center', va='bottom', fontweight='bold')
                 
    plt.ylabel('Execution Time (ms) for a 1,000,000-term dot product', fontsize=11, fontweight='bold')
    plt.title('Dot-Product Performance Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, df_ratio['Time_ms'].max() * 1.15)
    plt.tight_layout()
    
    ratio_img = "graphs/arithmetic_batch_benchmark.png"
    plt.savefig(ratio_img, dpi=300)
    plt.close()
    print(f"[OK] Generated batch arithmetic bar chart: '{ratio_img}'")
else:
    print(f"[WARN] '{batch_ratio_csv}' not found, skipping batch arithmetic bar chart.")

# ==============================================================================
# 3b. BAR CHART GENERATION (SANITY CHECK 1 BIS, SOLO - SINGLE MULTIPLICATION)
# ==============================================================================
if os.path.exists(solo_ratio_csv):
    df_solo = pd.read_csv(solo_ratio_csv)

    plt.figure(figsize=(10, 6))
    colors = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad']

    bars = plt.bar(df_solo['Operation'], df_solo['Time_ns'], color=colors, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + (height * 0.02),
                  f"{height:.2f} ns", ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Execution Time (ns) for a single multiplication', fontsize=11, fontweight='bold')
    plt.title('Single-Multiplication Performance Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, df_solo['Time_ns'].max() * 1.15)
    plt.tight_layout()

    solo_img = "graphs/arithmetic_solo_benchmark.png"
    plt.savefig(solo_img, dpi=300)
    plt.close()
    print(f"[OK] Generated solo single-multiplication bar chart: '{solo_img}'")
else:
    print(f"[WARN] '{solo_ratio_csv}' not found, skipping solo arithmetic bar chart.")

# ==============================================================================
# NEW ! TO UNDERSTAND
# 4. WHOLE-PROTOCOL SEQUENTIAL VS PARALLEL BENCHMARK (bench_run_seq_vs_parallel)
#    Renamed from the offline-only version: EvalProductSV has no offline/online split
#    anymore, so this now compares run_sequential() vs run() for the WHOLE protocol,
#    fixed at Variables = 14, across all degrees.
# ==============================================================================
if os.path.exists(run_seq_vs_parallel_csv):
    df_run = pd.read_csv(run_seq_vs_parallel_csv)

    fixed_l = 14
    df_fixed = df_run[df_run['Variables'] == fixed_l].sort_values(by='Degree')

    if not df_fixed.empty:
        degrees = df_fixed['Degree'].tolist()
        x_positions = range(len(degrees))
        bar_width = 0.35

        plt.figure(figsize=(11, 7))

        bars_seq = plt.bar(
            [x - bar_width / 2 for x in x_positions], df_fixed['Run_Sequential_ms'],
            width=bar_width, color='#34495e', label='Sequential (run_sequential)'
        )
        bars_par = plt.bar(
            [x + bar_width / 2 for x in x_positions], df_fixed['Run_Parallel_ms'],
            width=bar_width, color='#1abc9c', label='Parallel (run)'
        )

        for bar in list(bars_seq) + list(bars_par):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                      f"{height:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.yscale('log')
        plt.xticks(list(x_positions), [f'd = {d}' for d in degrees])
        plt.xlabel('Degree', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('EvalProductSV::run execution time (ms) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'EvalProductSV: Sequential vs Parallel (Whole Protocol, Variables = {fixed_l})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()

        run_img = 'graphs/run_seq_vs_parallel_benchmark.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(run_img, dpi=300)
        plt.close()
        print(f"[OK] Generated run sequential vs parallel bar chart: '{run_img}'")
    else:
        print(f"[WARN] No run seq/parallel data found for Variables = {fixed_l}, skipping bar chart.")
else:
    print(f"[WARN] '{run_seq_vs_parallel_csv}' not found, skipping run seq/parallel bar chart.")

# ==============================================================================
# 5. MEMORY BENCHMARK GRAPHS (Arkworks vs LinearTimeSC vs EvalProductSV)
#    NEW ! TO UNDERSTAND : EvalProductSV_Offline_KB / EvalProductSV_Online_KB no longer exist
#    (single run() call) -- same guarded-plot pattern as section 1, degrades gracefully.
# ==============================================================================
if os.path.exists(csv_3d_memory_filename):
    df_mem_global = pd.read_csv(csv_3d_memory_filename)
    unique_mem_degrees = df_mem_global['Degree'].unique()

    for d in unique_mem_degrees:
        df_d = df_mem_global[df_mem_global['Degree'] == d].sort_values(by='Variables')

        plt.figure(figsize=(11, 7))

        plt.plot(df_d['Variables'], df_d['Arkworks_KB'], 's-', color='orange', label='Arkworks framework (KB)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['LinearTime_Vanilla_KB'], '^-', color='teal', label='LinearTime_SC (KB)', linewidth=2)
        plt.plot(df_d['Variables'], df_d['EvalProductSV_Total_KB'], 'o-', color='crimson', label='EvalProductSV (KB)', linewidth=2.5)

        if 'EvalProductSV_Offline_KB' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Offline_KB'], 'x--', color='purple', label='EvalProductSV - Offline (Precomp) (KB)', linewidth=1.5, alpha=0.8)
        if 'EvalProductSV_Online_KB' in df_d.columns:
            plt.plot(df_d['Variables'], df_d['EvalProductSV_Online_KB'], 'v--', color='coral', label='EvalProductSV - Online Phase (KB)', linewidth=1.5, alpha=0.8)

        # NEW ! TO UNDERSTAND : same BigInt overlay as the time-curve plots (Section 1).
        if df_bigint_mem_all is not None:
            df_bigint_mem_d = df_bigint_mem_all[df_bigint_mem_all['Degree'] == d].sort_values(by='Variables')
            if not df_bigint_mem_d.empty:
                plt.plot(df_bigint_mem_d['Variables'], df_bigint_mem_d['Vanilla_KB'], 'D--', color='dimgray', label='BigInt (StdFr2) - Vanilla (KB)', linewidth=1.5, alpha=0.85)
                plt.plot(df_bigint_mem_d['Variables'], df_bigint_mem_d['SBAll_KB'], 'D-', color='#8e44ad', label='BigInt (StdFr2) - sb-all (KB)', linewidth=1.5, alpha=0.85)

        plt.yscale('log')
        plt.xlabel('Number of variables ($\\ell$)', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Peak extra memory (KB) - Log scale', fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f'Comparative Memory Benchmark: Multivariate Sumcheck (Degree d={d})', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        plt.tight_layout()

        mem_curve_img = f'graphs/sumcheck_memory_curve_d{d}.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(mem_curve_img, dpi=300)
        plt.close()
        print(f"[OK] Generated memory 2D curve plot for degree d={d}: '{mem_curve_img}'")

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
    ax2.set_title('EvalProductSV Memory Cost', fontsize=12, fontweight='bold', pad=10)
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

# ==============================================================================
# NEW ! TO UNDERSTAND
# 6. BIGINT FIELD BENCHMARK (bench_bigint_vanilla_vs_sb)
#    Standalone bar chart, fixed at Variables = 14, complementing the curve overlays
#    added to Sections 1 and 5 above -- kept as-is for a clean, degree-by-degree
#    side-by-side comparison of vanilla vs 1-sb vs sb-all in one place.
# ==============================================================================
if df_bigint_all is not None:
    df_bigint_fixed = df_bigint_all[df_bigint_all['Variables'] == 14].sort_values(by='Degree')

    if not df_bigint_fixed.empty:
        degrees = df_bigint_fixed['Degree'].tolist()
        x_positions = range(len(degrees))
        bar_width = 0.25

        plt.figure(figsize=(11, 7))

        bars_vanilla = plt.bar(
            [x - bar_width for x in x_positions], df_bigint_fixed['Vanilla_ms'],
            width=bar_width, color='#34495e', label='Vanilla'
        )
        bars_sb1 = plt.bar(
            x_positions, df_bigint_fixed['SB1_ms'],
            width=bar_width, color='#1abc9c', label='1-sb (round 0 only)'
        )
        bars_sball = plt.bar(
            [x + bar_width for x in x_positions], df_bigint_fixed['SBAll_ms'],
            width=bar_width, color='#f39c12', label='sb-all (every round)'
        )

        for bar in list(bars_vanilla) + list(bars_sb1) + list(bars_sball):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height * 1.005,
                      f"{height:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.xticks(list(x_positions), [f'd = {d}' for d in degrees])
        plt.xlabel('Degree', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Full-protocol execution time (ms)', fontsize=12, fontweight='bold', labelpad=10)
        plt.title('BigInt Field (StdFr2): Vanilla vs 1-sb vs sb-all (Variables = 14)', fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        plt.ylim(0, df_bigint_fixed['Vanilla_ms'].max() * 1.15)
        plt.tight_layout()

        bigint_img = 'graphs/bigint_vanilla_vs_sb_benchmark.png'
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(bigint_img, dpi=300)
        plt.close()
        print(f"[OK] Generated BigInt vanilla vs sb bar chart: '{bigint_img}'")
    else:
        print(f"[WARN] No BigInt data found for Variables = 14, skipping BigInt vanilla vs sb bar chart.")
else:
    print(f"[WARN] '{bigint_csv}' not found, skipping BigInt vanilla vs sb bar chart.")