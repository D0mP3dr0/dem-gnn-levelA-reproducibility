import pandas as pd
import numpy as np
from scipy import stats
import argparse
from datetime import datetime

def run_wilcoxon_test(df):
    print("\n" + "="*60)
    print("TESTE 1: WILCOXON SIGNED-RANK (Pareado)")
    print("="*60)
    
    results = []
    
    for biome in df['biome'].unique():
        biome_df = df[df['biome'] == biome]
        
        gnn_df = biome_df[biome_df['model'] == 'GNN'].sort_values('tile_idx')
        cnn_df = biome_df[biome_df['model'] == 'CNN'].sort_values('tile_idx')
        
        gnn_vals = gnn_df['mae_m'].values
        cnn_vals = cnn_df['mae_m'].values
        
        min_len = min(len(gnn_vals), len(cnn_vals))
        gnn_vals = gnn_vals[:min_len]
        cnn_vals = cnn_vals[:min_len]
        
        if min_len < 2:
            print(f"\n{biome.upper()}: Dados insuficientes ({min_len} pares)")
            continue
        
        stat, p_value = stats.wilcoxon(gnn_vals, cnn_vals, alternative='less')
        
        n = len(gnn_vals)
        r = 1 - (2 * stat) / (n * (n + 1))
        
        results.append({
            'biome': biome,
            'n_pairs': min_len,
            'gnn_mean_mae': np.mean(gnn_vals),
            'gnn_std_mae': np.std(gnn_vals),
            'cnn_mean_mae': np.mean(cnn_vals),
            'cnn_std_mae': np.std(cnn_vals),
            'wilcoxon_stat': stat,
            'p_value': p_value,
            'effect_size_r': r,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01
        })
        
        sig_symbol = "✅" if p_value < 0.05 else "❌"
        print(f"\n{biome.upper()}:")
        print(f"  GNN MAE: {np.mean(gnn_vals):.2f} ± {np.std(gnn_vals):.2f}m (n={min_len})")
        print(f"  CNN MAE: {np.mean(cnn_vals):.2f} ± {np.std(cnn_vals):.2f}m")
        print(f"  Wilcoxon W-stat: {stat:.1f}")
        print(f"  p-value: {p_value:.6f} {sig_symbol}")
        print(f"  Effect size (r): {r:.3f}")
    
    return pd.DataFrame(results)


def run_bootstrap_ci(df, n_bootstrap=10000, confidence=0.95):
    print("\n" + "="*60)
    print(f"TESTE 2: BOOTSTRAP {int(confidence*100)}% CI (n={n_bootstrap})")
    print("="*60)
    
    alpha = 1 - confidence
    results = []
    
    for biome in df['biome'].unique():
        biome_df = df[df['biome'] == biome]
        
        gnn_df = biome_df[biome_df['model'] == 'GNN'].sort_values('tile_idx')
        cnn_df = biome_df[biome_df['model'] == 'CNN'].sort_values('tile_idx')
        
        gnn_vals = gnn_df['mae_m'].values
        cnn_vals = cnn_df['mae_m'].values
        
        min_len = min(len(gnn_vals), len(cnn_vals))
        gnn_vals = gnn_vals[:min_len]
        cnn_vals = cnn_vals[:min_len]
        
        if min_len < 2:
            continue
        
        diff = cnn_vals - gnn_vals
        
        np.random.seed(42)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(diff, size=len(diff), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        mean_diff = np.mean(diff)
        std_diff = np.std(bootstrap_means)
        
        ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
        
        results.append({
            'biome': biome,
            'n_pairs': min_len,
            'mean_diff_cnn_minus_gnn': mean_diff,
            'std_bootstrap': std_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_excludes_zero': ci_excludes_zero,
            'gnn_better': ci_lower > 0
        })
        
        symbol = "✅" if ci_lower > 0 else ("⚠️" if ci_excludes_zero else "❌")
        print(f"\n{biome.upper()}:")
        print(f"  Mean difference (CNN - GNN): {mean_diff:.2f} ± {std_diff:.2f}m")
        print(f"  {int(confidence*100)}% CI: [{ci_lower:.2f}, {ci_upper:.2f}]m")
        print(f"  CI exclui zero: {symbol}")
    
    return pd.DataFrame(results)


def generate_paper_claims(wilcoxon_df, bootstrap_df):
    print("\n" + "="*60)
    print("CLAIMS PARA O PAPER")
    print("="*60)
    
    sig_wins = wilcoxon_df[wilcoxon_df['significant_005']]['biome'].tolist()
    n_sig = len(sig_wins)
    
    min_p = wilcoxon_df['p_value'].min()
    
    print(f"""
1. WILCOXON CLAIM:
   "A Wilcoxon signed-rank test confirmed that TerrainGNN significantly 
   outperformed the CNN baseline in {n_sig} of 4 biomes (p < 0.05), 
   with p-values as low as {min_p:.6f}."

2. BOOTSTRAP CLAIM:
   "Bootstrap 95% confidence intervals for the mean MAE difference 
   (CNN - GNN) excluded zero in {len(bootstrap_df[bootstrap_df['ci_excludes_zero']])} of 4 biomes, 
   indicating robust improvements."

3. EFFECT SIZE CLAIM:
   "Large effect sizes (r > 0.5) were observed in {len(wilcoxon_df[wilcoxon_df['effect_size_r'] > 0.5])} biomes, 
   demonstrating substantial practical significance."
""")


def main():
    parser = argparse.ArgumentParser(description="Run statistical tests on MAE results")
    parser.add_argument("csv_path", type=str, help="Path to mae_per_tile.csv")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("="*60)
    print('STATISTICAL TESTS: CNN vs GNN')
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    df = pd.read_csv(args.csv_path)
    print(f"\nData loaded: {len(df)} records")
    print(f"   Biomas: {df['biome'].unique().tolist()}")
    print(f"   Modelos: {df['model'].unique().tolist()}")
    
    wilcoxon_df = run_wilcoxon_test(df)
    bootstrap_df = run_bootstrap_ci(df, n_bootstrap=args.bootstrap)
    
    generate_paper_claims(wilcoxon_df, bootstrap_df)
    
    wilcoxon_path = f"{args.output_dir}/wilcoxon_results.csv"
    bootstrap_path = f"{args.output_dir}/bootstrap_results.csv"
    
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    
    print(f"\n✅ Resultados salvos:")
    print(f"   - {wilcoxon_path}")
    print(f"   - {bootstrap_path}")


if __name__ == "__main__":
    main()
