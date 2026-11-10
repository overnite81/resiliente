#!/usr/bin/env python3
"""
Analisa os arquivos de interpolação gerados pelo MechanismStudy.
Calcula chi_local do custo para cada valor de lambda e mode.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def compute_scaling(df, group_cols=['dist', 'mode', 'lambda']):
    """Calcula chi_local entre tamanhos consecutivos de n."""
    results = []
    
    for keys, group in df.groupby(group_cols):
        g = group.sort_values('n')
        ns = g['n'].unique()
        
        for i in range(len(ns) - 1):
            n1, n2 = ns[i], ns[i+1]
            data1 = g[g['n'] == n1]['cost']
            data2 = g[g['n'] == n2]['cost']
            
            if len(data1) < 2 or len(data2) < 2:
                continue
                
            sd1 = data1.std()
            sd2 = data2.std()
            
            if sd1 > 0 and sd2 > 0:
                chi = np.log(sd2 / sd1) / np.log(n2 / n1)
                var_over_n = (sd2 ** 2) / n2
            else:
                chi = np.nan
                var_over_n = np.nan
            
            results.append({
                **dict(zip(group_cols, keys)),
                'n1': n1,
                'n2': n2,
                'sd1': sd1,
                'sd2': sd2,
                'chi_local': chi,
                'var_over_n': var_over_n,
                'count1': len(data1),
                'count2': len(data2)
            })
    
    return pd.DataFrame(results)


def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_interpolation.py <arquivo.csv> [arquivo2.csv ...]")
        sys.exit(1)
    
    all_dfs = []
    for path in sys.argv[1:]:
        print(f"Lendo {path}...")
        df = pd.read_csv(path)
        all_dfs.append(df)
    
    data = pd.concat(all_dfs, ignore_index=True)
    
    print("\n=== Resumo dos dados ===")
    print(data.groupby(['dist', 'mode', 'lambda'])['cost'].agg(['mean', 'std', 'count']).head(20))
    
    print("\n=== Calculando chi_local por lambda ===")
    scaling = compute_scaling(data)
    
    # Ordena por lambda e mode para facilitar visualização
    scaling = scaling.sort_values(['dist', 'mode', 'lambda', 'n1'])
    
    print(scaling.to_string(index=False))
    
    # Salva resultado
    output = Path("interpolation-scaling.csv")
    scaling.to_csv(output, index=False)
    print(f"\nResultados salvos em: {output}")


if __name__ == "__main__":
    main()