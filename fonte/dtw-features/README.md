# dtw-features

Teste estatístico de equivalência da distribuição-limite do DTW clássico com a percolação de última passagem e teste de pertencimento à classe KPZ (expoente 1/3, geodésico 2/3, forma Tracy–Widom GUE).

Modelos simulados:

- **LPP**: última passagem clássica (passos direita/acima).
- **LPP3**: última passagem com três vizinhos (direita, acima e diagonal).
- **FPP3**: primeira passagem com diagonal (controle estrutural do DTW).
- **DTW**: DTW quadrado clássico, custo `|Xi−Yj|` ou `(Xi−Yj)²`.

Pesos de LPP/LPP3/FPP3 e as sequências do DTW usam uma distribuição escolhida na linha de comando: `EXPONENTIAL` (Exp(1)), `NORMAL` (N(0,1)), `POISSON` (Poisson(1)) ou `UNIFORM` (Uniforme[0,1]). O padrão é pesos exponenciais e sequências normais.

O controle de Johansson no relatório do LPP usa a escala teórica `(L−4n)/(2^(4/3) n^(1/3))`, válida para pesos Exp(1). Com outra `--weight-dist`, esse controle não deve ser lido como teste da lei limite.

## Compilação

```bash
cd fonte/dtw-features
./run.sh --check
```

`run.sh` compila com javac 17 e executa. `--check` valida a tabela Tracy–Widom e uma fumaça dos núcleos (LPP, LPP3, FPP3 e DTW).

## Execução

```bash
./run.sh --n 64,128,256 --replicates 800 --pilot 400 --bootstrap 199 \
  --weight-dist EXPONENTIAL --seq-dist NORMAL \
  --out results/kpz-equivalence.csv
```

Parâmetros recomendados para produção:

```
--n 64,128,256,512,1024 --replicates 5000 --pilot 2000 --bootstrap 399
```

Opções:

| Opção | Padrão | Função |
| --- | --- | --- |
| `--n` | `64,128,256` | grades (pelo menos 3, cada `n >= 8`) |
| `--replicates` | `400` | amostra de avaliação por n (`>= 30`) |
| `--pilot` | `200` | amostra independente para média/desvio (`>= 30`) |
| `--bootstrap` | `99` | réplicas bootstrap dos IC/UCB |
| `--alpha` | `0.01` | nível em `(0, 0.5)` |
| `--seed` | `20260325` | semente |
| `--threads` | `0` | `0` = todos os processadores |
| `--cost` | `abs` | custo local do DTW: `abs` ou `square` |
| `--weight-dist` | `EXPONENTIAL` | pesos de LPP, LPP3 e FPP3 |
| `--seq-dist` | `NORMAL` | distribuição das sequências do DTW |
| `--no-geodesic` | geodésico ligado | não estima o expoente transversal 2/3 |
| `--out` | `results/kpz-equivalence.csv` | CSV de saída (o `.txt` sai ao lado) |
| `--check` | — | valida TW e os núcleos e sai |

## Saídas

- CSV com todos os valores (modelo, n, chave, valor), inclusive `weight_dist` e `seq_dist`
- TXT com relatório textual e decisões de gatekeeping

## Decisões

- **KPZ**: expoente 1/3 no IC, forma TW2 não rejeitada, geodésico 2/3 (se pedido).
- **Equivalência**: UCB de W1(inf) < ε, onde ε = ½ min(W1(TW2*,Normal), W1(TW2*,TW1*)).

O bootstrap reamostra os vetores já studentizados (sem re-studentização), corrigindo o viés que fazia UCB < estimativa pontual.

## Dependências

Java 17, sem bibliotecas externas. O estudo conclusivo está em `KpzEquivalenceStudy`; o alvo de DTW/FPP3 é `-TW2`.
