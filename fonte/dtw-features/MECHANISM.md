# Estudo mecanístico adicional

O runner `run-mechanism.sh` compila o mesmo conjunto `src/` e executa `dtw.kpz.MechanismStudy`. As classes do estudo ficam no pacote `dtw.kpz` para acessar os núcleos package-private de `DirectedPassage`. O estudo principal (`run.sh`) agora inclui DTW de custo máximo puro (`DTW_MAX`), preservando as execuções mínimas anteriores.

## Verificação e reanálise

```bash
cd fonte/dtw-features
./run-mechanism.sh --phase check
./run-mechanism.sh --phase reanalysis --out results/mechanism
```

`check` compara DTW mínimo, DTW_MAX, DIAG, caminho central, LPP3/FPP3, os extremos da recorrência interpolada e a tabela TW2 com as implementações existentes. Também verifica que o extremo `lambda=1` da interpolação máxima coincide com o custo máximo puro. A reanálise lê os CSV originais e grava `reanalysis.csv`; não tenta reconstruir a variância a partir de estatísticas agregadas que não a contêm.

## Sonda principal

```bash
./run-mechanism.sh --phase probe --n 64,128,256,512,1024,2048 \
  --replicates 5000 --pilot 2000 --seed 20260325 --seq-dist NORMAL
./run-mechanism.sh --phase probe --n 64,128,256,512,1024,2048 \
  --replicates 5000 --pilot 2000 --seed 20260325 --seq-dist UNIFORM
```

O journal retomável do mínimo fica em `results/mechanism/probe-{normal,uniform}-raw.csv`; uma assinatura de configuração impede retomar com parâmetros incompatíveis. O arquivo contém DTW, `DIAG`, `Gain=DIAG-OPT`, resíduo em relação à regressão OPT~DIAG do piloto, caminho, e controles LPP3/FPP3. O residual do piloto é gravado como `NaN` intencionalmente: o ajuste é estimado no piloto e aplicado às réplicas independentes de avaliação. Para as grandezas de custo, o resumo gera W1 após studentização com os parâmetros do piloto contra Normal padrão e TW2 refletida (sinal apropriado ao mínimo), além de momentos, escala local de desvio e associação OPT–DIAG. Para rugosidade/deslocamento do caminho, reporta momentos e escala; W1 contra Normal/TW2 fica `NaN`, pois essas leis são referências de custo, não de geometria. LPP3 registra também a rugosidade do caminho como calibração geométrica.

A execução padrão é deliberadamente limitada a `n<=2048`. A memória do caminho cresce como `n²` por tarefa; o custo de tempo total também cresce quadraticamente. Aumente `n` em etapas e confira memória/tempo antes de incluir 4096 ou maiores. As execuções são paralelizadas pelo common pool do Java; não há opção de controle `--threads` nesta versão.

A sonda máxima pode ser executada separadamente; usa arquivos e checkpoints próprios, e registra `DTW_MAX` na coluna `model`:

```bash
./run-mechanism.sh --phase maxprobe --n 128,256,512 --replicates 2000 --pilot 1000 --seq-dist NORMAL
```

As saídas são `max-probe-{normal,uniform}-raw.csv`, `max-probe-summary.csv`, `max-probe-scaling.csv` e `max-probe-association.csv`. As referências de forma do resumo usam TW2 não refletida, ao contrário da sonda mínima, mas isso é apenas uma comparação descritiva, não uma afirmação de lei limite.

## Interpolação e testes de ordem

```bash
./run-mechanism.sh --phase interpolation --n 128,256,512,1024 \
  --replicates 1000 --chi 0.40 --seq-dist NORMAL
./run-mechanism.sh --phase ranks --n 128,256,512,1024 --replicates 5000
./run-mechanism.sh --phase permutations --n 128,256,512,1024 --replicates 5000
```

A interpolação grava caminhos de custo mínimo e máximo, mistura de pesos exponenciais iid com `|X_i-Y_j|`, valores fixos de lambda e `u*n^(1/3-chi)` para `u={.25,.5,1,2,4}`. `--chi` deve ser substituído pela inclinação medida na sonda, por lei de sequência. `lambda=1` no modo mínimo coincide com DTW absoluto mínimo; `lambda=1` no modo máximo coincide com DTW_MAX absoluto puro. `lambda=0` no modo máximo é LPP3 e `lambda=0` no mínimo é FPP3. DTW_MAX mantém a vizinhança DTW (direita, acima e diagonal), maximizando a soma dos custos locais `|Xi−Yj|`; é o análogo de custo máximo puro do DTW mínimo, não a interpolação aleatória completa.

`ranks` aplica ranks às sequências contínuas iid e usa custo absoluto das diferenças de rank normalizadas. `permutations` gera duas permutações independentes uniformes e usa seus valores inteiros como sequências DTW. As variantes `ranks-max` e `permutations-max` gravam saídas separadas `ranks-max.csv` e `permutations-max.csv` com custo máximo. Estes observáveis são testes do ambiente separável; não implementam a inversa Sakai–Inenaga nem demonstram Plancherel/TW2.

## Saídas e limitações estatísticas

Todas as saídas novas ficam sob `results/mechanism/` (ou `--out`). `probe-summary.csv` traz estimativas pontuais, `probe-scaling.csv` inclinações locais e `probe-association.csv` regressão/associação; os equivalentes da sonda máxima usam prefixo `max-probe-`. Esta versão ainda não calcula intervalos bootstrap, gates inferenciais automáticos ou inclinações/assimetria/W1 resumidas para os arquivos de interpolação/ranks/permutação; esses CSVs são registros brutos para análise posterior. Portanto, não interpretar estimativas de fumaça ou sem IC como decisão de classe.
