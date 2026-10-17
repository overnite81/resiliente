# Experimento discriminativo do CTM

Este pacote compara quatro representacoes da mesma janela de ECG usando o mesmo SVM:

- ECG bruto;
- coeficientes Wavelet;
- magnitude da STFT;
- celulas validas do CTM triangular.

A divisao usa grupos por registro. Padronizacao e busca de hiperparametros ficam dentro de cada fold para evitar leakage.

## Ambiente

Na raiz do projeto:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Testes locais

```bash
python -m unittest discover -s experiments/tests
```

## MIT-BIH via PhysioNet

O comando abaixo baixa os registros por WFDB quando eles ainda nao estao no cache local:

```bash
python -m experiments.run_comparison \
  --records 100 101 103 105 106 108 109 111 112 113 115 116 118 119 121 122 123 124 \
  --window-samples 360 \
  --ctm-span 32 \
  --output experiments/results/comparison.csv
```

Para dados ja baixados, use `--data-dir` apontando para a pasta que contem os arquivos WFDB.

Se quiser preencher esse diretório uma vez e depois reutilizá-lo offline, use o helper abaixo, que valida a presença de `.hea`, `.dat` e `.atr` para cada registro:

```bash
python -m experiments.download_mitdb \
  --records 100 101 103 105 \
  --data-dir data/mitdb
```

O primeiro estudo usa a medida `amplitude`. A variante experimental por energia pode ser executada com:

```bash
python -m experiments.run_comparison --ctm-measure energy
```

## Diagnostico de `Instrução ilegal` (SIGILL) no Raspberry Pi 4

No Raspberry Pi 4, execute primeiro o diagnostico abaixo antes de reinstalar qualquer pacote. Ele roda cada etapa em subprocessos separados para identificar qual biblioteca nativa mata o processo:

```bash
source .venv/bin/activate
python -m experiments.diagnose_env | tee experiments/results/diagnose_env.txt
python -m pip show numpy scipy PyWavelets pillow torch torchvision \
  | tee experiments/results/pip_show_env.txt
python -m pip freeze | tee experiments/results/pip_freeze.txt
```

Leitura rapida do resultado:

- falha em `scipy.signal.stft`: suspeite de `scipy`/BLAS;
- falha em `pywt.cwt`: suspeite de `PyWavelets`;
- falha em `torch import` ou `torch forward`: suspeite de `torch`/`torchvision`;
- se a string de versao do PyTorch/torchvision trouxer sufixo `+cu...` em uma instalacao CPU-only no Pi, trate isso como suspeito e reinstale a pilha correspondente para a propria plataforma ARM64.

Apos identificar o ponto de falha, reinstale primeiro apenas a pilha afetada e rode o diagnostico novamente:

```bash
python -m pip install --force-reinstall --no-cache-dir numpy scipy PyWavelets pillow
```

```bash
python -m pip uninstall -y torch torchvision
python -m pip install --no-cache-dir -r requirements-resnet.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -m experiments.diagnose_env
```

Somente depois de o diagnostico passar sem SIGILL vale repetir o piloto ResNet.

## Piloto de imagens + ResNet-18 (CPU)

O piloto mantém o estudo SVM acima intacto e adiciona uma comparação separada entre imagens STFT, escalograma CWT (Wavelet contínua) e CTM, todas redimensionadas para 96 × 96. A divisão é por registro: por padrão, treina nos registros 100, 101 e 103 e testa no registro 105. Para manter o piloto pequeno, limita a 64 exemplos por classe por registro e roda 2 épocas. As imagens são codificadas por execução, não gravadas em disco.

Use a `.venv` já existente na raiz do projeto. Ative-a antes de instalar PyTorch e torchvision na VM ARM64 (Neoverse N1); as dependências do piloto ficam separadas das do SVM:

```bash
source .venv/bin/activate
python -m pip install -r requirements-resnet.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Faça primeiro o smoke test das transformações e do CLI:

```bash
python -m unittest discover -s experiments/tests
python -m experiments.run_resnet_pilot --help
```

Execute o piloto pequeno (pesos aleatórios, sem download de pesos pré-treinados):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
python -m experiments.run_resnet_pilot \
  --records 100 101 103 105 --test-record 105 \
  --limit-per-class-record 64 --epochs 2 --batch-size 16 \
  --image-size 96 --ctm-span 48 --threads 2 \
  --output experiments/results/resnet_pilot.csv \
  --time-log experiments/results/resnet_pilot_timing.csv
```

O log separa tempo de codificação das imagens, épocas e treino/inferência. `--weights imagenet` habilita inicialização pré-treinada (baixa os pesos na primeira execução); registre essa escolha e compare todas as representações com a mesma configuração. O conjunto de teste contém somente um registro, portanto as métricas do piloto são diagnósticas, não uma estimativa robusta de generalização.

### Avaliação de representações + CNNs com 5 folds

O runner compara cada representação (Wavelet, STFT e CTM) com três backbones do torchvision: ResNet-18 (`resnet18`), ResNet-34 (`resnet34`) e EfficientNet-B0 (`efficientnet_b0`). Por padrão, executa todas as 9 combinações nos mesmos 5 folds `StratifiedGroupKFold`, agrupados por registro, usando os 18 registros listados, até 64 exemplos por classe/registro e 20 épocas. Nenhum registro aparece simultaneamente no treino e na validação do mesmo fold. Isso corresponde a 45 ajustes de modelo (900 épocas no total com os padrões), portanto o custo é aproximadamente três vezes o experimento anterior apenas com ResNet-18. Use `--models resnet34 efficientnet_b0` para executar somente os dois backbones novos. Prepare os dados antes e passe `--resume` após uma desalocação:

```bash
.venv/bin/python -m experiments.download_mitdb \
  --records 100 101 103 105 106 108 109 111 112 113 115 116 118 119 121 122 123 124 \
  --data-dir data/mitdb

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
.venv/bin/python -m experiments.run_resnet_cv \
  --data-dir data/mitdb --folds 5 --epochs 20 --threads 2 --seed 42
```

O runner grava métricas por fold em `experiments/results/cnn_cv5/cnn_cv5_folds.csv`, resumo por modelo e representação em `cnn_cv5_summary.csv` e tempos em `cnn_cv5_timing.csv`. O resumo calcula média, desvio-padrão amostral e IC bilateral de 95% usando a distribuição t de Student (`n=5`, 4 graus de liberdade). Checkpoints atômicos, separados por modelo, representação e fold, guardam modelo/otimizador/estado aleatório ao fim de cada época; os CSVs são atualizados ao fim de cada fold. Para retomar após desalocação, use `--resume` e mantenha os mesmos dados, modelos, parâmetros e seed. Guarde resultados e checkpoints em armazenamento persistente da VM; os arquivos no disco efêmero não sobrevivem à desalocação.

O IC entre folds é uma descrição da variação observada, não um intervalo inferencial perfeito: os conjuntos de treino dos folds se sobrepõem e, portanto, as métricas não são independentes. O protocolo usa 5 folds e uma seed; para inferência mais sólida, repita com outras seeds ou valide em registros externos. Pesos são aleatórios por padrão (`--weights none`); com `--weights imagenet`, cada backbone usa seus pesos ImageNet correspondentes. Use a mesma opção para todas as combinações e registre-a.

### Perfil longo para Azure Spot ARM64

Foi adicionado o perfil `spot-arm64-long`: 18 registros MIT-BIH (com 105 como teste), limite de 64 exemplos por classe/registro e 12 épocas por representação. Isso aumenta o conjunto de treino e o tempo computacional em relação ao piloto de 2 épocas. Execute na VM Spot ARM64, idealmente com os arquivos WFDB já no disco, para não gastar a janela de uptime com download:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
python -m experiments.run_resnet_pilot --profile spot-arm64-long \
  --data-dir data/mitdb --threads 2 --seed 42
```

O perfil grava resultados e tempos por representação assim que cada uma termina, em `resnet_spot_arm64_long.csv` e `resnet_spot_arm64_long_timing.csv`. Se a VM for desalocada, reinicie com os mesmos argumentos e `--resume`; representações concluídas serão ignoradas. A representação interrompida durante treino é reiniciada do começo (não há checkpoint de época). Sem `--resume`, os CSVs de destino são sobrescritos.

O CSV de tempos também registra o uptime Linux da VM no início do processo e ao fim de cada representação (`/proc/uptime`). Isso registra uptime observado enquanto o processo executa, mas não detecta por si só uma eviction abrupta. Para calcular uptime até desalocação, combine os logs preservados em armazenamento persistente com os eventos de desalocação do Azure (e registre também inícios de alocação sem resultado final, que são observações censuradas). Uma execução que termina normalmente fornece somente um limite inferior, não o instante da eviction. Também não há um uptime médio universal/garantido para Spot: ele varia com SKU, região, zona e capacidade disponível. Portanto, as 12 épocas são uma configuração de carga, não uma previsão de que o trabalho caberá numa janela típica. Depois das primeiras medições, estime a duração total usando os tempos por época e escolha `--epochs` deixando margem para carga dos dados, codificação, inferência e gravação. Não use a média como garantia de sobrevivência de uma alocação.

`--records`, `--epochs`, `--limit-per-class-record`, `--representations`, `--models` e os caminhos de saída podem ser sobrescritos para ajustar a carga. Ao retomar, mantenha a configuração e a seed originais; o modo `--resume` identifica trabalho concluído pelo modelo, representação e fold, mas não valida se os demais parâmetros coincidem.

A execução aleatória continua sendo validação do pipeline, **não** resultado científico. O teste em um único registro e as limitações de um único split exigem validação agrupada por registro e múltiplas seeds antes de qualquer conclusão sobre desempenho.

Se PhysioNet estiver inacessível, baixe os arquivos WFDB antecipadamente e passe `--data-dir` com o diretório local.

### Semântica das imagens

- **STFT:** log-magnitude da STFT.
- **Wavelet:** escalograma de magnitude da CWT Morlet, não o vetor de coeficientes DWT do estudo SVM.
- **CTM:** células válidas da grade triangular atual, com medida `amplitude` (integral assinada) por padrão; preenchimento fora da região triangular permanece zero. `energy` é variante exploratória.

As três representações são normalizadas por janela e redimensionadas com interpolação bilinear; isso é uma configuração de piloto e deve ser congelada/documentada antes da avaliação final.

Arquivos adicionados: `experiments/image_representations.py`, `experiments/run_resnet_pilot.py`, `experiments/tests/test_image_representations.py` e `requirements-resnet.txt`.

> **Limitação de avaliação:** uma única divisão hold-out por registro, poucas épocas e inicialização aleatória não sustentam alegações sobre superioridade do CTM. Após validar a execução e medir o custo no N1, o próximo protocolo deve usar mais registros/folds agrupados e várias sementes, com tratamento explícito do desbalanceamento.
