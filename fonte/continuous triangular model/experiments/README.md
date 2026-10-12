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

O log separa tempo de codificação das imagens, épocas e treino/inferência. Use o tempo medido por época para projetar um experimento maior; a execução aleatória é apenas validação do pipeline, **não** resultado científico. `--weights imagenet` habilita inicialização pré-treinada (baixa os pesos na primeira execução); registre essa escolha e compare todas as representações com a mesma configuração. O conjunto de teste contém somente um registro, portanto as métricas do piloto são diagnósticas, não uma estimativa robusta de generalização.

Se PhysioNet estiver inacessível, baixe os arquivos WFDB antecipadamente e passe `--data-dir` com o diretório local.

### Semântica das imagens

- **STFT:** log-magnitude da STFT.
- **Wavelet:** escalograma de magnitude da CWT Morlet, não o vetor de coeficientes DWT do estudo SVM.
- **CTM:** células válidas da grade triangular atual, com medida `amplitude` (integral assinada) por padrão; preenchimento fora da região triangular permanece zero. `energy` é variante exploratória.

As três representações são normalizadas por janela e redimensionadas com interpolação bilinear; isso é uma configuração de piloto e deve ser congelada/documentada antes da avaliação final.

Arquivos adicionados: `experiments/image_representations.py`, `experiments/run_resnet_pilot.py`, `experiments/tests/test_image_representations.py` e `requirements-resnet.txt`.

> **Limitação de avaliação:** uma única divisão hold-out por registro, poucas épocas e inicialização aleatória não sustentam alegações sobre superioridade do CTM. Após validar a execução e medir o custo no N1, o próximo protocolo deve usar mais registros/folds agrupados e várias sementes, com tratamento explícito do desbalanceamento.
