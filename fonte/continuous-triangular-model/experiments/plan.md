# Plano Experimental: CTM em ECG

## Objetivo
Testar se a representação Continuous Triangular Model (CTM), que representa intervalos da série em um domínio triangular multiescala, discrimina arritmias cardíacas e se operações sobre a superfície CTM acrescentam informação além de tratar a matriz como um vetor de características.

## Hipóteses
- **H1 principal:** CTM apresenta desempenho superior ou não inferior a ECG bruto, Wavelet e STFT na classificação binária normal vs arritmia, usando o mesmo classificador e divisão por registro/paciente.
- **H1 secundária:** perfis por escala e operações espaciais sobre a superfície CTM apresentam capacidade discriminativa adicional em relação ao vetor CTM achatado.
- **H0:** não há diferença estatisticamente significativa entre as representações.
- Detecção precoce e interpretação fisiológica são extensões futuras, não conclusões deste primeiro estudo.

## Decisões metodológicas
- Dataset: MIT-BIH Arrhythmia Database, inicialmente.
- Unidade de avaliação: registro/paciente; janelas de um mesmo grupo nunca podem aparecer em treino e teste.
- Tarefa inicial: classificação binária normal vs arritmia, com regra de rótulo explicitada e auditável.
- Classificador controlado: mesmo SVM RBF para todas as representações, com padronização e busca de hiperparâmetros dentro do treino.
- Mesmas janelas, rótulos, grupos e folds para bruto, Wavelet, STFT e CTM.
- A representação CTM científica será matriz + máscara + metadados de intervalos; PNG será apenas visualização.

## Fases
1. **Dados e protocolo:** fixar registros/seeds; carregar anotações MIT-BIH; definir canal, duração da janela, frequência, normalização, exclusão de bordas e mapeamento de símbolos normal/arritmia.
2. **Definição matemática CTM:** explicitar domínio contínuo dos intervalos `[t_start, t_end]`, coordenadas/eixos, escalas e discretização. Escolher a medida primária (integral assinada, média ou energia `|x(t)|^2`) antes de avaliar; variantes ficam como ablações. Distinguir “todos os intervalos contínuos” da grade finita efetivamente computada.
3. **Superfície CTM:** implementar matriz triangular, máscara e metadados temporais sem dependência de imagem; preservar visualização global simultânea de escalas e intervalos.
4. **Comparação discriminativa:** extrair ECG bruto, Wavelet, STFT e CTM das mesmas janelas; executar nested grouped CV com SVM comum. Transformações aprendidas devem ser ajustadas exclusivamente nos dados de treino.
5. **Análise espacial própria do CTM:** comparar CTM achatado com perfis por escala, energia/amplitude por escala, centro de massa, entropia, concentração, gradientes/regiões de alta contribuição, diferenças entre mapas médios por classe e distâncias entre superfícies.
6. **Métricas e estatística:** AUC-ROC, AUC-PR, sensibilidade, especificidade, F1 e balanced accuracy; intervalos de confiança por bootstrap no nível do paciente/registro; comparações pareadas e correção para múltiplos testes. Reportar tempo de extração/treino e dimensões.
7. **Ablações:** integral vs energia, média vs soma, escalas curtas/longas, `span`, normalização e operações espaciais. Verificar se o ganho decorre da estrutura de intervalos ou somente de suavização.
8. **Relato reproduzível:** tabelas, curvas, superfícies com eixos em tempo, mapas médios por classe, perfis por escala, parâmetros, seeds, contagens e limitações.

## Estado implementado
- Ambiente `.venv` e dependências em `requirements.txt`.
- `experiments/data.py`: loader WFDB e segmentação centrada em anotações; smoke test do registro 100 retornou 2.271 janelas.
- `experiments/representations.py`: bruto, Wavelet, STFT, CTM amplitude/energia.
- `experiments/evaluate.py`: nested grouped CV com SVM RBF e métricas por fold.
- `experiments/run_comparison.py`: CLI que salva resultados CSV.
- `experiments/tests/test_pipeline.py`: testes básicos passam; pipeline sintético completo validado.
- Ainda não foi executada a avaliação científica completa em todos os registros; resultados sintéticos não contam como evidência científica.

## Arquivos previstos/principais
- `tools/ctm_plot_image_fixed.py`: cálculo CTM atual e renderização.
- `experiments/data.py`, `representations.py`, `evaluate.py`, `run_comparison.py`.
- `experiments/ctm_surface.py`: próximo módulo para superfície e operações espaciais.
- `experiments/tests/`.
- `experiments/results/` para artefatos gerados.

## Critérios de validade
- Nenhum grupo compartilhado entre treino e teste.
- Quatro representações usam amostras, alvos e folds idênticos.
- Nenhum pré-processamento ajustado com dados do fold de teste.
- Rótulos, exclusões e número de janelas auditáveis.
- Superioridade/não inferioridade só declarada com incerteza e teste estatístico apropriados; caso contrário, reportar resultado inconclusivo/negativo.

## Limites e riscos
- MIT-BIH é desbalanceado; acurácia não será métrica única.
- Registro usado como proxy de paciente deve ser justificado com a estrutura da base.
- A interpretação de “energia” não deve ser afirmada para integral assinada.
- A grade triangular de resolução finita é uma discretização, não enumeração literal do contínuo.
- Não alegar detecção precoce nem mecanismo fisiológico sem experimento dedicado.
- CNNs, validação externa e predição pré-evento ficam fora do escopo inicial.
