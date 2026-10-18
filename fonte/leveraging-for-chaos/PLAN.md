PLANO EXPERIMENTAL PARA AVALIAR A APLICAÇÃO DE CONCEITOS TOPOLÓGICOS AO RANK SELECTION PERSISTENCE

Este plano tem como objetivo avaliar se o Rank Selection Persistence, além de medir a frequência com que posições ordenadas são selecionadas, identifica estruturas que possam ser interpretadas por conceitos da topologia. A investigação será concentrada em componentes conexas, conjuntos de supernível, filtragens, árvores de fusão e homologia persistente de dimensão zero.

OBJETIVO GERAL

Avaliar se o perfil de persistência da seleção de ranks representa características topológicas estáveis das regiões de alta densidade local presentes em amostras ordenadas.

OBJETIVOS ESPECÍFICOS

O experimento deverá:

a) verificar se regiões de alta persistência de seleção correspondem a componentes conexas de alta densidade;

b) determinar se clusters unimodais, bimodais e anômalos produzem assinaturas topológicas diferentes;

c) comparar a persistência de seleção com a persistência topológica;

d) avaliar a estabilidade das estruturas detectadas diante de ruído, contaminação e variações dos parâmetros;

e) investigar o papel de K como parâmetro de escala;

f) avaliar se a razão entre a quantidade esperada de observações do cluster e o número de posições selecionadas determina uma fronteira de detectabilidade;

g) identificar as condições em que o método detecta e localiza corretamente um cluster.

QUESTÃO DE PESQUISA

A questão principal será:

Em que condições o perfil de Rank Selection Persistence representa componentes topológicas estáveis da função de densidade local definida sobre as posições ordenadas?

As questões complementares serão:

A seleção Top-K identifica componentes conexas associadas a regiões de alta densidade?

O perfil de persistência de seleção concorda com medidas obtidas por homologia persistente?

A interpretação topológica permanece válida quando epsilon, K, o tamanho amostral e a intensidade do ruído são alterados?

Existe uma razão de escala capaz de explicar a transição entre baixa e alta detectabilidade?

HIPÓTESE CENTRAL

A hipótese central é que o Rank Selection Persistence constitui uma estatística de ensemble relacionada às componentes conexas dos conjuntos de supernível da densidade local no domínio ordenado dos ranks.

O método não será inicialmente tratado como um invariante topológico. A hipótese a ser examinada é que o perfil estatístico produzido pelo método contém informação sobre estruturas topológicas subjacentes.

CONSTRUÇÃO DO DOMÍNIO TOPOLÓGICO

Para cada realização r, os dados serão ordenados:

x_r,(1) <= x_r,(2) <= ... <= x_r,(m).

O conjunto das posições ordenadas será:

V = {1, 2, ..., m}.

Sobre esse conjunto será construído um grafo de caminho:

G_m = (V, E),

em que:

E = {(i, i+1): i = 1, ..., m-1}.

Duas posições serão consideradas adjacentes quando forem consecutivas na ordenação. Essa construção transforma a sequência de ranks em um espaço discreto dotado de conectividade.

A densidade local será calculada por:

D_r(i) = soma, para j de 1 até m, de I(|x_r,(j) - x_r,(i)| <= epsilon).

A função D_r será tratada como uma função escalar definida nos vértices do grafo.

CONJUNTOS DE SUPERNÍVEL

Para cada limiar lambda, será construído o conjunto:

L_r(lambda) = {i em V: D_r(i) >= lambda}.

O subgrafo correspondente incluirá os vértices pertencentes a L_r(lambda) e as arestas entre posições consecutivas que também pertençam ao conjunto.

Quando lambda diminui, novos vértices são incorporados. Assim, obtém-se uma sequência aninhada:

L_r(lambda_1) contido em L_r(lambda_2), quando lambda_1 > lambda_2.

Essa sequência será interpretada como uma filtração de supernível da densidade local.

INVARIANTE TOPOLÓGICO PRINCIPAL

Como o domínio é um grafo de caminho, o principal invariante será a homologia de dimensão zero.

A quantidade:

beta_0(lambda)

representará o número de componentes conexas existentes no conjunto de supernível no limiar lambda.

Não são esperados ciclos topológicos não triviais no grafo de caminho. Por isso, a análise principal será baseada em H0, e não em H1.

Para cada componente serão registrados:

a) o limiar de nascimento;

b) o limiar de fusão com uma componente mais antiga;

c) a persistência da componente;

d) os ranks pertencentes à componente;

e) a largura da componente no domínio dos ranks;

f) a posição do máximo de densidade da componente.

A persistência de uma componente C será definida por:

persistência(C) = lambda_nascimento(C) - lambda_fusão(C).

RELAÇÃO ENTRE TOP-K E A FILTRAÇÃO

Na ausência de empates, a seleção dos K maiores valores de densidade local pode ser escrita como um conjunto de supernível definido por um limiar adaptativo.

Assim:

S_r(K) = {i: D_r(i) >= lambda_r,K}.

O Top-K original pode, portanto, ser interpretado como uma seção específica da filtração.

A análise topológica ampliará essa construção ao considerar vários valores de K ou todos os limiares distintos de densidade local.

Será avaliada a sequência:

S_r(1) contido em S_r(2) contido em ... contido em S_r(m).

Essa sequência permitirá observar o crescimento e a fusão das regiões selecionadas.

CENÁRIOS DE SIMULAÇÃO

Cenário 1: distribuição uniforme

Será utilizada uma distribuição uniforme no intervalo de zero a um.

Esse cenário funcionará como controle negativo. Não existe uma região interna naturalmente mais densa.

Espera-se um perfil relativamente plano, com possíveis efeitos de borda e componentes de curta persistência.

Cenário 2: distribuição normal unimodal

Será utilizada uma distribuição normal padrão.

Espera-se uma região central de alta densidade, uma componente dominante e um perfil de persistência aproximadamente unimodal.

Cenário 3: mistura bimodal bem separada

Será utilizada uma mistura de duas distribuições normais com médias suficientemente afastadas.

Espera-se o aparecimento de duas componentes de alta densidade, que deverão permanecer separadas por uma faixa de limiares antes de se fundirem.

Cenário 4: mistura bimodal parcialmente sobreposta

Será utilizada uma mistura de duas distribuições normais, variando-se progressivamente a distância entre as médias ou o desvio-padrão.

O objetivo será observar a transição entre uma estrutura bimodal e uma estrutura efetivamente unimodal.

Cenário 5: mistura com pesos desiguais

Será utilizada uma mistura com um componente principal e um componente secundário de menor peso.

Esse cenário permitirá avaliar se o Top-K fixo elimina a componente menor e se a análise multiescala consegue recuperá-la.

Cenário 6: cluster anômalo localizado

Uma fração p da amostra será gerada por uma distribuição estreita localizada em uma região específica da distribuição de referência.

Serão variados:

a) a fração anômala p;

b) o tamanho amostral m;

c) o número K de ranks selecionados;

d) o desvio-padrão sigma do cluster;

e) a posição do cluster no domínio dos quantis.

Cenário 7: distribuição principal com outliers dispersos

Uma pequena fração das observações será gerada longe da distribuição principal, mas sem formar um cluster concentrado.

Esse cenário permitirá distinguir um cluster denso de um conjunto de observações extremas isoladas.

PARÂMETROS EXPERIMENTAIS

Serão utilizados diferentes valores de tamanho amostral, por exemplo:

m em {128, 256, 512, 1024, 2048}.

O número de ranks selecionados poderá assumir:

K em {5, 10, 20, 40, 80}.

A fração anômala poderá assumir:

p em {0; 0,0025; 0,005; 0,01; 0,02; 0,05; 0,10}.

O desvio-padrão do cluster poderá assumir valores entre uma configuração muito concentrada e uma configuração difusa.

O número de repetições Monte Carlo deverá ser suficientemente grande para estimar potência, erro de localização e variabilidade. Recomenda-se começar com 100 repetições por configuração e aumentar esse número nas regiões próximas à transição de detectabilidade.

RAZÃO DE ESCALA

Será definida a razão:

rho = p vezes m dividido por K.

O numerador p vezes m representa o número esperado de observações pertencentes ao cluster anômalo.

O denominador K representa o orçamento de seleção, isto é, o número de ranks selecionados em cada realização.

A hipótese será que a detectabilidade depende fortemente de rho.

Serão investigados três regimes:

a) rho muito menor que 1, correspondente a baixa detectabilidade;

b) rho aproximadamente entre 0,1 e 1, correspondente à região de transição;

c) rho maior que 1, correspondente à alta detectabilidade.

A letra R não deverá ser usada para essa razão, pois R já representa o número de realizações do ensemble.

PROCEDIMENTO POR REALIZAÇÃO

Para cada configuração experimental:

Gerar uma amostra com m observações.

Ordenar a amostra.

Calcular a densidade local para cada rank.

Construir o grafo de caminho dos ranks.

Selecionar os K ranks de maior densidade local.

Registrar a seleção na matriz A.

Construir os conjuntos de supernível para todos os valores distintos de densidade.

Calcular as componentes conexas em cada nível.

Registrar o nascimento e a fusão das componentes.

Calcular a persistência de cada componente.

Registrar a posição, a largura e a massa de cada componente.

Repetir o procedimento ao longo do ensemble.

Construir o perfil de Rank Selection Persistence.

Agregar os descritores topológicos entre as realizações.

PERFIL DE PERSISTÊNCIA DE SELEÇÃO

Para um valor fixo de K, será calculado:

Pi_hat_K(i) = 1/R vezes a soma, para r de 1 até R, de I(i pertence a S_r(K)).

Esse perfil estima a probabilidade de o rank i pertencer ao Top-K.

O resultado será interpretado como uma medida de persistência de seleção, e não como uma estimativa clássica de densidade.

SUPERFÍCIE DE PERSISTÊNCIA POR RANK E K

Para reduzir a dependência de um único valor de K, será calculada:

Pi_hat(i, K) = 1/R vezes a soma, para r de 1 até R, de I(i pertence a S_r(K)).

O resultado será uma superfície definida por rank e K.

Essa superfície permitirá observar:

a) quando cada rank entra na seleção;

b) por quanto tempo o rank permanece selecionado;

c) como regiões inicialmente separadas crescem;

d) quando duas regiões se fundem;

e) como a escolha de K modifica a largura da região persistente.

PERFIL TOPOLÓGICO POR RANK

Para cada rank i e realização r, será atribuída uma pontuação topológica:

T_r(i) = maior persistência entre as componentes que contêm o rank i.

O perfil topológico agregado será:

T_hat(i) = 1/R vezes a soma, para r de 1 até R, de T_r(i).

Esse perfil será comparado com Pi_hat_K(i).

Pi_hat_K(i) mede a frequência de seleção.

T_hat(i) mede a intensidade da evidência topológica associada à posição.

As duas medidas serão tratadas como complementares.

TRATAMENTO DOS EMPATES

Como a densidade local assume valores inteiros, poderão ocorrer platôs e empates extensos.

Serão comparadas três estratégias.

Estratégia 1: desempate determinístico por rank.

Estratégia 2: inclusão de todos os ranks empatados no limiar.

Estratégia 3: atribuição de peso fracionário aos ranks empatados.

Na estratégia fracionária, quando existem t ranks empatados e faltam q posições para completar o Top-K, cada rank empatado recebe peso q/t.

Essa estratégia preserva a soma total de seleção igual a K e reduz o efeito de uma decisão arbitrária.

Na análise topológica, os ranks empatados serão incorporados simultaneamente no mesmo nível da filtração.

MÉTRICAS DE AVALIAÇÃO

17.1 Potência de detecção

A potência será definida como a proporção de realizações em que a hipótese nula é rejeitada no nível de significância estabelecido.

A potência será estudada em função de p, m, K, sigma e rho.

17.2 Controle do erro de tipo I

Sob p igual a zero, será verificado se a taxa de rejeição permanece próxima do nível nominal.

17.3 Erro de localização

Será calculada a distância entre o quantil estimado do pico e o quantil verdadeiro do cluster.

Uma medida possível será:

erro_localização = valor absoluto de quantil_estimado menos quantil_verdadeiro.

17.4 Largura da região detectada

Será registrada a largura da componente dominante em número de ranks e como proporção de m.

17.5 Número de componentes recuperadas

O número de componentes topológicas detectadas será comparado com o número conhecido de regiões densas no modelo gerador.

Serão registradas:

a) recuperações corretas;

b) subestimações;

c) superestimações;

d) componentes espúrias.

17.6 Concordância entre os perfis

Será calculada a correlação de Spearman entre:

Pi_hat_K(i)

e

T_hat(i).

Uma correlação elevada indicará que ranks frequentemente selecionados também pertencem a componentes topologicamente persistentes.

17.7 Sobreposição com a região verdadeira

Quando a região verdadeira do cluster for conhecida, será calculado o índice de Jaccard entre a região estimada e a região de referência.

17.8 Estabilidade sob perturbação

Cada amostra será perturbada por ruído de pequena magnitude.

O perfil original e o perfil perturbado serão comparados por:

Delta_Pi = 1/m vezes a soma dos valores absolutos das diferenças entre os perfis.

Também serão comparados os diagramas de persistência das duas amostras.

17.9 Sensibilidade a K

Perfis produzidos por valores próximos de K serão comparados.

Essa análise permitirá verificar se as conclusões dependem excessivamente de uma escolha específica, como os 10 por cento superiores.

17.10 Divergência em relação à distribuição nula

A divergência de Jensen-Shannon poderá ser utilizada para comparar o perfil observado com o perfil de referência obtido sob a hipótese nula.

A interpretação deverá ser relativa à distribuição nula, pois a divergência média sob ausência de cluster não é necessariamente zero.

HIPÓTESES EXPERIMENTAIS

Hipótese 1: distribuições unimodais produzirão uma única componente dominante.

Hipótese 2: misturas bem separadas produzirão duas componentes persistentes.

Hipótese 3: a persistência das componentes diminuirá à medida que os modos se aproximarem ou se tornarem mais dispersos.

Hipótese 4: o aumento de m permitirá detectar frações anômalas menores.

Hipótese 5: a potência crescerá com a razão rho.

Hipótese 6: clusters mais concentrados apresentarão maior potência e menor erro de localização.

Hipótese 7: K funcionará como parâmetro de escala.

Hipótese 8: um K muito pequeno poderá representar apenas o núcleo do cluster.

Hipótese 9: um K muito grande poderá expandir excessivamente a região selecionada e provocar fusões com o fundo.

Hipótese 10: a análise de vários valores de K será mais informativa do que a utilização isolada de K igual a 10 por cento de m.

Hipótese 11: a persistência de seleção e a persistência topológica estarão relacionadas, mas não serão equivalentes.

MODELO ESTATÍSTICO PARA A POTÊNCIA

A potência poderá ser modelada por regressão logística.

Uma formulação inicial será:

logit(Potência) = beta_0 + beta_1 log(rho) + beta_2 log(K) + beta_3 sigma + beta_4 log(m) + interações.

O modelo permitirá avaliar se a razão rho explica a transição de detectabilidade depois de controlar os efeitos de K, m e sigma.

Também será avaliado se uma curva sigmoide baseada apenas em rho fornece um colapso de escala adequado.

RESULTADOS ESPERADOS

Na distribuição uniforme, espera-se baixa estabilidade espacial das posições selecionadas e ausência de componentes internas persistentes.

Na distribuição normal, espera-se uma componente central dominante e um perfil aproximadamente unimodal.

Na mistura bimodal bem separada, esperam-se duas componentes persistentes que se fundem quando o limiar diminui.

Na mistura parcialmente sobreposta, espera-se redução gradual da persistência da componente secundária.

No cluster anômalo concentrado, espera-se elevado pico de persistência na região verdadeira, baixa largura e pequeno erro de localização.

No cluster anômalo difuso, espera-se redução da potência, aumento da largura e maior erro de localização.

Nos outliers dispersos, espera-se baixa persistência topológica, mesmo quando existirem observações extremas.

CRITÉRIOS PARA SUSTENTAR A INTERPRETAÇÃO TOPOLÓGICA

A interpretação topológica será considerada sustentada se forem observados simultaneamente:

a) correspondência entre componentes persistentes e regiões densas conhecidas;

b) estabilidade diante de pequenas perturbações;

c) concordância entre regiões de alta persistência de seleção e componentes de longa persistência topológica;

d) comportamento coerente na transição entre unimodalidade e multimodalidade;

e) capacidade de distinguir clusters concentrados de outliers dispersos;

f) vantagem da análise multiescala em relação ao uso de um único K;

g) resultados que não possam ser explicados apenas pela centralidade dos ranks ou por efeitos de borda.

DISTINÇÕES CONCEITUAIS

O estudo deverá distinguir três noções.

Persistência de seleção é a frequência com que um rank aparece no Top-K ao longo do ensemble.

Persistência topológica é o intervalo de limiares durante o qual uma componente permanece identificável antes de se fundir com outra componente.

Estabilidade estatística é a resistência do resultado a novas amostras, ruído e perturbações.

As três noções podem estar relacionadas, mas não são equivalentes.

VISUALIZAÇÕES

Para cada cenário, deverão ser apresentados:

a) a distribuição geradora;

b) uma amostra ordenada;

c) a densidade local por rank;

d) o grafo de caminho ponderado pela densidade;

e) os conjuntos de supernível em diferentes limiares;

f) o número de componentes por limiar;

g) a árvore de fusão;

h) o barcode ou diagrama de persistência de H0;

i) o perfil Pi_hat_K(i);

j) a superfície Pi_hat(i, K);

k) o perfil topológico T_hat(i);

l) a potência em função de p;

m) a potência em função de rho;

n) o erro de localização;

o) a largura da componente dominante;

p) a potência em função de sigma.

PRODUTO FINAL

O experimento deverá produzir:

a) uma definição formal do domínio topológico dos ranks;

b) uma descrição da filtração de supernível;

c) um algoritmo para acompanhar componentes conexas;

d) uma comparação entre persistência de seleção e persistência topológica;

e) uma análise da fronteira de detectabilidade;

f) uma avaliação da estabilidade do método;

g) recomendações para a escolha de K e epsilon;

h) uma delimitação clara das condições em que o método admite interpretação topológica.

CONCLUSÃO ESPERADA

A conclusão a ser testada é que o Rank Selection Persistence não constitui diretamente uma estimativa de densidade nem um invariante topológico. O método constitui uma estatística de ensemble sobre posições ordenadas.

Quando as posições são equipadas com uma relação de adjacência e a densidade local é analisada por conjuntos de supernível, o método pode ser relacionado a componentes conexas e à homologia persistente de dimensão zero.

A principal contribuição potencial será mostrar que a frequência de seleção, a escala K e a persistência das componentes oferecem descrições complementares da estabilidade e da geometria das regiões de alta densidade.

EXPLICAÇÃO DAS MELHORIAS

O plano foi reorganizado em uma sequência experimental completa: definição do domínio, construção topológica, cenários de simulação, parâmetros, métricas, hipóteses e critérios de validação. A redação também separa claramente persistência de seleção, persistência topológica e estabilidade estatística, evitando que conceitos distintos sejam tratados como equivalentes.