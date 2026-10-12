#!/bin/bash
# Bateria de testes de corretude para GrafoPorAnomaliasAzureSpot

set -euo pipefail

SOURCE_DIR="/home/altamir/repos/sistdin-refatorado/ferramentas/resiliente/fonte"
OUTPUT_DIR="/tmp/testes-grafo"
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "BATERIA DE TESTES - GrafoPorAnomaliasAzureSpot"
echo "======================================"

# Função auxiliar para rodar teste e capturar saída
run_test() {
    local name="$1"
    shift
    echo "--- [TESTE] $name ---"
    if java -cp "$SOURCE_DIR" GrafoPorAnomaliasAzureSpot "$@" > "$OUTPUT_DIR/$name.log" 2>&1; then
        echo "PASSOU: $name"
        cat "$OUTPUT_DIR/$name.log"
        return 0
    else
        echo "FALHOU: $name"
        cat "$OUTPUT_DIR/$name.log"
        return 1
    fi
}

# Teste 1: Execução básica com argumentos mínimos válidos
# 2 rolls, 3 iters, radius=1, multipliers=1.0, 0, false, 50.0
# Precisa de arquivo de entrada com 5 valores (2 rolls + 3 orbits)
echo "Criando arquivo de entrada..."
cat > "$OUTPUT_DIR/input1.txt" <<EOF
0.1 0.2 0.3 0.4 0.5
EOF

echo "--- [TESTE 1] Execução básica com parâmetros mínimos ---"
if java -cp "$SOURCE_DIR" GrafoPorAnomaliasAzureSpot \
    --no-azure \
    --checkpoint="$OUTPUT_DIR/checkpoint1.checkpoint" \
    2 3 1 1.0 1.0 0 false 50.0 < "$OUTPUT_DIR/input1.txt" \
    > "$OUTPUT_DIR/test1.log" 2>&1; then
    echo "PASSOU: Teste 1 (Execução básica)"
else
    echo "FALHOU: Teste 1 (Execução básica)"
    cat "$OUTPUT_DIR/test1.log"
fi

# Teste 2: Argumentos insuficientes
echo "--- [TESTE 2] Argumentos insuficientes ---"
if java -cp "$SOURCE_DIR" GrafoPorAnomaliasAzureSpot \
    --no-azure \
    1 1 2>&1 | head -n 5 > "$OUTPUT_DIR/test2.log"; then
    echo "FALHOU: Teste 2 (deveria falhar)"
else
    echo "PASSOU: Teste 2 (Argumentos insuficientes rejeitados)"
fi
cat "$OUTPUT_DIR/test2.log"

# Teste 3: Checkpoint salvo
echo "--- [TESTE 3] Existência de arquivo de checkpoint ---"
if [ -f "$OUTPUT_DIR/checkpoint1.checkpoint" ]; then
    echo "PASSOU: Teste 3 (Checkpoint salvo)"
else
    echo "FALHOU: Teste 3 (Checkpoint não encontrado)"
fi

# Teste 4: Parâmetros inválidos (percentil fora do intervalo)
echo "--- [TESTE 4] Percentil fora do intervalo ---"
if java -cp "$SOURCE_DIR" GrafoPorAnomaliasAzureSpot \
    --no-azure \
    2 3 1 1.0 1.0 0 false 150.0 < "$OUTPUT_DIR/input1.txt" \
    > "$OUTPUT_DIR/test4.log" 2>&1; then
    echo "FALHOU: Teste 4 (deveria falhar com percentil 150)"
else
    echo "PASSOU: Teste 4 (Percentil inválido rejeitado)"
    cat "$OUTPUT_DIR/test4.log"
fi

# Teste 5: Execução com --checkpoint existente (resume)
echo "--- [TESTE 5] Resume de checkpoint ---"
# Copiar checkpoint existente como se fosse um checkpoint anterior
cp "$OUTPUT_DIR/checkpoint1.checkpoint" "$OUTPUT_DIR/checkpoint5.checkpoint"
if java -cp "$SOURCE_DIR" GrafoPorAnomaliasAzureSpot \
    --no-azure \
    --checkpoint="$OUTPUT_DIR/checkpoint5.checkpoint" \
    < /dev/null > "$OUTPUT_DIR/test5.log" 2>&1; then
    echo "PASSOU: Teste 5 (Resume de checkpoint)"
else
    echo "FALHOU: Teste 5 (Resume de checkpoint)"
fi
cat "$OUTPUT_DIR/test5.log"

echo "======================================"
echo "RESULTADOS FINAIS"
echo "======================================"
ls -la "$OUTPUT_DIR/"
echo "Todos os testes executados. Verifique logs acima."
