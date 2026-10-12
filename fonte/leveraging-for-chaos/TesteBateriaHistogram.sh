
#!/bin/bash
# Bateria de testes para HistogramCalculator (matriz n linhas × m colunas)
# Execute com: bash TesteBateriaHistogram.sh

set -uo pipefail
SOURCE="/home/altamir/repos/sistdin-refatorado/ferramentas/resiliente/fonte"
OUT="/tmp/testes-histogram"
mkdir -p "$OUT"
PASS=0; FAIL=0

run_test() {
    local name="$1"; shift
    local expected="$1"; shift
    local result
    result=$(java -cp "$SOURCE" HistogramCalculator "$@" 2>&1)
    if [ "$result" = "$expected" ]; then
        echo "✓ PASSOU: $name => $result"
        PASS=$((PASS+1))
    else
        echo "✗ FALHOU: $name => got='$result' expected='$expected'"
        FAIL=$((FAIL+1))
    fi
}

echo "======================================"
echo "BATERIA DE TESTES - HistogramCalculator"
echo "======================================"

# Teste 1: Execução básica (2 linhas x 4 colunas, left=1, right=3)
echo "1.0 3.0 2.0 4.0 5.0 7.0 6.0 8.0" > "$OUT/input1.txt"
run_test "ExecucaoBasica" "0,2,0,0" "$OUT/input1.txt" 2 4 0 10 0.5 1 3 0 2

# Teste 2: Matriz 3x3 com todos os ranks
echo "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0" > "$OUT/input2.txt"
run_test "Matriz3x3FullRank" "3,3,3" "$OUT/input2.txt" 3 3 0 20 5.0 0 3 0 3

# Teste 3: Valores duplicados (mesma coluna repetida)
echo "5.0 5.0 5.0 5.0" > "$OUT/input3.txt"
run_test "ValoresDuplicados" "2,2" "$OUT/input3.txt" 2 2 0 10 0.1 0 2 0 2

# Teste 4: Epsilon zero (contagem exata)
echo "1.0 2.0 3.0 4.0" > "$OUT/input4.txt"
run_test "EpsilonZero" "1,0,0,0" "$OUT/input4.txt" 1 4 0 10 0.0 0 4 0 1

# Teste 5: left=right (faixa de ranks vazia) -> nenhum incremento
echo "1.0 2.0 3.0 4.0" > "$OUT/input5.txt"
run_test "FaixaVazia" "0,0,0,0" "$OUT/input5.txt" 1 4 0 10 1.0 2 2 0 4

# Teste 6: Limites excludentes (valores abaixo do lower)
echo "0.5 1.5 2.5 3.5 4.5 5.5" > "$OUT/input6.txt"
run_test "LimitesExcludentes" "1,1,0" "$OUT/input6.txt" 2 3 1 6 0.5 1 3 0 2

# Teste 7: left=0, right=n (todos)
echo "3.0 1.0 2.0" > "$OUT/input7.txt"
run_test "LeftZeroRightN" "1,0,0" "$OUT/input7.txt" 1 3 0 10 1.0 0 3 0 1

# Teste 8: Argumentos insuficientes (deve falhar)
echo "--- [TESTE 8] Argumentos insuficientes ---"
if java -cp "$SOURCE" HistogramCalculator "$OUT/input1.txt" 2 4 0 10 0.5 1 > "$OUT/test8.log" 2>&1; then
    echo "✗ FALHOU: ArgumentosInsuficientes (deveria falhar)"
    FAIL=$((FAIL+1))
else
    echo "✓ PASSOU: ArgumentosInsuficientes"
    PASS=$((PASS+1))
fi

# Teste 9: Saída CSV válida (sem espaço)
echo "10.5 20.5 30.5 40.5 50.5" > "$OUT/input9.txt"
result=$(java -cp "$SOURCE" HistogramCalculator "$OUT/input9.txt" 1 5 10 60 5.0 2 4 0 1 2>&1)
if [ -n "$result" ] && [ "$result" != *" "* ]; then
    echo "✓ PASSOU: SaidaCSVValida => $result"
else
    echo "✗ FALHOU: SaidaCSVValida => got='$result'"
    FAIL=$((FAIL+1))
fi

# Teste 10: Matriz 1x1 (caso extremo)
echo "42.0" > "$OUT/input10.txt"
run_test "Matriz1x1" "1" "$OUT/input10.txt" 1 1 0 100 1.0 0 1 0 1

echo ""
echo "======================================"
echo "RESULTADO FINAL: $PASS passaram, $FAIL falharam"
echo "======================================"
if [ $FAIL -eq 0 ]; then
    echo "✓ TODOS OS TESTES PASSARAM"
else
    echo "✗ ALGUNS TESTES FALHARAM"
    exit 1
fi
