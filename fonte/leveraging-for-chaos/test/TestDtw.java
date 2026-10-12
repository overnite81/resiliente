import java.util.Arrays;

public class TestDtw {
    // Cópia do método dtwDistance para teste isolado
    private static double dtwDistance(double[] values, int first, int second, int length) {
        double[][] dtw = new double[length + 1][length + 1];
        for (int i = 0; i < length; i++) Arrays.fill(dtw[i], Double.POSITIVE_INFINITY);
        dtw[0][0] = 0.0;
        for (int i = 1; i <= length; i++) {
            for (int j = 1; j <= length; j++) {
                double cost = Math.abs(values[first + i - 1] - values[second + j - 1]);
                dtw[i][j] = cost + Math.min(dtw[i - 1][j],
                    Math.min(dtw[i][j - 1], dtw[i - 1][j - 1]));
            }
        }
        return dtw[length][length];
    }

    private static int testsPassados = 0;
    private static int testsFalhados = 0;

    private static void assertEquals(String nome, double esperado, double obtido, double tolerancia) {
        if (Math.abs(esperado - obtido) <= tolerancia) {
            System.out.println("[OK] " + nome);
            testsPassados++;
        } else {
            System.out.println("[FAIL] " + nome + " - esperado: " + esperado + ", obtido: " + obtido);
            testsFalhados++;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Testes do algoritmo DTW (janela deslizante) ===\n");

        // Teste 1: Séries idênticas -> distância 0
        double[] s1 = {1.0, 2.0, 3.0};
        double d1 = dtwDistance(s1, 0, 0, 3);
        assertEquals("Séries idênticas", 0.0, d1, 1e-9);
        assertEquals("Séries idênticas (s1[0..2] vs s1[0..2])", 0.0, d1, 1e-9);

        // Teste 2: Séries deslocadas de 1 -> distância = 2 (cada elemento difere de 1, e há 3 elementos)
        double[] s2 = {1.0, 2.0, 3.0, 4.0, 5.0};
        // Compara s2[0..2] = {1,2,3} com s2[2..4] = {3,4,5}
        // Pode alinhar perfeitamente: 1->3, 2->4, 3->5 -> custo = |1-3|+|2-4|+|3-5| = 2+2+2 = 6
        double d2 = dtwDistance(s2, 0, 2, 3);
        assertEquals("Séries deslocadas {1,2,3} vs {3,4,5}", 6.0, d2, 1e-9);

        // Teste 3: Séries completamente diferentes
        double[] s3 = {0.0, 0.0, 0.0};
        double[] s4 = {10.0, 10.0, 10.0};
        // Compara {0,0,0} com {10,10,10} -> custo = 3 * 10 = 30
        double d3 = dtwDistance(s3, 0, 0, 3);
        assertEquals("Séries constantes diferentes", 30.0, d3, 1e-9);

        // Teste 4: Comprimento 1
        double[] s5 = {5.0, 7.0};
        double d4 = dtwDistance(s5, 0, 1, 1);
        assertEquals("Comprimento 1: |5-7|", 2.0, d4, 1e-9);

        // Teste 5: Séries com mesmo valor
        double[] s6 = {3.0, 3.0, 3.0, 3.0};
        double d5 = dtwDistance(s6, 0, 1, 3);
        assertEquals("Séries com mesmo valor", 0.0, d5, 1e-9);

        // Teste 6: Séries crescentes lineares
        double[] s7 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        // Compara {0,1,2} com {3,4,5}
        // Alinhamento direto: |0-3|+|1-4|+|2-5| = 3+3+3 = 9
        // Ou pode esticar: 0->3, 1->3, 1->4, 2->4, 2->5, etc.
        // DTW deve encontrar o menor custo
        double d6 = dtwDistance(s7, 0, 3, 3);
        assertEquals("Séries crescentes lineares", 9.0, d6, 1e-9);

        // Teste 7: Janela de tamanho 5
        double[] s8 = {0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0};
        // Mesma forma, deslocada -> distância 0
        double d7 = dtwDistance(s8, 0, 4, 5);
        assertEquals("Padrão triangular idêntico", 0.0, d7, 1e-9);

        // Teste 8: Simetria (DTW é simétrico em relação à troca de argumentos)
        double[] s9 = {1.0, 3.0, 2.0, 5.0};
        double d8a = dtwDistance(s9, 0, 1, 3);  // {1,3,2} vs {3,2,5}
        double d8b = dtwDistance(s9, 1, 0, 3);  // {3,2,5} vs {1,3,2}
        assertEquals("Simetria do DTW", d8a, d8b, 1e-9);

        // Teste 9: Verificar limites do loop
        // Se first+length-1 ou second+length-1 exceder o array, deve lançar exceção
        // (isso é esperado, não é um bug, mas documenta o comportamento)
        boolean lançouExcecao = false;
        try {
            dtwDistance(s8, 5, 5, 5);  // índices fora do range
        } catch (ArrayIndexOutOfBoundsException e) {
            lançouExcecao = true;
        }
        if (lançouExcecao) {
            System.out.println("[OK] Lança exceção para índices fora do range (esperado)");
            testsPassados++;
        } else {
            System.out.println("[FAIL] Deveria lançar exceção para índices fora do range");
            testsFalhados++;
        }

        // Teste 10: Valores com ponto flutuante
        double[] s10 = {0.1, 0.2, 0.3, 0.4, 0.5};
        double d10 = dtwDistance(s10, 0, 2, 3);
        // {0.1, 0.2, 0.3} vs {0.3, 0.4, 0.5}
        // Custo direto: 0.2 + 0.2 + 0.2 = 0.6
        assertEquals("Valores com ponto flutuante", 0.6, d10, 1e-9);

        System.out.println("\n=== Resultado ===");
        System.out.println("Passados: " + testsPassados);
        System.out.println("Falhados: " + testsFalhados);

        if (testsFalhados == 0) {
            System.out.println("\n[SUCESSO] Algoritmo DTW está correto!");
        } else {
            System.out.println("\n[FALHA] Há problemas no algoritmo DTW.");
            System.exit(1);
        }
    }
}