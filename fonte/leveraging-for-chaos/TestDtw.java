import java.util.Arrays;

public class TestDtw {
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
            System.out.println("[OK] " + nome + " = " + obtido);
            testsPassados++;
        } else {
            System.out.println("[FAIL] " + nome + " - esperado: " + esperado + ", obtido: " + obtido);
            testsFalhados++;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Testes do algoritmo DTW (janela deslizante) ===\n");

        // Teste 1: Series identicas -> distancia 0
        double[] s1 = {1.0, 2.0, 3.0};
        double d1 = dtwDistance(s1, 0, 0, 3);
        assertEquals("Series identicas", 0.0, d1, 1e-9);

        // Teste 2: Series deslocadas
        double[] s2 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double d2 = dtwDistance(s2, 0, 2, 3);
        assertEquals("Series deslocadas {1,2,3} vs {3,4,5}", 6.0, d2, 1e-9);

        // Teste 3: Series constantes diferentes
        double[] s3 = {0.0, 0.0, 0.0};
        double d3 = dtwDistance(s3, 0, 0, 3);
        assertEquals("Series constantes diferentes", 30.0, d3, 1e-9);

        // Teste 4: Comprimento 1
        double[] s5 = {5.0, 7.0};
        double d4 = dtwDistance(s5, 0, 1, 1);
        assertEquals("Comprimento 1: |5-7|", 2.0, d4, 1e-9);

        // Teste 5: Series com mesmo valor
        double[] s6 = {3.0, 3.0, 3.0, 3.0};
        double d5 = dtwDistance(s6, 0, 1, 3);
        assertEquals("Series com mesmo valor", 0.0, d5, 1e-9);

        // Teste 6: Series crescentes lineares
        double[] s7 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        double d6 = dtwDistance(s7, 0, 3, 3);
        assertEquals("Series crescentes lineares", 9.0, d6, 1e-9);

        // Teste 7: Padrao triangular identico
        double[] s8 = {0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0};
        double d7 = dtwDistance(s8, 0, 4, 5);
        assertEquals("Padrao triangular identico", 0.0, d7, 1e-9);

        // Teste 8: Simetria do DTW
        double[] s9 = {1.0, 3.0, 2.0, 5.0};
        double d8a = dtwDistance(s9, 0, 1, 3);
        double d8b = dtwDistance(s9, 1, 0, 3);
        assertEquals("Simetria do DTW", d8a, d8b, 1e-9);

        // Teste 9: Excecao para indices fora do range
        boolean lançouExcecao = false;
        try {
            dtwDistance(s8, 5, 5, 5);
        } catch (ArrayIndexOutOfBoundsException e) {
            lançouExcecao = true;
        }
        if (lançouExcecao) {
            System.out.println("[OK] Lanca excecao para indices fora do range (esperado)");
            testsPassados++;
        } else {
            System.out.println("[FAIL] Deveria lancar excecao para indices fora do range");
            testsFalhados++;
        }

        // Teste 10: Valores com ponto flutuante
        double[] s10 = {0.1, 0.2, 0.3, 0.4, 0.5};
        double d10 = dtwDistance(s10, 0, 2, 3);
        assertEquals("Valores com ponto flutuante", 0.6, d10, 1e-9);

        // Teste 11: Monotonicidade - quanto maior a diferenca, maior o custo
        double[] s11a = {0.0, 0.0, 0.0};
        double[] s11b = {1.0, 1.0, 1.0};
        double[] s11c = {2.0, 2.0, 2.0};
        double da = dtwDistance(s11a, 0, 0, 3);
        double db = dtwDistance(s11a, 0, 0, 3);  // reusando s11a
        // Comparar via copia
        double[] orig = {0.0, 0.0, 0.0};
        double[] diff1 = {1.0, 1.0, 1.0};
        double[] diff2 = {2.0, 2.0, 2.0};
        double cost1 = dtwDistance(orig, 0, 0, 3);
        double cost2 = dtwDistance(diff1, 0, 0, 3);
        if (cost1 < cost2) {
            System.out.println("[OK] Monotonicidade: maior diferenca = maior custo");
            testsPassados++;
        } else {
            System.out.println("[FAIL] Monotonicidade violada");
            testsFalhados++;
        }

        // Teste 12: Deslocamento temporal minimo
        double[] s12 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        // s12[0..2] = {1,2,3} vs s12[5..7] = {6,7,8}
        // Alinhamento direto: |1-6|+|2-7|+|3-8| = 5+5+5 = 15
        double d12 = dtwDistance(s12, 0, 5, 3);
        assertEquals("Deslocamento temporal", 15.0, d12, 1e-9);

        System.out.println("\n=== Resultado ===");
        System.out.println("Passados: " + testsPassados);
        System.out.println("Falhados: " + testsFalhados);

        if (testsFalhados == 0) {
            System.out.println("\n[SUCESSO] Algoritmo DTW esta correto!");
        } else {
            System.out.println("\n[FALHA] Ha problemas no algoritmo DTW.");
            System.exit(1);
        }
    }
}
