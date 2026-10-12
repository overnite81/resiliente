import java.io.*;
import java.util.regex.*;

/**
 * Testa especificamente se as arestas do grafo são criadas
 * APENAS onde o cálculo de percentil permite.
 */
public class BateriaPercentil2 {

    private static int executar(String[] args) throws Exception {
        java.util.List<String> cmd = new java.util.ArrayList<>();
        cmd.add("java"); cmd.add("-cp"); cmd.add("."); 
        cmd.add("GrafoPorAnomaliasAzureSpot"); cmd.add("--no-azure");
        for (String a : args) cmd.add(a);
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.directory(new File("."));
        pb.redirectInput(new File("input.txt"));
        Process p = pb.start();
        p.waitFor();
        if (p.exitValue() != 0) {
            System.err.println("Execucao falhou: " + p.exitValue());
            return -1;
        }
        String saida = new String(p.getInputStream().readAllBytes());
        return saida.split("<edge ", -1).length - 1;
    }

    public static void main(String[] args) throws Exception {
        // nIters=7, nRolls=2, radius=1 gera 3 pares de DTW
        // Sequência: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        // rolls = 0.1, 0.2
        // orbit = 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9");
        }

        System.out.println("=== TESTE: Arestas conforme Percentil ===\n");
        System.out.println("Configuracao: nIters=7, radius=1");
        System.out.println("Sequencia crescente: 0.3 0.4 0.5 0.6 0.7 0.8 0.9");
        System.out.println("Pares de DTW gerados (j >= i+2*radius+1 = i+3):");
        System.out.println("  (i=1, j=4): orbit[0..2] vs orbit[3..5] = 0.3,0.4,0.5 vs 0.6,0.7,0.8");
        System.out.println("  (i=2, j=5): orbit[1..3] vs orbit[4..6] = 0.4,0.5,0.6 vs 0.7,0.8,0.9");
        System.out.println("  (i=3, j=6): orbit[2..4] vs orbit[5..7] = 0.5,0.6,0.7 vs 0.8,0.9,... FALHA!");
        System.out.println("   ^^^ Espera, j=6 com nIters=7, j < nIters-radius=6 -> nao entra!");
        System.out.println("");
        System.out.println("Recontando pares:");
        System.out.println("  i=1, j in [4, 6): j=4, j=5 -> 2 pares");
        System.out.println("  i=2, j in [5, 6): j=5 -> 1 par");
        System.out.println("  i=3, j in [6, 6): vazio");
        System.out.println("Total: 3 pares? Nao, 2+1=3 pares na verdade.");
        System.out.println("");
        
        // Vamos analisar o cálculo de percentil para cada caso
        // Para o teste 5 (nIters=7, radius=1) com 3 pares de DTW
        // DTW entre 0.3,0.4,0.5 e 0.6,0.7,0.8 = 0.6
        // DTW entre 0.3,0.4,0.5 e 0.7,0.8,0.9 = 1.2
        // DTW entre 0.4,0.5,0.6 e 0.7,0.8,0.9 = 0.6
        
        System.out.println("--- Cenarios de teste ---\n");

        int total = 0, passou = 0, falhou = 0;

        // Cenario 1: lessThan=true, percentile=100 -> todas as arestas (maximo)
        // Como 100 percentil = max, qualquer dtw < max e' verdade
        int r1 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "true", "100.0"});
        System.out.println("lessThan=true, p=100: " + r1 + " arestas (esperado: todas)");
        total++;
        if (r1 == 3) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 2: lessThan=true, percentile=0 -> nenhuma aresta (minimo)
        // Como 0 percentil = min, qualquer dtw < min e' falso
        int r2 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "true", "0.0"});
        System.out.println("lessThan=true, p=0: " + r2 + " arestas (esperado: 0)");
        total++;
        if (r2 == 0) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 3: lessThan=false, percentile=0 -> maioria (dtw > min para todos menos 1)
        int r3 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "false", "0.0"});
        System.out.println("lessThan=false, p=0: " + r3 + " arestas (esperado: maioria, 2)");
        total++;
        if (r3 == 2) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 4: lessThan=false, percentile=100 -> nenhuma (dtw > max sempre falso)
        int r4 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "false", "100.0"});
        System.out.println("lessThan=false, p=100: " + r4 + " arestas (esperado: 0)");
        total++;
        if (r4 == 0) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 5: lessThan=true, percentile=50 -> metade
        int r5 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "true", "50.0"});
        System.out.println("lessThan=true, p=50: " + r5 + " arestas");
        // DTWs: 0.6, 1.2, 0.6 -> ordenados: 0.6, 0.6, 1.2
        // Posicao = 0.5 * (3+1) = 2.0
        // sorted[1] + 0.0 * (sorted[2] - sorted[1]) = 0.6
        // Entao dtw < 0.6 -> apenas o 0.6, nao. Nenhuma.
        // Hmm, mas se o bound for 0.6 e dtw == 0.6, entao e' 0 arestas com lessThan
        total++;
        if (r5 == 0) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 6: lessThan=false, percentile=50 -> maioria
        int r6 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "false", "50.0"});
        System.out.println("lessThan=false, p=50: " + r6 + " arestas (esperado: 2 ou 3)");
        total++;
        if (r6 >= 2) passou++; else { falhou++; System.out.println("  [FAIL]"); }

        System.out.println("\n=== Verificacao de Logica ===");
        System.out.println("Se arestas sao criadas APENAS onde o percentil permite:");
        System.out.println("  - p=100, lessThan=true: TODAS (3) -> " + (r1==3 ? "OK" : "FALHOU"));
        System.out.println("  - p=0, lessThan=true: NENHUMA (0) -> " + (r2==0 ? "OK" : "FALHOU"));
        System.out.println("  - p=0, lessThan=false: MAIORIA (2) -> " + (r3==2 ? "OK" : "FALHOU"));
        System.out.println("  - p=100, lessThan=false: NENHUMA (0) -> " + (r4==0 ? "OK" : "FALHOU"));
        System.out.println("  - p=50, lessThan=true: METADE (1) -> " + (r5<=1 ? "OK" : "FALHOU") + " (obteve " + r5 + ")");
        System.out.println("  - p=50, lessThan=false: METADE (2) -> " + (r6>=2 ? "OK" : "FALHOU") + " (obteve " + r6 + ")");

        System.out.println("\n=== RESUMO ===");
        System.out.println("Total: " + total + " | Passou: " + passou + " | Falhou: " + falhou);
        if (falhou == 0) {
            System.out.println("\n>>> SIM, o codigo cria arestas APENAS onde o percentil permite <<<");
        } else {
            System.out.println("\n>>> NAO, ha problemas na logica do percentil <<<");
        }
    }
}
