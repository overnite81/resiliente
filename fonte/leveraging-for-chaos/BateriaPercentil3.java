import java.io.*;
import java.util.regex.*;

/**
 * Testa se as arestas do grafo são criadas APENAS onde o cálculo
 * de percentil permite. Aumenta o tamanho para ter mais pares.
 */
public class BateriaPercentil3 {

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
        // nIters=10, nRolls=2, radius=1 -> 10*9/2 = 45 pares possiveis
        // Mas com restricao j >= i+3 e j < 9:
        // i=1: j in [4,9) -> 5 pares
        // i=2: j in [5,9) -> 4 pares
        // ...
        // i=7: j in [10,9) -> 0
        // Total: 5+4+3+2+1+0 = 15 pares
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2; i++) sb.append((0.1*i) + " ");
        for (int i = 0; i < 10; i++) sb.append((0.3 + 0.1*i) + " ");
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println(sb.toString());
        }

        System.out.println("=== TESTE: Arestas conforme Percentil (nIters=10) ===\n");
        System.out.println("Configuracao: nIters=10, nRolls=2, radius=1");
        System.out.println("Sequencia crescente: 0.3, 0.4, ..., 1.2");
        System.out.println("Pares de DTW possiveis: 15");
        System.out.println("");

        int total = 0, passou = 0, falhou = 0;
        boolean[] resultados = new boolean[6];

        // Cenario 1: lessThan=true, percentile=100 -> todas as 15
        int r1 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "100.0"});
        System.out.println("lessThan=true,  p=100: " + r1 + " arestas (esperado: 15)");
        total++;
        if (r1 == 15) { passou++; resultados[0] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 2: lessThan=true, percentile=0 -> 0
        int r2 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "0.0"});
        System.out.println("lessThan=true,  p=0:   " + r2 + " arestas (esperado: 0)");
        total++;
        if (r2 == 0) { passou++; resultados[1] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 3: lessThan=false, percentile=0 -> 14 (todas menos a menor)
        int r3 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "0.0"});
        System.out.println("lessThan=false, p=0:   " + r3 + " arestas (esperado: 14)");
        total++;
        if (r3 == 14) { passou++; resultados[2] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 4: lessThan=false, percentile=100 -> 0
        int r4 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "100.0"});
        System.out.println("lessThan=false, p=100: " + r4 + " arestas (esperado: 0)");
        total++;
        if (r4 == 0) { passou++; resultados[3] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 5: lessThan=true, percentile=50 -> metade (~7-8)
        int r5 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "50.0"});
        System.out.println("lessThan=true,  p=50:  " + r5 + " arestas (esperado: ~7-8)");
        total++;
        if (r5 >= 5 && r5 <= 10) { passou++; resultados[4] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario 6: lessThan=false, percentile=50 -> metade
        int r6 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "50.0"});
        System.out.println("lessThan=false, p=50:  " + r6 + " arestas (esperado: ~7-8)");
        total++;
        if (r6 >= 5 && r6 <= 10) { passou++; resultados[5] = true; }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Cenario bonus: lessThan=true, p=100 + lessThan=true, p=0 = 15 (todas)
        int r1r2 = r1 + r2;
        System.out.println("\nSoma p=100(true) + p=0(true) = " + r1r2 + " (esperado: 15 = total)");

        // lessThan=false, p=0 + lessThan=true, p=0 = 14 (todas menos 1)
        int r3maisr2 = r3 + r2;
        System.out.println("Soma p=0(false) + p=0(true) = " + r3maisr2 + " (esperado: 14)");

        System.out.println("\n=== Verificacao da Logica de Percentil ===");
        System.out.println("Se arestas seguem o percentil CORRETAMENTE:");
        System.out.println("  [1] p=100, true:  TODAS (15)        -> " + (resultados[0] ? "OK" : "FALHOU"));
        System.out.println("  [2] p=0,   true:  NENHUMA (0)      -> " + (resultados[1] ? "OK" : "FALHOU"));
        System.out.println("  [3] p=0,   false: MAIORIA (14)     -> " + (resultados[2] ? "OK" : "FALHOU"));
        System.out.println("  [4] p=100, false: NENHUMA (0)      -> " + (resultados[3] ? "OK" : "FALHOU"));
        System.out.println("  [5] p=50,  true:  METADE (5-10)    -> " + (resultados[4] ? "OK" : "FALHOU"));
        System.out.println("  [6] p=50,  false: METADE (5-10)    -> " + (resultados[5] ? "OK" : "FALHOU"));

        System.out.println("\n=== RESUMO ===");
        System.out.println("Total: " + total + " | Passou: " + passou + " | Falhou: " + falhou);
        if (falhou == 0) {
            System.out.println("\n>>> SIM, as arestas sao criadas APENAS onde o percentil permite <<<");
        } else {
            System.out.println("\n>>> NAO, ha problemas na logica do percentil <<<");
        }
    }
}
