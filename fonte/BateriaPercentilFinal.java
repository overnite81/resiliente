import java.io.*;
import java.util.*;

/**
 * Testa se as arestas do grafo são criadas APENAS onde o cálculo
 * de percentil permite. Usa a logica CORRETA.
 */
public class BateriaPercentilFinal {

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
        // Sequencia com valores distintos para teste claro
        // nIters=10, radius=1 -> 15 pares de DTW
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2; i++) sb.append((0.1*i) + " ");
        for (int i = 0; i < 10; i++) sb.append((0.3 + 0.1*i) + " ");
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println(sb.toString());
        }

        System.out.println("=== TESTE FINAL: Arestas conforme Percentil ===\n");
        System.out.println("nIters=10, radius=1 -> 15 pares de DTW\n");

        int total = 0, passou = 0, falhou = 0;
        boolean[] resultados = new boolean[8];

        // A logica correta:
        // lessThan=true: seleciona dtw < bound (menor que percentil)
        // lessThan=false: seleciona dtw > bound (maior que percentil)
        
        // Percentil 100 = maximo -> bound = maior DTW
        // Percentil 0 = minimo -> bound = menor DTW

        // Teste 1: p=100, lessThan=true -> TODOS MENOS O MAXIMO
        // dtw < max = todos exceto o maximo = 14
        int r1 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "100.0"});
        System.out.println("p=100, lessThan=true: " + r1 + " arestas");
        System.out.println("  Esperado: 14 (todos MENOS o maximo, pois dtw < max)");
        total++;
        if (r1 == 14) { passou++; resultados[0] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 2: p=0, lessThan=true -> NENHUMA
        // dtw < min = falso para todos = 0
        int r2 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "0.0"});
        System.out.println("p=0, lessThan=true: " + r2 + " arestas");
        System.out.println("  Esperado: 0 (nenhum menor que o minimo)");
        total++;
        if (r2 == 0) { passou++; resultados[1] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 3: p=0, lessThan=false -> TODOS MENOS O MINIMO
        // dtw > min = todos exceto o minimo = 14
        int r3 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "0.0"});
        System.out.println("p=0, lessThan=false: " + r3 + " arestas");
        System.out.println("  Esperado: 14 (todos MENOS o minimo)");
        total++;
        if (r3 == 14) { passou++; resultados[2] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 4: p=100, lessThan=false -> NENHUMA
        // dtw > max = falso para todos = 0
        int r4 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "100.0"});
        System.out.println("p=100, lessThan=false: " + r4 + " arestas");
        System.out.println("  Esperado: 0 (nenhum maior que o maximo)");
        total++;
        if (r4 == 0) { passou++; resultados[3] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 5: p=50, lessThan=true -> METADE
        // dtw < mediana = ~7-8 (metade inferior)
        int r5 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "true", "50.0"});
        System.out.println("p=50, lessThan=true: " + r5 + " arestas");
        System.out.println("  Esperado: ~7 (metade inferior)");
        total++;
        if (r5 >= 5 && r5 <= 9) { passou++; resultados[4] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 6: p=50, lessThan=false -> METADE
        // dtw > mediana = ~7-8 (metade superior)
        int r6 = executar(new String[]{"2", "10", "1", "1.0", "1.0", "0", "false", "50.0"});
        System.out.println("p=50, lessThan=false: " + r6 + " arestas");
        System.out.println("  Esperado: ~7 (metade superior)");
        total++;
        if (r6 >= 5 && r6 <= 9) { passou++; resultados[5] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        // Teste 7: Verificar que p=true + p=false = total (menos os iguais ao bound)
        int soma1 = r1 + r2;
        int soma2 = r3 + r4;
        System.out.println("\nSoma p=100(true) + p=0(true) = " + soma1 + " (esperado: 14)");
        total++;
        if (soma1 == 14) { passou++; resultados[6] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        System.out.println("Soma p=0(false) + p=100(false) = " + soma2 + " (esperado: 14)");
        total++;
        if (soma2 == 14) { passou++; resultados[7] = true; System.out.println("  [OK]"); }
        else { falhou++; System.out.println("  [FAIL]"); }

        System.out.println("\n=== RESUMO ===");
        System.out.println("Total: " + total + " | Passou: " + passou + " | Falhou: " + falhou);
        if (falhou == 0) {
            System.out.println("\n>>> SIM, o codigo cria arestas APENAS onde o percentil permite <<<");
        } else {
            System.out.println("\n>>> NAO, ha problemas na logica do percentil <<<");
        }
    }
}
