import java.io.*;
import java.util.regex.*;

/**
 * Testa se as arestas do grafo são criadas APENAS onde o cálculo
 * de percentil permite (análise estática + funcional).
 */
public class BateriaPercentil {

    private static int total = 0;
    private static int passou = 0;
    private static int falhou = 0;

    /**
     * Executa o programa com parâmetros dados e retorna a saída GraphML.
     */
    private static String executar(String[] args) throws Exception {
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot", "--no-azure");
            pb.directory(new File("."));
            String[] fullArgs = new String[args.length + 3];
            fullArgs[0] = "java";
            fullArgs[1] = "-cp";
            fullArgs[2] = ".";
            // Monta corretamente
            java.util.List<String> cmd = new java.util.ArrayList<>();
            cmd.add("java"); cmd.add("-cp"); cmd.add("."); 
            cmd.add("GrafoPorAnomaliasAzureSpot"); cmd.add("--no-azure");
            for (String a : args) cmd.add(a);
            pb = new ProcessBuilder(cmd);
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                falhou++;
                System.out.println("[FAIL] Execucao retornou codigo " + p.exitValue());
                return null;
            }
            passou++;
            return new String(p.getInputStream().readAllBytes());
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
            return null;
        }
    }

    /**
     * Conta o número de arestas no GraphML.
     */
    private static int contarArestas(String graphml) {
        if (graphml == null) return -1;
        return graphml.split("<edge ", -1).length - 1;
    }

    public static void main(String[] args) throws Exception {
        // Cria arquivo de entrada com 10 valores (2 rolls + 3 iteracoes)
        // Para testar: nRolls=2, nIters=3, radius=1
        // Precisamos de 2 + 3 = 5 valores
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5");
        }

        System.out.println("=== Analise: Arestas vs Calculo de Percentil ===\n");

        // Analise 1: Verificar que arestas SÓ são criadas conforme o percentil
        // Com nIters=3, nRolls=2, radius=1, e lessThan=true,
        // apenas DTWs MENORES que o bound (50 percentil) devem virar arestas
        System.out.println("--- Teste 1: lessThan=true, percentile=50 ---");
        String g1 = executar(new String[]{"2", "3", "1", "1.0", "1.0", "0", "true", "50.0"});
        int arestas1 = contarArestas(g1);
        System.out.println("Arestas criadas: " + arestas1);
        System.out.println("Com nIters=3 e radius=1, total de pares possiveis: 3 (j>i+radius*2)");
        System.out.println("  (0,2), (1,3 se existisse) - com nIters=3 só temos (0,2)");
        // Na verdade, com nIters=3, radius=1: i=1, j>=1+3=4 -> nenhum par!
        // Hmm, na verdade o cálculo é: i < nIters, j < nIters - radius
        // i pode ser 1, j pode ser < 2. j >= i + 2*radius + 1 = 1 + 3 = 4
        // Então j nunca satisfaz j >= 4 e j < 2. Zero pares!
        System.out.println("Esperado: 0 arestas (limites do loop)");
        if (arestas1 == 0) {
            System.out.println("[OK] Nenhuma aresta criada (correto, fora do alcance)\n");
        } else {
            System.out.println("[INFO] Arestas = " + arestas1 + "\n");
        }

        // Analise 2: Teste com nIters=4, nRolls=2, radius=1
        // i=1, j>=1+3=4, j<3 -> 0 pares
        // i=2, j>=5, j<3 -> 0 pares
        // Nenhum par! Sempre zero com nIters pequeno
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5 0.6");
        }
        // nIters=4, nRolls=2 -> precisa de 2+4=6 valores
        System.out.println("--- Teste 2: nIters=4, nRolls=2, radius=1 ---");
        System.out.println("  nRolls=2 + nIters=4 = 6 valores necessarios");
        String g2 = executar(new String[]{"2", "4", "1", "1.0", "1.0", "0", "true", "50.0"});
        int arestas2 = contarArestas(g2);
        System.out.println("Arestas criadas: " + arestas2);
        // Com nIters=4, radius=1: i=1, j>=4, j<3 -> nada
        if (arestas2 == 0) {
            System.out.println("[OK] Sem arestas (j nunca satisfaz j >= i+2*radius+1 com nIters=4)\n");
        }

        // Analise 3: nIters=5, nRolls=2, radius=1
        // i=1, j>=4, j<4 -> j=4 não entra (j<4). Nada.
        // i=2, j>=5, j<4 -> nada.
        // i=3, j>=6, j<4 -> nada.
        // Zero arestas sempre!
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5 0.6 0.7");
        }
        System.out.println("--- Teste 3: nIters=5, nRolls=2, radius=1 ---");
        System.out.println("  nRolls=2 + nIters=5 = 7 valores necessarios");
        String g3 = executar(new String[]{"2", "5", "1", "1.0", "1.0", "0", "true", "50.0"});
        int arestas3 = contarArestas(g3);
        System.out.println("Arestas criadas: " + arestas3);
        
        // Analise 4: nIters=6, nRolls=2, radius=1
        // i=1, j>=4, j<5 -> j=4 ENTRA! 1 par
        // i=2, j>=5, j<5 -> j=5 não (j<5). 0 pares nessa linha
        // i=3, j>=6, j<5 -> nada
        // i=4, j>=7, j<5 -> nada
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8");
        }
        System.out.println("\n--- Teste 4: nIters=6, nRolls=2, radius=1 ---");
        System.out.println("  nRolls=2 + nIters=6 = 8 valores necessarios");
        String g4 = executar(new String[]{"2", "6", "1", "1.0", "1.0", "0", "true", "50.0"});
        int arestas4 = contarArestas(g4);
        System.out.println("Arestas criadas (lessThan=true, p=50): " + arestas4);
        String g5 = executar(new String[]{"2", "6", "1", "1.0", "1.0", "0", "false", "50.0"});
        int arestas5 = contarArestas(g5);
        System.out.println("Arestas criadas (lessThan=false, p=50): " + arestas5);
        
        // Com 1 par de DTW, lessThan=true seleciona só se dtw < bound (50 percentil)
        // lessThan=false seleciona só se dtw > bound (50 percentil)
        // Um dos dois vai ficar vazio!
        System.out.println("\n--- Analise logica ---");
        System.out.println("Com 1 par de DTW:");
        System.out.println("  lessThan=true: cria aresta SÓ se dtw < bound (50 percentil)");
        System.out.println("  lessThan=false: cria aresta SÓ se dtw > bound (50 percentil)");
        System.out.println("  bound = percentil 50 = mediana (pode ser 0 ou 1 aresta)");

        // Analise 5: nIters=7, nRolls=2, radius=1
        // i=1, j=4 -> 1 par
        // i=2, j=5 -> 1 par
        // i=3, j=6 -> 1 par
        // Total: 3 pares
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9");
        }
        System.out.println("\n--- Teste 5: nIters=7, nRolls=2, radius=1 ---");
        System.out.println("  nRolls=2 + nIters=7 = 9 valores");
        String g6 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "true", "100.0"});
        int arestas6 = contarArestas(g6);
        System.out.println("Arestas com percentile=100 (lessThan=true): " + arestas6);
        // 100 percentil = max, então dtw < max é sempre true -> todas as arestas
        String g7 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "true", "0.0"});
        int arestas7 = contarArestas(g7);
        System.out.println("Arestas com percentile=0 (lessThan=true): " + arestas7);
        // 0 percentil = min, então dtw < min é sempre false -> 0 arestas
        String g8 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "false", "0.0"});
        int arestas8 = contarArestas(g8);
        System.out.println("Arestas com percentile=0 (lessThan=false): " + arestas8);
        // 0 percentil = min, então dtw > min é true se houver mais de 1 valor, e dtw != min
        String g9 = executar(new String[]{"2", "7", "1", "1.0", "1.0", "0", "false", "100.0"});
        int arestas9 = contarArestas(g9);
        System.out.println("Arestas com percentile=100 (lessThan=false): " + arestas9);
        // 100 percentil = max, então dtw > max é sempre false -> 0 arestas

        System.out.println("\n=== Verificacao da logica de percentil ===");
        // Com 3 pares de DTW:
        // - percentile=100, lessThan=true: TODAS as arestas (dtw < max sempre true)
        // - percentile=0, lessThan=true: NENHUMA aresta (dtw < min sempre false)
        // - percentile=0, lessThan=false: 2 ou 3 arestas (dtw > min para todos menos o min)
        // - percentile=100, lessThan=false: NENHUMA aresta (dtw > max sempre false)
        
        boolean testeA = (arestas6 == 3);  // p=100, lessThan=true: todas
        boolean testeB = (arestas7 == 0);  // p=0, lessThan=true: nenhuma
        boolean testeC = (arestas8 >= 2);  // p=0, lessThan=false: maioria
        boolean testeD = (arestas9 == 0);  // p=100, lessThan=false: nenhuma
        
        System.out.println("\nResultados esperados vs obtidos:");
        System.out.println("  p=100, lessThan=true -> todas (3): " + (testeA ? "OK" : "FALHOU") + " obtido=" + arestas6);
        System.out.println("  p=0,   lessThan=true -> nenhuma (0): " + (testeB ? "OK" : "FALHOU") + " obtido=" + arestas7);
        System.out.println("  p=0,   lessThan=false -> maioria (2+): " + (testeC ? "OK" : "FALHOU") + " obtido=" + arestas8);
        System.out.println("  p=100, lessThan=false -> nenhuma (0): " + (testeD ? "OK" : "FALHOU") + " obtido=" + arestas9);

        System.out.println("\n=== RESUMO ===");
        System.out.println("Total de execucoes: " + total);
        System.out.println("Passou: " + passou);
        System.out.println("Falhou: " + falhou);

        if (testeA && testeB && testeC && testeD) {
            System.out.println("\n>>> SIM, as arestas sao criadas APENAS conforme o percentil permite <<<");
        } else {
            System.out.println("\n>>> HA INCONSISTENCIA com a logica do percentil <<<");
        }
    }
}
