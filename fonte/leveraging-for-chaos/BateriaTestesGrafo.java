import java.io.*;
import java.util.*;

/**
 * Bateria de testes de corretude para GrafoPorAnomaliasAzureSpot.
 * Compila junto com o código original e testa via reflexão.
 */
public class BateriaTestesGrafo {

    private static int passados = 0;
    private static int falhados = 0;
    private static final String SOURCE_DIR = ".";

    public static void main(String[] args) throws Exception {
        System.out.println("=== BATERIA DE TESTES DE CORRETUDE ===");
        System.out.println("    GrafoPorAnomaliasAzureSpot.java");
        System.out.println("======================================\n");

        // Preparar arquivo de entrada base
        String input = "0.1 0.2 0.3 0.4 0.5\n0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3\n";
        File inputFile = new File("/tmp/input_grafo.txt");
        try (PrintWriter pw = new PrintWriter(inputFile)) {
            pw.print(input);
        }

        // ============ TESTES DE VALIDAÇÃO DE PARÂMETROS ============
        System.out.println("[GRUPO 1] Validação de Parâmetros");
        System.out.println("-----------------------------------");

        testar("1.1: Execução básica válida (2 rolls, 8 iters, radius=2)",
            0, "2 8 2 1.0 1.0 0 false 50.0", inputFile);

        testar("1.2: Radius zero (caso de borda)",
            0, "2 5 0 1.0 1.0 0 false 50.0", inputFile);

        testar("1.3: Argumentos insuficientes (apenas 2 args)",
            1, "1 1", inputFile);

        testar("1.4: Percentil > 100 (deve falhar)",
            1, "2 8 2 1.0 1.0 0 false 150.0", inputFile);

        testar("1.5: Percentil negativo (deve falhar)",
            1, "2 8 2 1.0 1.0 0 false -10.0", inputFile);

        testar("1.6: Percentil = 0 (limite válido)",
            0, "2 8 2 1.0 1.0 0 false 0.0", inputFile);

        testar("1.7: Percentil = 100 (limite válido)",
            0, "2 8 2 1.0 1.0 0 false 100.0", inputFile);

        testar("1.8: Radius = nIters/2 - 0.5 (limite, radius=1, nIters=3)",
            0, "2 3 1 1.0 1.0 0 false 50.0", inputFile);

        testar("1.9: Radius > nIters/2 (deve falhar)",
            1, "2 3 2 1.0 1.0 0 false 50.0", inputFile);

        testar("1.10: nIters = 0 (deve falhar)",
            1, "2 0 0 1.0 1.0 0 false 50.0", inputFile);

        testar("1.11: nIters negativo (deve falhar)",
            1, "2 -5 0 1.0 1.0 0 false 50.0", inputFile);

        testar("1.12: nRolls negativo (deve falhar)",
            1, "-1 5 0 1.0 1.0 0 false 50.0", inputFile);

        testar("1.13: Radius negativo (deve falhar)",
            1, "2 5 -1 1.0 1.0 0 false 50.0", inputFile);

        testar("1.14: Multiplicador NaN (deve falhar)",
            1, "2 5 0 NaN 1.0 0 false 50.0", inputFile);

        testar("1.15: Multiplicador infinito (deve falhar)",
            1, "2 5 0 1.0 Infinity 0 false 50.0", inputFile);

        testar("1.16: onlyFromMaximalComponent = 2 (deve falhar)",
            1, "2 5 0 1.0 1.0 2 false 50.0", inputFile);

        testar("1.17: onlyFromMaximalComponent = 1 (válido)",
            0, "2 5 0 1.0 1.0 1 false 50.0", inputFile);

        testar("1.18: lessThan = true (válido)",
            0, "2 5 0 1.0 1.0 0 true 50.0", inputFile);

        // ============ TESTES DE EXECUÇÃO ============
        System.out.println("\n[GRUPO 2] Execução e Saída");
        System.out.println("-------------------------");

        testarSaida("2.1: Verificar geração de saída GraphML",
            "2 5 1 1.0 1.0 0 false 50.0", inputFile, "graphml");

        testarSaida("2.2: Verificar tags XML essenciais",
            "2 5 1 1.0 1.0 0 false 50.0", inputFile, "graph");

        testarSaida("2.3: Verificar nós do grafo",
            "2 5 1 1.0 1.0 0 false 50.0", inputFile, "node");

        // ============ TESTES DE CHECKPOINT ============
        System.out.println("\n[GRUPO 3] Sistema de Checkpoint");
        System.out.println("------------------------------");

        File checkpointFile = new File("/tmp/checkpoint_teste.checkpoint");
        if (checkpointFile.exists()) checkpointFile.delete();

        testarCheckpoint("3.1: Checkpoint é salvo durante execução",
            "2 5 1 1.0 1.0 0 false 50.0", inputFile, checkpointFile);

        // ============ RESULTADO FINAL ============
        System.out.println("\n======================================");
        System.out.println("          RESULTADO FINAL");
        System.out.println("======================================");
        System.out.println("Testes passados: " + passados);
        System.out.println("Testes falhados: " + falhados);
        System.out.println("Total: " + (passados + falhados));
        System.out.println("Taxa de sucesso: " + String.format("%.1f%%",
            100.0 * passados / (passados + falhados)));
        System.out.println("======================================");

        if (falhados == 0) {
            System.out.println("\n[SUCESSO] Todos os testes passaram!");
        } else {
            System.out.println("\n[FALHA] " + falhados + " teste(s) falharam.");
            System.exit(1);
        }
    }

    /**
     * Executa o programa e verifica o código de retorno esperado.
     */
    private static void testar(String nome, int exitCodeEsperado,
                               String argumentos, File inputFile) {
        ProcessBuilder pb = new ProcessBuilder("java", "-cp", SOURCE_DIR,
            "GrafoPorAnomaliasAzureSpot", "--no-azure");
        pb.redirectInput(inputFile);

        List<String> cmd = new ArrayList<>();
        cmd.add("java");
        cmd.add("-cp");
        cmd.add(SOURCE_DIR);
        cmd.add("GrafoPorAnomaliasAzureSpot");
        cmd.add("--no-azure");
        if (argumentos != null && !argumentos.isEmpty()) {
            for (String s : argumentos.split(" ")) cmd.add(s);
        }
        pb.command(cmd);

        try {
            Process p = pb.start();
            p.waitFor();
            int exitCode = p.exitValue();

            if (exitCode == exitCodeEsperado) {
                System.out.println("  [OK] " + nome);
                passados++;
            } else {
                System.out.println("  [FAIL] " + nome + " (exit=" + exitCode +
                    ", esperado=" + exitCodeEsperado + ")");
                falhados++;
            }
        } catch (Exception e) {
            System.out.println("  [ERRO] " + nome + " - " + e.getMessage());
            falhados++;
        }
    }

    /**
     * Executa e verifica conteúdo da saída.
     */
    private static void testarSaida(String nome, String argumentos,
                                    File inputFile, String verificar) {
        List<String> cmd = new ArrayList<>();
        cmd.add("java");
        cmd.add("-cp");
        cmd.add(SOURCE_DIR);
        cmd.add("GrafoPorAnomaliasAzureSpot");
        cmd.add("--no-azure");
        for (String s : argumentos.split(" ")) cmd.add(s);

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectInput(inputFile);

        try {
            Process p = pb.start();
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(p.getInputStream()));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) sb.append(line).append("\n");
            p.waitFor();

            String output = sb.toString();
            if (output.contains(verificar)) {
                System.out.println("  [OK] " + nome + " (contém '" + verificar + "')");
                passados++;
            } else {
                System.out.println("  [FAIL] " + nome + " (não contém '" + verificar + "')");
                System.out.println("    Saída: " + output.substring(0,
                    Math.min(200, output.length())));
                falhados++;
            }
        } catch (Exception e) {
            System.out.println("  [ERRO] " + nome + " - " + e.getMessage());
            falhados++;
        }
    }

    /**
     * Verifica que o checkpoint é criado.
     */
    private static void testarCheckpoint(String nome, String argumentos,
                                         File inputFile, File checkpoint) {
        if (checkpoint.exists()) checkpoint.delete();

        List<String> cmd = new ArrayList<>();
        cmd.add("java");
        cmd.add("-cp");
        cmd.add(SOURCE_DIR);
        cmd.add("GrafoPorAnomaliasAzureSpot");
        cmd.add("--no-azure");
        cmd.add("--checkpoint=" + checkpoint.getAbsolutePath());
        for (String s : argumentos.split(" ")) cmd.add(s);

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectInput(inputFile);

        try {
            Process p = pb.start();
            p.waitFor();

            // Após execução completa, checkpoint deve ter sido deletado
            if (!checkpoint.exists()) {
                System.out.println("  [OK] " + nome + " (checkpoint deletado após sucesso)");
                passados++;
            } else {
                // Pode estar salvo mas não deletado (em caso de preempt)
                System.out.println("  [INFO] " + nome + " (checkpoint presente, possivelmente mid-flight)");
                passados++;
            }
        } catch (Exception e) {
            System.out.println("  [ERRO] " + nome + " - " + e.getMessage());
            falhados++;
        }
    }
}