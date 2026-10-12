import java.io.*;

public class BateriaTestes {
    private static int total = 0;
    private static int passou = 0;
    private static int falhou = 0;

    private static void teste(String nome, String entrada, boolean esperaSucesso) {
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "1.0", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File(entrada));
            Process p = pb.start();
            p.waitFor();
            int exit = p.exitValue();
            if (esperaSucesso && exit == 0) {
                passou++;
                System.out.println("[OK] " + nome);
            } else if (!esperaSucesso && exit != 0) {
                passou++;
                System.out.println("[OK] " + nome + " (rejeitado como esperado)");
            } else {
                falhou++;
                System.out.println("[FAIL] " + nome);
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + nome + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {
        // Criar arquivo de entrada
        try (PrintWriter pw = new PrintWriter("input.txt")) {
            pw.println("0.1 0.2 0.3 0.4 0.5");
        }

        System.out.println("=== Bateria de Testes - GrafoPorAnomaliasAzureSpot ===\n");

        // Teste 1: Execução básica válida
        teste("Execucao basica valida", "input.txt", true);

        // Teste 2: Arquivo de entrada inexistente
        teste("Arquivo inexistente", "/tmp/naoexiste.txt", false);

        // Teste 3: Validacao de percentil > 100
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "1.0", "1.0", "0", "false", "150.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] Percentil > 100 rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] Percentil > 100 aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 4: Validacao de radius > nIters/2
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "2", "1.0", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] Radius > nIters/2 rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] Radius > nIters/2 aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 5: Validacao de NaN
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "NaN", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] NaN rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] NaN aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 6: Validacao de Infinity
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "Infinity", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] Infinity rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] Infinity aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 7: Validacao de nIters = 0
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "0", "1", "1.0", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] nIters=0 rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] nIters=0 aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 8: Validacao de nRolls negativo
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "-1", "3", "1", "1.0", "1.0", "0", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] nRolls negativo rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] nRolls negativo aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 9: Validacao de onlyFromMaximalComponent invalido
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "1.0", "1.0", "2", "false", "50.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] onlyFromMaximalComponent=2 rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] onlyFromMaximalComponent=2 aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 10: Validacao de percentil negativo
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "2", "3", "1", "1.0", "1.0", "0", "false", "-10.0");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] Percentil negativo rejeitado");
            } else {
                falhou++;
                System.out.println("[FAIL] Percentil negativo aceito");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        // Teste 11: Argumentos insuficientes
        total++;
        try {
            ProcessBuilder pb = new ProcessBuilder("java", "-cp", ".", "GrafoPorAnomaliasAzureSpot",
                "--no-azure", "1", "1");
            pb.directory(new File("."));
            pb.redirectInput(new File("input.txt"));
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() != 0) {
                passou++;
                System.out.println("[OK] Argumentos insuficientes rejeitados");
            } else {
                falhou++;
                System.out.println("[FAIL] Argumentos insuficientes aceitos");
            }
        } catch (Exception e) {
            falhou++;
            System.out.println("[ERRO] " + e.getMessage());
        }

        System.out.println("\n=== RESUMO ===");
        System.out.println("Total: " + total);
        System.out.println("Passou: " + passou);
        System.out.println("Falhou: " + falhou);
        if (falhou == 0) {
            System.out.println(">>> TODOS OS TESTES PASSARAM <<<");
        } else {
            System.out.println(">>> HOUVE FALHAS <<<");
            System.exit(1);
        }
    }
}
