import java.io.*;
import java.util.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * HistogramCalculator com checkpoint cooperativo para Azure Spot.
 * 
 * <p>Calcula histogramas de um conjunto de dados com tolerância a interrupções de VM, seguindo os padrões de resiliência
 * de GrafoPorAnomaliasAzureSpot.java.</p>
 */
public class HistogramCalculatorAzureSpot {
    private static final String EVENT_URL =
        "http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01";
    private static final long DEFAULT_POLL_MS = 5_000L;
    private static final long CHECKPOINT_EVERY = 16L;

    private static final AtomicBoolean stopRequested = new AtomicBoolean();
    private static volatile HistogramState activeState;
    private static volatile String checkpointPath;

    private static final class HistogramState implements Serializable {
        private static final long serialVersionUID = 2L;
        List<Double> inputData;
        double[] sortedData;
        double lower;
        double upper;
        double eps;
        int left;
        int right;
        int[] density;
        Integer[] indices;
        int[] histogram;
        int phase;  // 0: reading, 1: density calculation, 2: histogram by rank
        int processedCount;
        int windowStart;
        int currentIndex;
    }

    private static final class Options {
        String inputPath;
        String checkpointPath = "histogram-calculator.checkpoint";
        long pollMs = DEFAULT_POLL_MS;
        boolean azure = true;
        String[] positional;
    }

    public static void main(String[] args) throws Exception {
        Options options = parseOptions(args);
        checkpointPath = options.checkpointPath;
        HistogramState state = readCheckpoint(options.checkpointPath);
        activeState = state;

        Thread shutdownHook = new Thread(() -> saveQuietly(activeState),
            "histogram-calculator-checkpoint");
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        Thread eventMonitor = null;
        if (options.azure) {
            eventMonitor = startAzureEventMonitor(options.pollMs);
        }

        try {
            run(options, state);
        } finally {
            if (eventMonitor != null) eventMonitor.interrupt();
        }
    }

    private static Options parseOptions(String[] args) {
        Options options = new Options();
        java.util.ArrayList<String> positional = new java.util.ArrayList<>();
        for (String arg : args) {
            if (arg.startsWith("--checkpoint=")) {
                options.checkpointPath = arg.substring("--checkpoint=".length());
            } else if (arg.startsWith("--poll-ms=")) {
                options.pollMs = Long.parseLong(arg.substring("--poll-ms=".length()));
            } else if (arg.equals("--no-azure")) {
                options.azure = false;
            } else {
                positional.add(arg);
            }
        }
        // Primeiro argumento posicional é o arquivo de entrada (opcional, senão usa stdin)
        if (positional.size() > 0 && !isNumeric(positional.get(0))) {
            options.inputPath = positional.get(0);
            positional.remove(0);
        }
        options.positional = positional.toArray(new String[0]);
        return options;
    }

    private static boolean isNumeric(String str) {
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static void run(Options options, HistogramState state) throws Exception {
        boolean keepCheckpoint = Boolean.getBoolean("histogram.test.keepCheckpoint");
        if (state == null) {
            if (options.positional.length < 5) {
                throw new IllegalArgumentException("Uso: [arquivo] lower upper eps left right [--checkpoint=arquivo]");
            }
            state = loadInput(options);
            activeState = state;
            save(state);
        } else {
            System.err.println("Retomando checkpoint " + checkpointPath +
                " na fase " + state.phase + " (processados=" + state.processedCount + ")");
        }

        if (state.phase == 0) {
            // Fase 1: Leitura dos dados
            state.phase = 1;
            processInputData(state);
            save(state);
        }

        if (state.phase == 1) {
            // Fase 2: Cálculo da densidade
            state.phase = 2;
            processDensityCalculation(state);
            save(state);
        }

        if (state.phase == 2) {
            // Fase 3: Histograma por rank
            state.phase = 3;
            processHistogramByRank(state);
            save(state);
        }

        if (stopRequested.get()) return;
        writeHistogram(state);
        if (!keepCheckpoint) deleteCheckpoint();
    }

    private static HistogramState loadInput(Options options) throws IOException {
        HistogramState state = new HistogramState();
        InputStream in = options.inputPath == null ? System.in : new FileInputStream(options.inputPath);
        Scanner sc = new Scanner(in).useLocale(Locale.US);
        
        // Parse arguments
        String[] args = options.positional;
        state.lower = Double.parseDouble(args[0]);
        state.upper = Double.parseDouble(args[1]);
        state.eps = Double.parseDouble(args[2]);
        state.left = Integer.parseInt(args[3]);
        state.right = Integer.parseInt(args[4]);

        // Read sequence
        List<Double> list = new ArrayList<>();
        while (sc.hasNextDouble()) list.add(sc.nextDouble());
        state.inputData = list;
        double[] data = new double[list.size()];
        for (int i = 0; i < list.size(); i++) data[i] = list.get(i);
        
        // Sort the data
        state.sortedData = data.clone();
        Arrays.sort(state.sortedData);
        
        return state;
    }

    private static void processInputData(HistogramState state) {
        // Basic processing - in a real implementation, this would include checkpointing
        // at various stages to allow for interruption and recovery
        state.processedCount = state.inputData.size();
        state.windowStart = 0;
        state.currentIndex = 0;
    }

    private static void processDensityCalculation(HistogramState state) throws Exception {
        // Process density calculation with checkpointing
        state.density = new int[state.sortedData.length];
        for (int i = 0; i < state.sortedData.length; i++) {
            if (stopRequested.get()) return;
            
            double v = state.sortedData[i];
            if (v < state.lower + state.eps || v > state.upper - state.eps) {
                state.density[i] = 0;
                continue;
            }
            
            int p1 = lowerBound(state.sortedData, v - state.eps);
            int p2 = upperBound(state.sortedData, v + state.eps);
            state.density[i] = p2 - p1 + 1;
            
            if ((i & (CHECKPOINT_EVERY - 1)) == 0) {
                save(state);
            }
        }
    }

    private static void processHistogramByRank(HistogramState state) throws Exception {
        // Process histogram by rank with checkpointing
        int n = state.inputData.size();
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = state.inputData.get(i);
        
        state.indices = new Integer[n];
        for (int i = 0; i < n; i++) state.indices[i] = i;
        Arrays.sort(state.indices, (a, b) -> {
            int c = Double.compare(data[a], data[b]);
            return (c != 0) ? c : a - b;
        });
        
        state.histogram = new int[n];
        for (int r = state.left; r < state.right && r < n; r++) {
            if (stopRequested.get()) return;
            state.histogram[state.indices[r]]++;
            
            if ((r & (CHECKPOINT_EVERY - 1)) == 0) {
                save(state);
            }
        }
    }

    private static Thread startAzureEventMonitor(long pollMs) {
        Thread monitor = new Thread(() -> {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            while (!Thread.currentThread().isInterrupted() && !stopRequested.get()) {
                try {
                    HttpRequest request = HttpRequest.newBuilder(URI.create(EVENT_URL))
                        .timeout(Duration.ofSeconds(3)).header("Metadata", "true").GET().build();
                    HttpResponse<String> response = client.send(request,
                        HttpResponse.BodyHandlers.ofString());
                    if (response.statusCode() == 200 && isPreemptEvent(response.body())) {
                        stopRequested.set(true);
                        System.err.println(Instant.now() +
                            " Azure Spot Preempt recebido; checkpoint sera salvo.");
                        return;
                    }
                } catch (Exception ignored) {
                    // IMDS fica indisponivel fora do Azure; isso nao impede o processamento.
                }
                try {
                    Thread.sleep(Math.max(250L, pollMs));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }, "azure-spot-scheduled-events");
        monitor.setDaemon(true);
        monitor.start();
        return monitor;
    }

    private static boolean isPreemptEvent(String json) {
        return json.contains("\"EventType\":\"Preempt\"") ||
            json.contains("\"EventType\": \"Preempt\"");
    }

    private static synchronized void save(HistogramState state) throws IOException {
        activeState = state;
        String temporary = checkpointPath + ".tmp";
        try (ObjectOutputStream output = new ObjectOutputStream(new BufferedOutputStream(
                new FileOutputStream(temporary)))) {
            output.writeObject(state);
        }
        java.nio.file.Files.move(java.nio.file.Path.of(temporary),
            java.nio.file.Path.of(checkpointPath), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
            java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        System.err.println(Instant.now() + " checkpoint salvo: fase=" + state.phase +
            ", processados=" + state.processedCount);
    }

    private static HistogramState readCheckpoint(String path) {
        try (ObjectInputStream input = new ObjectInputStream(new BufferedInputStream(
                new FileInputStream(path)))) {
            return (HistogramState) input.readObject();
        } catch (IOException | ClassNotFoundException absent) {
            return null;
        }
    }

    private static void saveQuietly(HistogramState state) {
        if (state != null && checkpointPath != null && state.phase < 3) {
            try { save(state); } catch (IOException error) {
                System.err.println("Nao foi possivel salvar checkpoint: " + error.getMessage());
            }
        }
    }

    private static void deleteCheckpoint() throws IOException {
        java.nio.file.Files.deleteIfExists(java.nio.file.Path.of(checkpointPath));
    }

    private static void writeHistogram(HistogramState state) {
        PrintWriter out = new PrintWriter(System.out);
        out.print(state.histogram[0]);
        for (int i = 1; i < state.histogram.length; i++) out.print("," + state.histogram[i]);
        out.println();
        out.flush();
    }

    // Reusing the binary search methods from the original HistogramCalculator
    private static int lowerBound(double[] s, double t) {
        int lo = 0, hi = s.length;
        while (lo < hi) { int m = (lo + hi) >>> 1; if (s[m] < t) lo = m + 1; else hi = m; }
        return lo;
    }

    private static int upperBound(double[] s, double t) {
        int lo = 0, hi = s.length;
        while (lo < hi) { int m = (lo + hi) >>> 1; if (s[m] <= t) lo = m + 1; else hi = m; }
        return lo - 1;
    }
}