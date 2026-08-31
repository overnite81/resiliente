import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * GrafoPorAnomalias com checkpoint cooperativo para Azure Spot.
 *
 * <p>O processo consulta o Azure Scheduled Events (IMDS). Ao receber um
 * evento Preempt, grava o estado e deixa os loops terminarem no ponto atual.
 * Ao ser iniciado novamente, o checkpoint e carregado automaticamente.</p>
 */
public class GrafoPorAnomaliasAzureSpot {
    private static final String EVENT_URL =
        "http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01";
    private static final long DEFAULT_POLL_MS = 5_000L;
    private static final long CHECKPOINT_EVERY = 32L;

    private static final AtomicBoolean stopRequested = new AtomicBoolean();
    private static volatile Checkpoint activeCheckpoint;
    private static volatile String checkpointPath;

    private static final class Checkpoint implements Serializable {
        private static final long serialVersionUID = 2L;
        int nRolls;
        int nIters;
        int radius;
        double xMultiplier;
        double yMultiplier;
        int onlyFromMaximalComponent;
        boolean lessThan;
        double percentile;
        double[] orbit;
        // Apenas a metade superior é armazenada: [min(i,j)][max(i,j)-min(i,j)-1].
        double[][] dtws;
        double[][] graph;
        int phase;
        int i;
        int j;
    }

    private static final class Options {
        String inputPath;
        String checkpointPath = "grafo-por-anomalias.checkpoint";
        long pollMs = DEFAULT_POLL_MS;
        boolean azure = true;
        String[] positional;
    }

    public static void main(String[] args) throws Exception {
        Options options = parseOptions(args);
        checkpointPath = options.checkpointPath;
        Checkpoint state = readCheckpoint(options.checkpointPath);
        activeCheckpoint = state;

        Thread shutdownHook = new Thread(() -> saveQuietly(activeCheckpoint),
            "grafo-por-anomalias-checkpoint");
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
        if (positional.size() > 0 && positional.get(0).equals("devTesting")) {
            if (positional.size() < 2) throw new IllegalArgumentException(
                "devTesting exige o caminho do arquivo de entrada");
            options.inputPath = positional.get(1);
            positional = new java.util.ArrayList<>(positional.subList(2, positional.size()));
        }
        options.positional = positional.toArray(new String[0]);
        return options;
    }

    private static void run(Options options, Checkpoint state) throws Exception {
        if (state == null) {
            if (options.positional.length < 8) {
                throw new IllegalArgumentException("Uso: [devTesting arquivo] nRolls nIters " +
                    "radius xMultiplier yMultiplier onlyFromMaximalComponent lessThan percentile " +
                    "[--checkpoint=arquivo]");
            }
            state = loadInput(options);
            activeCheckpoint = state;
            save(state);
        } else {
            System.err.println("Retomando checkpoint " + checkpointPath +
                " na fase " + state.phase + " (i=" + state.i + ", j=" + state.j + ")");
        }

        java.util.ArrayList<Double> dtwValues = new java.util.ArrayList<>();
        for (int row = 0; row < state.nIters; row++) {
            for (int column = row + 1; column < state.nIters; column++) {
                if (get(state.dtws, row, column) < Double.POSITIVE_INFINITY) {
                    dtwValues.add(get(state.dtws, row, column));
                }
            }
        }
        if (state.phase <= 1) {
            state.phase = 1;
            for (state.i = Math.max(state.radius, state.i); state.i < state.nIters; state.i++) {
                for (state.j = Math.max(state.i + 2 * state.radius + 1, state.j);
                     state.j < state.nIters - state.radius; state.j++) {
                    if (stopRequested.get()) {
                        save(state);
                        return;
                    }
                    double value = dtwDistance(state.orbit, state.i - state.radius,
                        state.j - state.radius, 2 * state.radius + 1);

                    set(state.dtws, state.i, state.j, value);
                    if ((state.j & (CHECKPOINT_EVERY - 1)) == 0) save(state);
                }
                state.j = state.i + 2 * state.radius + 1;
            }
            state.i = 0;
            state.j = 1;
            state.phase = 2;
            save(state);
        }

        if (state.phase <= 2) {
            // Na execução inicial a fase 1 acabou de preencher os DTWs.
            if (dtwValues.isEmpty()) {
                for (int row = 0; row < state.nIters; row++) {
                    for (int column = row + 1; column < state.nIters; column++) {
                        double value = get(state.dtws, row, column);
                        if (value < Double.POSITIVE_INFINITY) dtwValues.add(value);
                    }
                }
            }
            state.phase = 2;
            double bound = percentile(dtwValues, state.percentile);
            for (state.i = Math.max(0, state.i); state.i < state.nIters - 1; state.i++) {
                for (state.j = Math.max(state.i + 1, state.j); state.j < state.nIters; state.j++) {
                    if (stopRequested.get()) {
                        save(state);
                        return;
                    }
                    double dtw = get(state.dtws, state.i, state.j);
                    boolean selected = dtw < Double.POSITIVE_INFINITY &&
                        (state.lessThan ? dtw < bound : dtw > bound);
                    if (selected) {
                        double dx = state.i * state.xMultiplier - state.j * state.xMultiplier;
                        double dy = state.orbit[state.i] * state.yMultiplier -
                            state.orbit[state.j] * state.yMultiplier;
                        set(state.graph, state.i, state.j, Math.sqrt(dx * dx + dy * dy));
                    }
                }
                state.j = state.i + 1;
            }
            state.phase = 3;
            save(state);
        }

        if (stopRequested.get()) return;
        writeGraph(state);
        deleteCheckpoint();
    }

    private static Checkpoint loadInput(Options options) throws IOException {
        String[] a = options.positional;
        Checkpoint state = new Checkpoint();
        state.nRolls = Integer.parseInt(a[0]);
        state.nIters = Integer.parseInt(a[1]);
        state.radius = Integer.parseInt(a[2]);
        state.xMultiplier = Double.parseDouble(a[3]);
        state.yMultiplier = Double.parseDouble(a[4]);
        state.onlyFromMaximalComponent = Integer.parseInt(a[5]);
        state.lessThan = Boolean.parseBoolean(a[6]);
        state.percentile = Double.parseDouble(a[7]);
        if (state.nRolls < 0 || state.nIters <= 0 || state.radius < 0 ||
            state.radius * 2 + 1 > state.nIters ||
            !Double.isFinite(state.xMultiplier) || !Double.isFinite(state.yMultiplier) ||
            !Double.isFinite(state.percentile) || state.percentile < 0.0 ||
            state.percentile > 100.0 ||
            (state.onlyFromMaximalComponent != 0 && state.onlyFromMaximalComponent != 1)) {
            throw new IllegalArgumentException("Parâmetros inválidos");
        }
        state.orbit = new double[state.nIters];
        InputReader input = options.inputPath == null
            ? new InputReader(System.in) : new InputReader(new FileInputStream(options.inputPath));
        for (int i = 0; i < state.nRolls; i++) input.nextDouble();
        for (int i = 0; i < state.nIters; i++) state.orbit[i] = input.nextDouble();
        state.dtws = triangularMatrix(state.nIters);
        state.graph = triangularMatrix(state.nIters);
        return state;
    }

    private static double percentile(java.util.List<Double> values, double percentile) {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY;
        java.util.ArrayList<Double> sorted = new java.util.ArrayList<>(values);
        java.util.Collections.sort(sorted);
        double position = (percentile / 100.0) * (sorted.size() + 1.0);
        if (position <= 1.0) return sorted.get(0);
        if (position >= sorted.size()) return sorted.get(sorted.size() - 1);
        int lower = (int) position - 1;
        double fraction = position - Math.floor(position);
        return sorted.get(lower) + fraction * (sorted.get(lower + 1) - sorted.get(lower));
    }

    private static double[][] triangularMatrix(int size) {
        double[][] matrix = new double[size][];
        for (int i = 0; i < size; i++) {
            matrix[i] = new double[size - i - 1];
            Arrays.fill(matrix[i], Double.POSITIVE_INFINITY);
        }
        return matrix;
    }

    private static double get(double[][] matrix, int i, int j) {
        if (i == j) return Double.POSITIVE_INFINITY;
        if (i > j) { int swap = i; i = j; j = swap; }
        return matrix[i][j - i - 1];
    }

    private static void set(double[][] matrix, int i, int j, double value) {
        if (i == j) return;
        if (i > j) { int swap = i; i = j; j = swap; }
        matrix[i][j - i - 1] = value;
    }

    // Mantém a mesma recorrência da implementação original, sem criar List<Double>.
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

    private static int dfs(double[][] graph, boolean[] visited, int[] component,
                           int rank, int vertex) {
        visited[vertex] = true;
        component[vertex] = rank;
        int count = 1;
        for (int next = 0; next < graph.length; next++) {
            if (!visited[next] && get(graph, vertex, next) < Double.POSITIVE_INFINITY) {
                count += dfs(graph, visited, component, rank, next);
            }
        }
        return count;
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

    private static synchronized void save(Checkpoint state) throws IOException {
        activeCheckpoint = state;
        String temporary = checkpointPath + ".tmp";
        try (ObjectOutputStream output = new ObjectOutputStream(new BufferedOutputStream(
                new FileOutputStream(temporary)))) {
            output.writeObject(state);
        }
        java.nio.file.Files.move(java.nio.file.Path.of(temporary),
            java.nio.file.Path.of(checkpointPath), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
            java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        System.err.println(Instant.now() + " checkpoint salvo: fase=" + state.phase +
            ", i=" + state.i + ", j=" + state.j);
    }

    private static Checkpoint readCheckpoint(String path) {
        try (ObjectInputStream input = new ObjectInputStream(new BufferedInputStream(
                new FileInputStream(path)))) {
            return (Checkpoint) input.readObject();
        } catch (IOException | ClassNotFoundException absent) {
            return null;
        }
    }

    private static void saveQuietly(Checkpoint state) {
        if (state != null && checkpointPath != null && state.phase < 3) {
            try { save(state); } catch (IOException error) {
                System.err.println("Nao foi possivel salvar checkpoint: " + error.getMessage());
            }
        }
    }

    private static void deleteCheckpoint() throws IOException {
        java.nio.file.Files.deleteIfExists(java.nio.file.Path.of(checkpointPath));
    }

    private static void writeGraph(Checkpoint state) {
        PrintWriter out = new PrintWriter(System.out);
        int[] component = new int[state.nIters];
        Arrays.fill(component, -1);
        boolean[] visited = new boolean[state.nIters];
        int maximalComponent = -1;
        int maxCount = 0;
        for (int vertex = 0; vertex < state.nIters; vertex++) {
            if (!visited[vertex]) {
                int count = dfs(state.graph, visited, component, vertex, vertex);
                if (count > maxCount) {
                    maxCount = count;
                    maximalComponent = vertex;
                }
            }
        }
        out.println("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        out.println("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">");
        out.println("<key id=\"d1\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>");
        out.println("<key id=\"d2\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>");
        out.println("<key id=\"d3\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>");
        out.println("<graph id=\"G\" edgedefault=\"undirected\">");
        for (int i = 0; i < state.nIters; i++) {
            out.println("<node id=\"n" + i + "\"><data key=\"d2\">" +
                (1000.0 * i / state.nIters) + "</data><data key=\"d3\">" +
                (1000.0 * state.orbit[i]) + "</data></node>");
        }
        long edge = 0;
        for (int i = 0; i < state.nIters; i++) for (int j = i + 1; j < state.nIters; j++) {
            if (get(state.graph, i, j) < Double.POSITIVE_INFINITY &&
                (state.onlyFromMaximalComponent != 1 || component[i] == maximalComponent)) {
                out.println("<edge id=\"e" + edge++ + "\" source=\"n" + i +
                    "\" target=\"n" + j + "\"><data key=\"d1\">" +
                    get(state.graph, i, j) + "</data></edge>");
            }
        }
        out.println("</graph></graphml>");
        out.flush();
    }

    private static final class InputReader {
        private final BufferedReader reader;
        private StringTokenizer tokenizer;
        InputReader(InputStream stream) { reader = new BufferedReader(new java.io.InputStreamReader(stream)); }
        String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) try {
                String line = reader.readLine();
                if (line == null) throw new IllegalStateException("Entrada terminou antes dos dados");
                tokenizer = new StringTokenizer(line);
            } catch (IOException error) { throw new RuntimeException(error); }
            return tokenizer.nextToken();
        }
        double nextDouble() { return Double.parseDouble(next()); }
    }
}