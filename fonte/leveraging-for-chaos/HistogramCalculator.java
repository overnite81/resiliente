import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;
import java.util.Scanner;

public class HistogramCalculator {
    private static final class DensityIndex {

        final int density, position;
        DensityIndex(int density, int position) { this.density = density; this.position = position; }
    }
    public static void main(String[] args) throws Exception {

        if (args.length < 10) throw new IllegalArgumentException("Uso: java HistogramCalculator <arquivo> <n> <m> <lower> <upper> <eps> <left> <right> <indexStart> <indexEnd>");
        File file = new File(args[0]);


        int n = Integer.parseInt(args[1]); // n = número de linhas
        int m = Integer.parseInt(args[2]); // m = número de colunas
        double lower = Double.parseDouble(args[3]), upper = Double.parseDouble(args[4]), eps = Double.parseDouble(args[5]);
        int left = Integer.parseInt(args[6]), right = Integer.parseInt(args[7]);
        int indexStart = Integer.parseInt(args[8]), indexEnd = Integer.parseInt(args[9]);
        validate(m, n, lower, upper, eps, left, right, indexStart, indexEnd);
        double[] flat = readAllFromFile(file);



        if (flat.length != (long) m * n) throw new IllegalArgumentException("Arquivo contem " + flat.length + " valores; esperado " + ((long) m * n));
        double[][] matrix = new double[n][m];
        for (int row = 0; row < n; row++) System.arraycopy(flat, row * m, matrix[row], 0, m);

        int[] histogram = new int[m];



        for (int row = 0; row < n; row++) processRow(matrix[row], histogram, lower, upper, eps, left, right, indexStart, indexEnd);
        printCsv(histogram);
    }



    private static void validate(int m, int n, double lower, double upper, double eps, int left, int right, int start, int end) {
        if (m <= 0 || n <= 0) throw new IllegalArgumentException("m e n devem ser maiores que zero");
        if (!Double.isFinite(lower) || !Double.isFinite(upper) || !Double.isFinite(eps)) throw new IllegalArgumentException("parametros reais devem ser finitos");




        if (lower > upper || eps < 0 || left < 0 || right > m || left > right || start < 0 || end > m || start >= end) throw new IllegalArgumentException("parametros invalidos");
    }



    private static void processRow(double[] row, int[] histogram, double lower, double upper, double eps, int left, int right, int start, int end) {
        int m = row.length;

        // Passo 1: organiza a sequencia na reta real (ordena por valor).
        // A posicao resultante e o "indice" pedido: cada posicao e um
        // ponto-centro do calculo de densidade e a coordenada do histograma.
        double[] sorted = row.clone();
        Arrays.sort(sorted);

        // Passo 2: densidade de cada ponto-centro (posicao na reta real).
        // Valores fora de [lower, upper] recebem densidade 0, mas continuam
        // contando como vizinhos dos valores dentro da faixa.
        DensityIndex[] pairs = new DensityIndex[m];
        for (int position = 0; position < m; position++) {
            double value = sorted[position];
            int density = (value >= lower && value <= upper)
                ? countInWindow(sorted, value, eps)
                : 0;
            pairs[position] = new DensityIndex(density, position);
        }

        // Passo 3: ordena os pontos-centro por densidade DECRESCENTE (mais densos
        // primeiro), com a posicao de carona em ordem crescente. Assim os pontos
        // das pontas (menos densos) ficam nos ultimos ranks e so entram na faixa
        // [left, right) quando right for grande o suficiente.
        Arrays.sort(pairs, Comparator
            .comparingInt((DensityIndex p) -> p.density).reversed()
            .thenComparingInt(p -> p.position));

        // Passo 4: aplica o intervalo de ranks e filtra pela posicao na reta real.
        // As posicoes mais a esquerda/direita (fora de [start, end)) ficam de fora,
        // o que e o comportamento esperado.
        for (int rank = left; rank < right; rank++) {
            int position = pairs[rank].position;
            if (position >= start && position < end) histogram[position]++;
        }
    }

    /** Conta quantos valores ordenados caem em [value - eps, value + eps]. */
    private static int countInWindow(double[] sorted, double value, double eps) {
        return upperBound(sorted, value + eps) - lowerBound(sorted, value - eps);
    }

    /** Primeiro indice i tal que sorted[i] >= target. */
    private static int lowerBound(double[] sorted, double target) {
        int lo = 0, hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] < target) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    /** Primeiro indice i tal que sorted[i] > target (limite superior exclusivo). */
    private static int upperBound(double[] sorted, double target) {
        int lo = 0, hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= target) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private static double[] readAllFromFile(File file) throws Exception {
        java.util.ArrayList<Double> values = new java.util.ArrayList<>();
        try (Scanner sc = new Scanner(file).useLocale(Locale.US)) {
            sc.useDelimiter("[,\\s]+");
            while (sc.hasNext()) {
                String token = sc.next();
                try {
                    double value = Double.parseDouble(token);
                    if (!Double.isFinite(value)) throw new IllegalArgumentException("valor nao finito no arquivo: " + token);
                    values.add(value);





                } catch (NumberFormatException e) { throw new IllegalArgumentException("valor nao numerico no arquivo: " + token, e); }
            }
        }
        double[] result = new double[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }

    private static void printCsv(int[] values) {
        StringBuilder sb = new StringBuilder();




        for (int i = 0; i < values.length; i++) { if (i > 0) sb.append(','); sb.append(values[i]); }
        System.out.println(sb);
    }
}
