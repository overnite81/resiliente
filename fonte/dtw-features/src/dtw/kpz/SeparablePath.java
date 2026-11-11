package dtw.kpz;

import java.util.Arrays;

/** Núcleos adicionais isolados para os estudos mecanísticos; não altera DirectedPassage. */
final class SeparablePath {
    private SeparablePath() {}

    static final class Metrics {
        final double cost, diagonal, gain, rmsOffset, meanVisits;
        final int maxOffset, midColumn;
        Metrics(double cost, double diagonal, double rmsOffset, int maxOffset,
                double meanVisits, int midColumn) {
            this.cost = cost;
            this.diagonal = diagonal;
            this.gain = diagonal - cost;
            this.rmsOffset = rmsOffset;
            this.maxOffset = maxOffset;
            this.meanVisits = meanVisits;
            this.midColumn = midColumn;
        }
    }

    static Metrics dtw(double[] x, double[] y, double[] prev, double[] curr,
                       byte[] pred, boolean keepPath) {
        return dtw(x, y, prev, curr, pred, false, keepPath);
    }

    static Metrics dtw(double[] x, double[] y, double[] prev, double[] curr,
                       byte[] pred, boolean maximize, boolean keepPath) {
        int n = x.length;
        double diag = 0;
        for (int i = 0; i < n; i++) diag += Math.abs(x[i] - y[i]);
        double value = DirectedPassage.dtw(n, x, y, prev, curr, pred, true, maximize);
        return metrics(value, diag, pred, n, keepPath);
    }

    static Metrics mixed(double[] x, double[] y, double[] omega, double lambda,
                         double[] prev, double[] curr, byte[] pred,
                         boolean minimum, boolean keepPath) {
        int n = x.length;
        if (omega.length != n * n) throw new IllegalArgumentException("omega precisa ter n*n entradas");
        for (int j = 0; j < n; j++) {
            double local = local(x[0], y[j], omega[j], lambda);
            prev[j] = local + (j == 0 ? 0 : prev[j - 1]);
            if (pred != null) pred[j] = (byte)(j == 0 ? 2 : 0);
        }
        for (int i = 1; i < n; i++) {
            int row = i * n;
            curr[0] = local(x[i], y[0], omega[row], lambda) + prev[0];
            if (pred != null) pred[row] = 1;
            for (int j = 1; j < n; j++) {
                double left = curr[j - 1], up = prev[j], diagonal = prev[j - 1];
                byte p = 2;
                double best = diagonal;
                if (minimum ? left < best : left > best) { best = left; p = 0; }
                if (minimum ? up < best : up > best) { best = up; p = 1; }
                curr[j] = local(x[i], y[j], omega[row + j], lambda) + best;
                if (pred != null) pred[row + j] = p;
            }
            double[] t = prev; prev = curr; curr = t;
        }
        double diag = 0;
        for (int i = 0; i < n; i++) diag += Math.abs(x[i] - y[i]);
        return metrics(prev[n - 1], diag, pred, n, keepPath);
    }

    static double local(double x, double y, double omega, double lambda) {
        return (1.0 - lambda) * omega + lambda * Math.abs(x - y);
    }

    static Metrics metrics(double value, double diagonal, byte[] pred, int n, boolean keepPath) {
        if (!keepPath || pred == null) return new Metrics(value, diagonal,
                Double.NaN, -1, Double.NaN, -1);
        int i = n - 1, j = n - 1, mid = -1, max = 0, steps = 0;
        double sq = 0;
        int[] visits = new int[n];
        while (true) {
            int off = Math.abs(i - j);
            sq += (double)off * off;
            max = Math.max(max, off);
            visits[i]++;
            steps++;
            if (i == n / 2 && j > mid) mid = j;
            if (i == 0 && j == 0) break;
            byte p = pred[i * n + j];
            if (p == 0) j--;
            else if (p == 1) i--;
            else { i--; j--; }
        }
        int nonzero = 0;
        for (int v : visits) if (v != 0) nonzero++;
        return new Metrics(value, diagonal, Math.sqrt(sq / steps), max,
                (double)steps / n, mid);
    }

    static double[] sequence(int n, Xoroshiro128 rng, Distribution dist) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = draw(rng, dist);
        return a;
    }

    static double draw(Xoroshiro128 rng, Distribution dist) {
        switch (dist) {
            case NORMAL: return rng.nextGaussian();
            case UNIFORM: return rng.nextDouble();
            case POISSON: return poisson1(rng);
            case EXPONENTIAL: default: return rng.nextExponential();
        }
    }

    private static int poisson1(Xoroshiro128 rng) {
        double p = 1, limit = Math.exp(-1); int k = 0;
        do { k++; p *= rng.nextDouble(); } while (p > limit);
        return k - 1;
    }

    static double[] ranks(double[] x) {
        Integer[] order = new Integer[x.length];
        for (int i = 0; i < x.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(x[a], x[b]));
        double[] r = new double[x.length];
        for (int k = 0; k < order.length; k++) r[order[k]] = k / (double)Math.max(1, x.length - 1);
        return r;
    }
}
