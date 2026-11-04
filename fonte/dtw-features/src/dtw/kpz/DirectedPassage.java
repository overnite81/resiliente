
package dtw.kpz;
/**
 * Núcleos O(n²) de passagem dirigida.
 * Buffers reutilizados por thread. Predecessores em byte só se o geodésico for pedido.
 *
 * Convenção de predecessor: 0 = esquerda, 1 = acima, 2 = diagonal.
 * Fronteira ponto a ponto: célula (0,0) é só o custo local (D(−1,−1) = 0 e o resto da borda é infinito).
 */
public final class DirectedPassage {
    public enum Model {
        LPP(0),
        LPP3(3),
        FPP3(1),
        DTW(2);

        public final int id;
        Model(int id) { this.id = id; }
    }

    public enum Cost {
        ABS, SQUARE
    }

    private static final class Buf {
        double[] prev = new double[0];
        double[] curr = new double[0];
        double[] x = new double[0];
        double[] y = new double[0];
        byte[] pred = new byte[0];

        void ensure(int n, boolean geodesic) {
            if (prev.length < n) {
                prev = new double[n];
                curr = new double[n];
            }
            if (x.length < n) {
                x = new double[n];
                y = new double[n];
            }
            int need = geodesic ? n * n : 0;
            if (need > 0 && pred.length < need) {
                pred = new byte[need];
            }
        }
    }

    private static final ThreadLocal<Buf> BUFS = ThreadLocal.withInitial(Buf::new);

    private DirectedPassage() {}

    public static final class Outcome {
        public final double value;
        public final int midColumn;
        public final double mX;
        public final double mY;
        public final double w1xy;
        public final double emparelhado;

        public Outcome(double value, int midColumn) {
            this(value, midColumn, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
        }

        private Outcome(double value, int midColumn, double mX, double mY, double w1xy, double emparelhado) {
            this.value = value;
            this.midColumn = midColumn;
            this.mX = mX;
            this.mY = mY;
            this.w1xy = w1xy;
            this.emparelhado = emparelhado;
        }
    }

    public static Outcome simulate(Model model, int n, long seed, int replicate,
                                   boolean geodesic, Cost cost,
                                   Distribution weightDist, Distribution seqDist) {
        if (n < 2) {
            throw new IllegalArgumentException("n >= 2");
        }
        Buf buf = BUFS.get();
        buf.ensure(n, geodesic);
        Xoroshiro128 rng = new Xoroshiro128(Xoroshiro128.mix(seed, model.id, n, replicate));
        byte[] pred = geodesic ? buf.pred : null;
        double value;
        double mX = Double.NaN;
        double mY = Double.NaN;
        double w1xy = Double.NaN;
        double emparelhado = Double.NaN;
        switch (model) {
            case LPP:
                value = lpp(n, rng, buf.prev, buf.curr, pred, weightDist);
                break;
            case LPP3:
                value = lpp3(n, rng, buf.prev, buf.curr, pred, weightDist);
                break;
            case FPP3:
                value = fpp3(n, rng, buf.prev, buf.curr, pred, weightDist);
                break;
            case DTW:
                fillSequence(buf.x, n, rng, seqDist);
                fillSequence(buf.y, n, rng, seqDist);
                value = dtw(n, buf.x, buf.y, buf.prev, buf.curr, pred, cost == Cost.ABS);
                mX = mean(buf.x, n);
                mY = mean(buf.y, n);
                emparelhado = pairedAbsMean(buf.x, buf.y, n);
                w1xy = sortedAbsMean(buf.x, buf.y, n);
                break;
            default:
                throw new IllegalArgumentException(model.name());
        }
        int mid = geodesic ? rightmostOnRow(pred, n, n / 2) : -1;
        return new Outcome(value, mid, mX, mY, w1xy, emparelhado);
    }

    private static double mean(double[] values, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += values[i];
        }
        return sum / n;
    }

    private static double pairedAbsMean(double[] x, double[] y, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += Math.abs(x[i] - y[i]);
        }
        return sum / n;
    }

    private static double sortedAbsMean(double[] x, double[] y, int n) {
        double[] sx = java.util.Arrays.copyOf(x, n);
        double[] sy = java.util.Arrays.copyOf(y, n);
        java.util.Arrays.sort(sx);
        java.util.Arrays.sort(sy);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += Math.abs(sx[i] - sy[i]);
        }
        return sum / n;
    }

    private static void fillSequence(double[] a, int n, Xoroshiro128 rng, Distribution dist) {
        for (int i = 0; i < n; i++) {
            a[i] = nextValue(dist, rng);
        }
    }

    private static double nextValue(Distribution dist, Xoroshiro128 rng) {
        switch (dist) {
            case EXPONENTIAL: return rng.nextExponential();
            case NORMAL:      return rng.nextGaussian();
            case UNIFORM:     return rng.nextDouble();
            case POISSON:     return nextPoisson1(rng);
            default:          return rng.nextExponential();
        }
    }

    private static double nextPoisson1(Xoroshiro128 rng) {
        double L = Math.exp(-1.0);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > L);
        return k - 1;
    }

    /** Última passagem clássica, passos direita/acima. */
    static double lpp(int n, Xoroshiro128 rng, double[] prev, double[] curr, byte[] pred, Distribution dist) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
            acc += nextValue(dist, rng);
            prev[j] = acc;
            if (pred != null) {
                pred[j] = (byte) (j == 0 ? 1 : 0);
            }
        }
        for (int i = 1; i < n; i++) {
            double w0 = nextValue(dist, rng);
            curr[0] = w0 + prev[0];
            int row = i * n;
            if (pred != null) {
                pred[row] = 1;
            }
            double left = curr[0];
            for (int j = 1; j < n; j++) {
                double w = nextValue(dist, rng);
                double up = prev[j];
                double val;
                byte which;
                if (left >= up) {
                    val = w + left;
                    which = 0;
                } else {
                    val = w + up;
                    which = 1;
                }
                curr[j] = val;
                left = val;
                if (pred != null) {
                    pred[row + j] = which;
                }
            }
            double[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[n - 1];
    }

    /** Última passagem com 3 vizinhos (max + diagonal). */
    static double lpp3(int n, Xoroshiro128 rng, double[] prev, double[] curr, byte[] pred, Distribution dist) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
            acc += nextValue(dist, rng);
            prev[j] = acc;
            if (pred != null) {
                pred[j] = (byte) (j == 0 ? 2 : 0);
            }
        }
        for (int i = 1; i < n; i++) {
            int row = i * n;
            curr[0] = nextValue(dist, rng) + prev[0];
            if (pred != null) {
                pred[row] = 1;
            }
            double leftVal = curr[0];
            for (int j = 1; j < n; j++) {
                double w = nextValue(dist, rng);
                double fromLeft = leftVal;
                double fromUp = prev[j];
                double fromDiag = prev[j - 1];
                double best = fromDiag;
                byte which = 2;
                if (fromLeft > best) {
                    best = fromLeft;
                    which = 0;
                }
                if (fromUp > best) {
                    best = fromUp;
                    which = 1;
                }
                double val = w + best;
                curr[j] = val;
                leftVal = val;
                if (pred != null) {
                    pred[row + j] = which;
                }
            }
            double[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[n - 1];
    }

    /** Primeira passagem com diagonal. Controle estrutural do DTW. */
    static double fpp3(int n, Xoroshiro128 rng, double[] prev, double[] curr, byte[] pred, Distribution dist) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
            acc += nextValue(dist, rng);
            prev[j] = acc;
            if (pred != null) {
                pred[j] = (byte) (j == 0 ? 2 : 0);
            }
        }
        for (int i = 1; i < n; i++) {
            int row = i * n;
            curr[0] = nextValue(dist, rng) + prev[0];
            if (pred != null) {
                pred[row] = 1;
            }
            double leftVal = curr[0];
            for (int j = 1; j < n; j++) {
                double c = nextValue(dist, rng);
                double fromLeft = leftVal;
                double fromUp = prev[j];
                double fromDiag = prev[j - 1];
                double best = fromDiag;
                byte which = 2;
                if (fromLeft < best) {
                    best = fromLeft;
                    which = 0;
                }
                if (fromUp < best) {
                    best = fromUp;
                    which = 1;
                }
                double val = c + best;
                curr[j] = val;
                leftVal = val;
                if (pred != null) {
                    pred[row + j] = which;
                }
            }
            double[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[n - 1];
    }

    /** DTW clássico quadrado. Custo |Xi−Yj| ou (Xi−Yj)², passos (1,0),(0,1),(1,1). */
    static double dtw(int n, double[] x, double[] y, double[] prev, double[] curr,
                       byte[] pred, boolean absCost) {
        double acc = 0.0;
        double x0 = x[0];
        for (int j = 0; j < n; j++) {
            double d = x0 - y[j];
            acc += absCost ? Math.abs(d) : d * d;
            prev[j] = acc;
            if (pred != null) {
                pred[j] = (byte) (j == 0 ? 2 : 0);
            }
        }
        for (int i = 1; i < n; i++) {
            int row = i * n;
            double xi = x[i];
            double d0 = xi - y[0];
            curr[0] = (absCost ? Math.abs(d0) : d0 * d0) + prev[0];
            if (pred != null) {
                pred[row] = 1;
            }
            double leftVal = curr[0];
            for (int j = 1; j < n; j++) {
                double d = xi - y[j];
                double c = absCost ? Math.abs(d) : d * d;
                double fromLeft = leftVal;
                double fromUp = prev[j];
                double fromDiag = prev[j - 1];
                double best = fromDiag;
                byte which = 2;
                if (fromLeft < best) {
                    best = fromLeft;
                    which = 0;
                }
                if (fromUp < best) {
                    best = fromUp;
                    which = 1;
                }
                double val = c + best;
                curr[j] = val;
                leftVal = val;
                if (pred != null) {
                    pred[row + j] = which;
                }
            }
            double[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[n - 1];
    }

    /** Coluna mais à direita do geodésico na linha {@code row}. */
    static int rightmostOnRow(byte[] pred, int n, int row) {
        int i = n - 1;
        int j = n - 1;
        int rightmost = -1;
        while (i >= 0 && j >= 0) {
            if (i == row) {
                if (j > rightmost) {
                    rightmost = j;
                }
            }
            if (i == 0 && j == 0) {
                break;
            }
            byte p = pred[i * n + j];
            if (p == 0) {
                j--;
            } else if (p == 1) {
                i--;
            } else {
                i--;
                j--;
            }
        }
        return rightmost;
    }
}
