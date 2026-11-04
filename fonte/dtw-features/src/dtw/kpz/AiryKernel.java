package dtw.kpz;

/**
 * Determinante de Fredholm do núcleo de Airy em (s, ∞),
 * aproximado por Nyström com Gauss–Legendre composto.
 * F2(s) = det(I − K)_{L2(s,∞)}.
 */
public final class AiryKernel {
    private static final double[] LEG_X;
    private static final double[] LEG_W;
    private static final int GAUSS_N = 12;
    private static final double PANEL = 0.75;
    private static final double TCUT = 8.0;

    static {
        LEG_X = new double[GAUSS_N];
        LEG_W = new double[GAUSS_N];
        legendre(GAUSS_N, LEG_X, LEG_W);
    }

    private AiryKernel() {}

    /** log F2(s). Devolve 0 se s está além do corte (F2 indistinguível de 1 em double). */
    public static double logF2(double s) {
        if (s >= TCUT) {
            return 0.0;
        }
        double left = s;
        int panels = Math.max(1, (int) Math.ceil((TCUT - s) / PANEL));
        int m = panels * GAUSS_N;
        double[] x = new double[m];
        double[] w = new double[m];
        double[] ai = new double[m];
        double[] aip = new double[m];
        int p = 0;
        for (int panel = 0; panel < panels; panel++) {
            double a = left + panel * PANEL;
            double b = Math.min(a + PANEL, TCUT);
            if (panel == panels - 1) {
                b = TCUT;
            }
            double mid = 0.5 * (a + b);
            double half = 0.5 * (b - a);
            for (int g = 0; g < GAUSS_N; g++) {
                x[p] = mid + half * LEG_X[g];
                w[p] = half * LEG_W[g];
                ai[p] = Airy.ai(x[p]);
                aip[p] = Airy.aiPrime(x[p]);
                p++;
            }
        }
        m = p;
        double[][] a = new double[m][m];
        for (int i = 0; i < m; i++) {
            double sw = Math.sqrt(Math.max(w[i], 0.0));
            for (int j = 0; j <= i; j++) {
                double kij = kernel(x[i], x[j], ai[i], aip[i], ai[j], aip[j]);
                double scale = sw * Math.sqrt(Math.max(w[j], 0.0));
                double vij = scale * kij;
                a[i][j] = (i == j ? 1.0 : 0.0) - vij;
                if (i != j) {
                    a[j][i] = a[i][j];
                }
            }
        }
        return logDet(a, m);
    }

    private static double kernel(double x, double y, double aix, double aipx, double aiy, double aipy) {
        double d = x - y;
        if (Math.abs(d) < 1e-8 * (1.0 + Math.abs(x))) {
            return aipx * aipx - x * aix * aix;
        }
        return (aix * aipy - aipx * aiy) / d;
    }

    /** log|det| com sinal; se o sinal for negativo devolve NaN (quadratura inválida). */
    private static double logDet(double[][] a, int n) {
        double logAbs = 0.0;
        int sign = 1;
        for (int k = 0; k < n; k++) {
            int piv = k;
            double best = Math.abs(a[k][k]);
            for (int i = k + 1; i < n; i++) {
                double v = Math.abs(a[i][k]);
                if (v > best) {
                    best = v;
                    piv = i;
                }
            }
            if (best < 1e-300) {
                return Double.NEGATIVE_INFINITY;
            }
            if (piv != k) {
                double[] tmp = a[k];
                a[k] = a[piv];
                a[piv] = tmp;
                sign = -sign;
            }
            double diag = a[k][k];
            if (diag < 0.0) {
                sign = -sign;
            }
            logAbs += Math.log(Math.abs(diag));
            double inv = 1.0 / diag;
            for (int i = k + 1; i < n; i++) {
                double f = a[i][k] * inv;
                if (f == 0.0) {
                    continue;
                }
                for (int j = k + 1; j < n; j++) {
                    a[i][j] -= f * a[k][j];
                }
            }
        }
        if (sign < 0) {
            return Double.NaN;
        }
        return logAbs;
    }

    private static void legendre(int n, double[] x, double[] w) {
        for (int i = 0; i < (n + 1) / 2; i++) {
            double theta = Math.PI * (i + 0.75) / (n + 0.5);
            double xi = Math.cos(theta);
            double dp = 0.0;
            for (int iter = 0; iter < 30; iter++) {
                double p0 = 1.0;
                double p1 = xi;
                for (int k = 2; k <= n; k++) {
                    double p = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p0) / k;
                    p0 = p1;
                    p1 = p;
                }
                dp = n * (p0 - xi * p1) / (1.0 - xi * xi);
                double dx = p1 / dp;
                xi -= dx;
                if (Math.abs(dx) < 1e-15) {
                    break;
                }
            }
            x[i] = -xi;
            x[n - 1 - i] = xi;
            double wi = 2.0 / ((1.0 - xi * xi) * dp * dp);
            w[i] = wi;
            w[n - 1 - i] = wi;
        }
    }
}
