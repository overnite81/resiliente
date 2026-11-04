package dtw.kpz;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/** Estatística empírica sem alocação desnecessária no caminho de agregação. */
public final class EmpiricalStats {
    private EmpiricalStats() {}

    public static double mean(double[] x) {
        double s = 0.0;
        for (int i = 0; i < x.length; i++) {
            s += x[i];
        }
        return s / x.length;
    }

    public static double variance(double[] x) {
        double m = mean(x);
        double s = 0.0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - m;
            s += d * d;
        }
        return s / (x.length - 1);
    }

    public static double sd(double[] x) {
        return Math.sqrt(variance(x));
    }

    /** Assimetria amostral G1 (não enviesada). */
    public static double skewness(double[] x) {
        int n = x.length;
        double m = mean(x);
        double m2 = 0.0;
        double m3 = 0.0;
        for (int i = 0; i < n; i++) {
            double d = x[i] - m;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n;
        m3 /= n;
        if (m2 <= 0.0) {
            return 0.0;
        }
        double g1 = m3 / Math.pow(m2, 1.5);
        return Math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1;
    }

    public static double excessKurtosis(double[] x) {
        int n = x.length;
        double m = mean(x);
        double m2 = 0.0;
        double m4 = 0.0;
        for (int i = 0; i < n; i++) {
            double d = x[i] - m;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;
        if (m2 <= 0.0) {
            return 0.0;
        }
        double g2 = m4 / (m2 * m2) - 3.0;
        return ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0);
    }

    public static double wasserstein1Sorted(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("amostras de tamanhos diferentes");
        }
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s / a.length;
    }

    /**
     * Anderson–Darling para distribuição totalmente especificada.
     * {@code cdf} e {@code sf} devem ser a mesma lei (sf = 1−F, estável na cauda direita).
     */
    public static double andersonDarling(double[] sorted, DoubleUnaryOperator cdf, DoubleUnaryOperator sf) {
        int n = sorted.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double left = cdf.applyAsDouble(sorted[i]);
            double right = sf.applyAsDouble(sorted[n - 1 - i]);
            if (left < 1e-300) {
                left = 1e-300;
            }
            if (right < 1e-300) {
                right = 1e-300;
            }
            if (left > 1.0) {
                left = 1.0;
            }
            sum += (2.0 * (i + 1) - 1.0) * (Math.log(left) + Math.log(right));
        }
        return -n - sum / n;
    }

    /** OLS ponderado y = a + b x. Devolve {a, b}. */
    public static double[] wlsSlope(double[] x, double[] y, double[] w) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < x.length; i++) {
            double wi = w == null ? 1.0 : w[i];
            sw += wi;
            sx += wi * x[i];
            sy += wi * y[i];
            sxx += wi * x[i] * x[i];
            sxy += wi * x[i] * y[i];
        }
        double den = sw * sxx - sx * sx;
        if (Math.abs(den) < 1e-18) {
            return new double[] {sy / sw, Double.NaN};
        }
        double b = (sw * sxy - sx * sy) / den;
        double a = (sy - b * sx) / sw;
        return new double[] {a, b};
    }

    /** y = b0 + b1 x1 + b2 x2, pesos w. Devolve {b0,b1,b2}. */
    public static double[] wls3(double[] x1, double[] x2, double[] y, double[] w) {
        double s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
        double t0 = 0, t1 = 0, t2 = 0;
        for (int i = 0; i < y.length; i++) {
            double wi = w == null ? 1.0 : w[i];
            double a = 1.0;
            double b = x1[i];
            double c = x2[i];
            s00 += wi * a * a;
            s01 += wi * a * b;
            s02 += wi * a * c;
            s11 += wi * b * b;
            s12 += wi * b * c;
            s22 += wi * c * c;
            t0 += wi * a * y[i];
            t1 += wi * b * y[i];
            t2 += wi * c * y[i];
        }
        double[][] m = {
                {s00, s01, s02, t0},
                {s01, s11, s12, t1},
                {s02, s12, s22, t2}
        };
        for (int col = 0; col < 3; col++) {
            int piv = col;
            for (int r = col + 1; r < 3; r++) {
                if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) {
                    piv = r;
                }
            }
            double[] tmp = m[col];
            m[col] = m[piv];
            m[piv] = tmp;
            double div = m[col][col];
            if (Math.abs(div) < 1e-18) {
                return new double[] {Double.NaN, Double.NaN, Double.NaN};
            }
            for (int c = col; c < 4; c++) {
                m[col][c] /= div;
            }
            for (int r = 0; r < 3; r++) {
                if (r == col) {
                    continue;
                }
                double f = m[r][col];
                for (int c = col; c < 4; c++) {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
        return new double[] {m[0][3], m[1][3], m[2][3]};
    }

    public static double[] copy(double[] x) {
        return Arrays.copyOf(x, x.length);
    }

    public static double[] sortedCopy(double[] x) {
        double[] c = Arrays.copyOf(x, x.length);
        Arrays.sort(c);
        return c;
    }

    public static void studentize(double[] src, double[] dst, double mean, double sd) {
        double inv = 1.0 / sd;
        for (int i = 0; i < src.length; i++) {
            dst[i] = (src[i] - mean) * inv;
        }
    }

    public static double percentile(double[] sorted, double p) {
        if (p <= 0.0) {
            return sorted[0];
        }
        if (p >= 1.0) {
            return sorted[sorted.length - 1];
        }
        double pos = p * (sorted.length - 1);
        int i = (int) Math.floor(pos);
        double f = pos - i;
        if (i >= sorted.length - 1) {
            return sorted[sorted.length - 1];
        }
        return sorted[i] * (1.0 - f) + sorted[i + 1] * f;
    }

    /** Quantil normal padrão, aproximação racional de Acklam. */
    public static double normalQuantile(double p) {
        if (p <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (p >= 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        final double a1 = -3.969683028665376e+01;
        final double a2 = 2.209460984245205e+02;
        final double a3 = -2.759285104469687e+02;
        final double a4 = 1.383577518672690e+02;
        final double a5 = -3.066479806614716e+01;
        final double a6 = 2.506628277459239e+00;
        final double b1 = -5.447609879822406e+01;
        final double b2 = 1.615858368580409e+02;
        final double b3 = -1.556989798598866e+02;
        final double b4 = 6.680131188771972e+01;
        final double b5 = -1.328068155288572e+01;
        final double c1 = -7.784894002430293e-03;
        final double c2 = -3.223964580411365e-01;
        final double c3 = -2.400758277161838e+00;
        final double c4 = -2.549732539343734e+00;
        final double c5 = 4.374664141464968e+00;
        final double c6 = 2.938163982698783e+00;
        final double d1 = 7.784695709041462e-03;
        final double d2 = 3.224671290700398e-01;
        final double d3 = 2.445134137142996e+00;
        final double d4 = 3.754408661907416e+00;
        final double plow = 0.02425;
        double q;
        if (p < plow) {
            q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        if (p > 1.0 - plow) {
            q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        q = p - 0.5;
        double r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }

    /**
     * W1 entre leis studentizadas, por integração de quantis em (0,1).
     * qRaw devolve o quantil da lei bruta; m,s são média e desvio populacionais.
     */
    public static double wassersteinStudentized(DoubleUnaryOperator qA, double mA, double sA,
                                                DoubleUnaryOperator qB, double mB, double sB,
                                                int knots) {
        double acc = 0.0;
        for (int i = 1; i < knots; i++) {
            double u = (i + 0.5) / knots;
            double a = (qA.applyAsDouble(u) - mA) / sA;
            double b = (qB.applyAsDouble(u) - mB) / sB;
            acc += Math.abs(a - b);
        }
        return acc / (knots - 1);
    }
}
