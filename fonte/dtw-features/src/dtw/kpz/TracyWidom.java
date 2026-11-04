package dtw.kpz;

import java.util.Locale;

/**
 * Tracy–Widom F2 (GUE) e F1 (GOE).
 * F2 é o determinante de Fredholm do núcleo de Airy (Nyström).
 * F1 usa F1 = exp(−½∫q) √F2, com q obtido por derivação de log F2.
 */
public final class TracyWidom {
    public static final double TW2_MEAN_REF = -1.771086807411;
    public static final double TW2_VAR_REF = 0.8131947928329;
    public static final double TW2_SKEW_REF = 0.224084203610;
    public static final double TW1_MEAN_REF = -1.206533574582;
    public static final double TW1_VAR_REF = 1.607781034581;

    private final double tMin;
    private final double h;
    private final double[] f2;
    private final double[] logF2;
    private final double[] f1;
    private final double[] logF1;
    private final double mean2;
    private final double var2;
    private final double sd2;
    private final double skew2;
    private final double mean1;
    private final double var1;
    private final double sd1;
    private final boolean reliable;
    private final String diagnostic;

    private TracyWidom(double tMin, double h, double[] f2, double[] logF2, double[] f1, double[] logF1,
                       double mean2, double var2, double skew2, double mean1, double var1,
                       boolean reliable, String diagnostic) {
        this.tMin = tMin;
        this.h = h;
        this.f2 = f2;
        this.logF2 = logF2;
        this.f1 = f1;
        this.logF1 = logF1;
        this.mean2 = mean2;
        this.var2 = var2;
        this.sd2 = Math.sqrt(var2);
        this.skew2 = skew2;
        this.mean1 = mean1;
        this.var1 = var1;
        this.sd1 = Math.sqrt(Math.max(var1, 1e-12));
        this.reliable = reliable;
        this.diagnostic = diagnostic;
    }

    public static TracyWidom build() {
        double tMin = -8.0;
        double tMax = 6.0;
        double h = 0.025;
        int n = (int) Math.round((tMax - tMin) / h) + 1;
        double[] logF = new double[n];
        long t0 = System.nanoTime();
        java.util.stream.IntStream.range(0, n).parallel().forEach(i -> {
            double s = tMin + i * h;
            double lf = AiryKernel.logF2(s);
            if (Double.isNaN(lf)) {
                double x = Math.max(1e-6, -s);
                lf = -(x * x * x) / 12.0 - 0.125 * Math.log(x) - 0.136537;
            }
            logF[i] = lf;
        });
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            double fv = logF[i] < -700.0 ? 0.0 : Math.exp(logF[i]);
            if (fv > 1.0) {
                fv = 1.0;
            }
            if (i > 0 && fv < f[i - 1]) {
                fv = f[i - 1];
                logF[i] = fv <= 0.0 ? -700.0 : Math.log(fv);
            }
            f[i] = fv;
        }
        double[] m2 = moments(tMin, h, f);
        double[] f1 = new double[n];
        double[] logF1 = buildF1(h, logF, f1);
        double[] m1 = moments(tMin, h, f1);
        double seconds = (System.nanoTime() - t0) / 1e9;
        boolean ok = Math.abs(m2[0] - TW2_MEAN_REF) < 0.005
                && Math.abs(m2[1] - TW2_VAR_REF) < 0.005
                && Math.abs(m2[2] - TW2_SKEW_REF) < 0.01
                && Math.abs(m1[0] - TW1_MEAN_REF) < 0.02
                && Math.abs(m1[1] - TW1_VAR_REF) < 0.02;
        String diag = String.format(Locale.US,
                "Fredholm %.2fs  TW2 media=%.6f var=%.6f assim=%.4f  TW1 media=%.6f var=%.6f",
                seconds, m2[0], m2[1], m2[2], m1[0], m1[1]);
        return new TracyWidom(tMin, h, f, logF, f1, logF1, m2[0], m2[1], m2[2], m1[0], m1[1], ok, diag);
    }

    private static double[] buildF1(double h, double[] logF, double[] f1Out) {
        int n = logF.length;
        double[] u = new double[n];
        for (int i = 1; i < n - 1; i++) {
            u[i] = (logF[i + 1] - logF[i - 1]) / (2.0 * h);
        }
        u[0] = u[1];
        u[n - 1] = u[n - 2];
        for (int pass = 0; pass < 2; pass++) {
            double[] sm = u.clone();
            for (int i = 1; i < n - 1; i++) {
                sm[i] = 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1];
            }
            u = sm;
        }
        double[] q = new double[n];
        for (int i = 1; i < n - 1; i++) {
            double q2 = -(u[i + 1] - u[i - 1]) / (2.0 * h);
            q[i] = q2 > 0.0 ? Math.sqrt(q2) : 0.0;
        }
        q[0] = q[1];
        double[] r = new double[n];
        for (int i = n - 2; i >= 0; i--) {
            r[i] = r[i + 1] + 0.5 * (q[i] + q[i + 1]) * h;
        }
        double[] logF1 = new double[n];
        double prev = 0.0;
        for (int i = 0; i < n; i++) {
            double lf = -0.5 * r[i] + 0.5 * logF[i];
            double fv = lf < -700.0 ? 0.0 : Math.exp(lf);
            if (fv > 1.0) {
                fv = 1.0;
            }
            if (fv < prev) {
                fv = prev;
            }
            prev = fv;
            f1Out[i] = fv;
            logF1[i] = fv <= 0.0 ? -700.0 : Math.log(fv);
        }
        return logF1;
    }

    private static double[] moments(double tMin, double h, double[] f) {
        int n = f.length;
        double mean = 0.0;
        double second = 0.0;
        double third = 0.0;
        double prevF = f[0];
        double prevT = tMin;
        for (int i = 1; i < n; i++) {
            double t = tMin + i * h;
            double df = f[i] - prevF;
            if (df < 0.0) {
                df = 0.0;
            }
            double mid = 0.5 * (prevT + t);
            mean += mid * df;
            second += mid * mid * df;
            third += mid * mid * mid * df;
            prevF = f[i];
            prevT = t;
        }
        double tail = Math.max(0.0, 1.0 - prevF);
        mean += prevT * tail;
        second += prevT * prevT * tail;
        third += prevT * prevT * prevT * tail;
        double var = Math.max(0.0, second - mean * mean);
        double sd = Math.sqrt(var);
        double m3 = third - 3.0 * mean * second + 2.0 * mean * mean * mean;
        double skew = sd > 0.0 ? m3 / (sd * sd * sd) : Double.NaN;
        return new double[] {mean, var, skew};
    }

    private double interp(double[] a, double s) {
        if (s <= tMin) {
            return a[0];
        }
        double tmax = tMin + (a.length - 1) * h;
        if (s >= tmax) {
            return a[a.length - 1];
        }
        double pos = (s - tMin) / h;
        int i = (int) pos;
        double frac = pos - i;
        return a[i] * (1.0 - frac) + a[i + 1] * frac;
    }

    public double cdf2(double s) {
        return s < tMin ? 0.0 : interp(f2, s);
    }

    public double logCdf2(double s) {
        if (s < tMin) {
            double x = -s;
            return -(x * x * x) / 12.0;
        }
        return interp(logF2, s);
    }

    public double survival2(double s) {
        double sf = 1.0 - cdf2(s);
        return sf < 0.0 ? 0.0 : sf;
    }

    public double logSurvival2(double s) {
        double lf = logCdf2(s);
        if (lf > -1e-6) {
            double sf = -Math.expm1(lf);
            return sf <= 0.0 ? -700.0 : Math.log(sf);
        }
        double f = Math.exp(lf);
        if (f >= 1.0) {
            return -700.0;
        }
        return Math.log1p(-f);
    }

    public double cdf1(double s) {
        return s < tMin ? 0.0 : interp(f1, s);
    }

    public double logCdf1(double s) {
        return s < tMin ? -700.0 : interp(logF1, s);
    }

    public double survival1(double s) {
        double f = cdf1(s);
        return f >= 1.0 ? 0.0 : 1.0 - f;
    }

    public double logSurvival1(double s) {
        double f = cdf1(s);
        return f >= 1.0 ? -700.0 : Math.log1p(-f);
    }

    public double quantile2(double u) {
        return quantile(u, f2);
    }

    public double quantile1(double u) {
        return quantile(u, f1);
    }

    private double quantile(double u, double[] f) {
        if (u <= f[0]) {
            return tMin;
        }
        if (u >= f[f.length - 1]) {
            return tMin + (f.length - 1) * h;
        }
        int lo = 0;
        int hi = f.length - 1;
        while (lo + 1 < hi) {
            int mid = (lo + hi) >>> 1;
            if (f[mid] < u) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double denom = f[hi] - f[lo];
        double w = denom <= 0.0 ? 0.0 : (u - f[lo]) / denom;
        if (w < 0.0) {
            w = 0.0;
        } else if (w > 1.0) {
            w = 1.0;
        }
        return tMin + (lo + w) * h;
    }

    public double mean2() { return mean2; }
    public double variance2() { return var2; }
    public double sd2() { return sd2; }
    public double skewness2() { return skew2; }
    public double mean1() { return mean1; }
    public double variance1() { return var1; }
    public double sd1() { return sd1; }
    public double leftRelError() { return Math.abs(var2 - TW2_VAR_REF); }
    public boolean reliable() { return reliable; }
    public String diagnostic() { return diagnostic; }

    public double cdf2Studentized(double z) {
        return cdf2(mean2 + z * sd2);
    }

    public double survival2Studentized(double z) {
        return survival2(mean2 + z * sd2);
    }

    public double quantile2Studentized(double u) {
        return (quantile2(u) - mean2) / sd2;
    }

    public double quantile2ReflectedStudentized(double u) {
        return -quantile2Studentized(1.0 - u);
    }

    public double cdf2ReflectedStudentized(double z) {
        return survival2Studentized(-z);
    }

    public double survival2ReflectedStudentized(double z) {
        return cdf2Studentized(-z);
    }

    public double quantile1Studentized(double u) {
        return (quantile1(u) - mean1) / sd1;
    }

    public double sample2(Xoroshiro128 rng) {
        double u = rng.nextDouble();
        if (u < 1e-12) {
            u = 1e-12;
        } else if (u > 1.0 - 1e-12) {
            u = 1.0 - 1e-12;
        }
        return quantile2(u);
    }
}
