package dtw.kpz;

/** Funções Airy Ai e Ai' com séries e assintóticas NIST. */
public final class Airy {
    private Airy() {}

    private static final double AI0  = 0.355028053887817239;
    private static final double AIP0 = -0.258819403792806798;

    public static double ai(double x) {
        if (x >= 5.6) {
            return aiPos(x);
        }
        if (x <= -6.6) {
            return aiNeg(-x);
        }
        return aiSeries(x);
    }

    public static double aiPrime(double x) {
        if (x >= 5.6) {
            return aiPrimePos(x);
        }
        if (x <= -6.6) {
            return aiPrimeNeg(-x);
        }
        return aiPrimeSeries(x);
    }

    private static double aiSeries(double x) {
        double x3 = x * x * x;
        double t1 = 1.0;
        double t2 = x;
        double s1 = 1.0;
        double s2 = 1.0;
        double sum1 = 1.0;
        double sum2 = x;
        for (int k = 1; k < 50; k++) {
            s1 *= x3 / (3.0 * k * (3.0 * k - 1.0));
            t1 += s1;
            s2 *= x3 / (3.0 * k * (3.0 * k + 1.0));
            t2 += x * s2;
            sum1 = t1;
            sum2 = t2;
            if (Math.abs(s1) < 1e-17 * Math.abs(t1) && Math.abs(s2) < 1e-17 * Math.abs(t2)) {
                break;
            }
        }
        return AI0 * sum1 + AIP0 * sum2;
    }

    /** f' e g' com as mesmas recorrências da série de Ai. */
    private static double aiPrimeSeries(double x) {
        if (Math.abs(x) < 1e-12) {
            return AIP0;
        }
        double x3 = x * x * x;
        double c = 1.0;
        double x3k = 1.0;
        double sumF = 0.0;
        for (int k = 1; k < 80; k++) {
            c /= 3.0 * k * (3.0 * k - 1.0);
            x3k *= x3;
            double term = 3.0 * k * c * x3k / x;
            sumF += term;
            if (Math.abs(term) < 1e-18 * (1.0 + Math.abs(sumF))) {
                break;
            }
        }
        double d = 1.0;
        double xp = 1.0;
        double sumG = 1.0;
        for (int k = 1; k < 80; k++) {
            d /= 3.0 * k * (3.0 * k + 1.0);
            xp *= x3;
            double term = (3.0 * k + 1.0) * d * xp;
            sumG += term;
            if (Math.abs(term) < 1e-18 * (1.0 + Math.abs(sumG))) {
                break;
            }
        }
        return AI0 * sumF + AIP0 * sumG;
    }

    private static double aiPos(double x) {
        double zeta = (2.0 / 3.0) * Math.pow(x, 1.5);
        double pre = Math.exp(-zeta) / (2.0 * Math.sqrt(Math.PI) * Math.pow(x, 0.25));
        double invZ = 1.0 / zeta;
        double zp = invZ;
        double c = 1.0;
        double sum = 1.0;
        for (int k = 1; k <= 12; k++) {
            c *= (6.0 * k - 1.0) * (6.0 * k - 5.0) / (72.0 * k);
            double term = c * zp;
            if ((k & 1) == 1) {
                sum -= term;
            } else {
                sum += term;
            }
            zp *= invZ;
            if (Math.abs(term) < 1e-16) {
                break;
            }
        }
        return pre * sum;
    }

    private static double aiPrimePos(double x) {
        double zeta = (2.0 / 3.0) * Math.pow(x, 1.5);
        double pre = -Math.exp(-zeta) * Math.pow(x, 0.25) / (2.0 * Math.sqrt(Math.PI));
        double invZ = 1.0 / zeta;
        double zp = invZ;
        double c = 1.0;
        double sum = 1.0;
        for (int k = 1; k <= 12; k++) {
            c *= (6.0 * k - 1.0) * (6.0 * k - 5.0) / (72.0 * k);
            double d = c * (6.0 * k + 1.0) / (6.0 * k - 1.0);
            double term = d * zp;
            if ((k & 1) == 1) {
                sum -= term;
            } else {
                sum += term;
            }
            zp *= invZ;
            if (Math.abs(term) < 1e-16) {
                break;
            }
        }
        return pre * sum;
    }

    private static double aiNeg(double x) {
        double zeta = (2.0 / 3.0) * Math.pow(x, 1.5);
        double pre = 1.0 / (Math.sqrt(Math.PI) * Math.pow(x, 0.25));
        double invZ = 1.0 / zeta;
        double zp = invZ;
        double c = 1.0;
        double p = 0.0;
        double q = 1.0;
        for (int k = 1; k <= 12; k++) {
            c *= (6.0 * k - 1.0) * (6.0 * k - 5.0) / (72.0 * k);
            double term = c * zp;
            if ((k & 1) == 1) {
                q -= term;
            } else {
                p += term;
            }
            zp *= invZ;
        }
        double theta = zeta + Math.PI / 4.0;
        return pre * (Math.sin(theta) * p - Math.cos(theta) * q);
    }

    private static double aiPrimeNeg(double x) {
        double zeta = (2.0 / 3.0) * Math.pow(x, 1.5);
        double pre = Math.pow(x, 0.25) / Math.sqrt(Math.PI);
        double invZ = 1.0 / zeta;
        double zp = invZ;
        double c = 1.0;
        double r = 0.0;
        double s = 1.0;
        for (int k = 1; k <= 12; k++) {
            c *= (6.0 * k - 1.0) * (6.0 * k - 5.0) / (72.0 * k);
            double d = c * (6.0 * k + 1.0) / (6.0 * k - 1.0);
            double term = d * zp;
            if ((k & 1) == 1) {
                s -= term;
            } else {
                r += term;
            }
            zp *= invZ;
        }
        double theta = zeta + Math.PI / 4.0;
        return -pre * (Math.cos(theta) * r + Math.sin(theta) * s);
    }
}
