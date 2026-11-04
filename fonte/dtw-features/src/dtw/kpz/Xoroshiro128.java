package dtw.kpz;

/**
 * Gerador Xoroshiro128++ (Blackman & Vigna) com semente SplitMix64.
 * Fluxos independentes por réplica, sem estado compartilhado no laço quente.
 */
public final class Xoroshiro128 {
    private long s0;
    private long s1;
    private double nextGaussian;
    private boolean hasGaussian;

    public Xoroshiro128(long seed) {
        long z0 = splitmix(seed);
        long z1 = splitmix(seed + 0x9E3779B97F4A7C15L);
        if (z0 == 0L && z1 == 0L) {
            z0 = 0x9E3779B97F4A7C15L;
        }
        this.s0 = z0;
        this.s1 = z1;
    }

    public static long splitmix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Semente estável por (estudo, modelo, n, índice de réplica). */
    public static long mix(long seed, int model, int n, int replicate) {
        long z = seed;
        z ^= (long) model * 0xD1B54A32D192ED03L;
        z ^= (long) n * 0x94D049BB133111EBL;
        z ^= (long) replicate * 0x9E3779B97F4A7C15L;
        return splitmix(z);
    }

    public long nextLong() {
        long s0 = this.s0;
        long s1 = this.s1;
        long result = Long.rotateLeft(s0 + s1, 17) + s0;
        s1 ^= s0;
        this.s0 = Long.rotateLeft(s0, 49) ^ s1 ^ (s1 << 21);
        this.s1 = Long.rotateLeft(s1, 28);
        return result;
    }

    /** Uniforme em [0, 1). */
    public double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    public int nextInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound");
        }
        return (int) Long.remainderUnsigned(nextLong(), bound);
    }

    public double nextExponential() {
        long bits = nextLong() >>> 11;
        if (bits == 0L) {
            bits = 1L;
        }
        return -Math.log(bits * 0x1.0p-53);
    }

    public double nextGaussian() {
        if (hasGaussian) {
            hasGaussian = false;
            return nextGaussian;
        }
        double u;
        double v;
        double s;
        do {
            u = 2.0 * nextDouble() - 1.0;
            v = 2.0 * nextDouble() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        double m = Math.sqrt(-2.0 * Math.log(s) / s);
        nextGaussian = v * m;
        hasGaussian = true;
        return u * m;
    }
}
