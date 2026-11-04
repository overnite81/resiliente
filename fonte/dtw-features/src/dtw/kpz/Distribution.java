package dtw.kpz;

/** Distribuições suportadas para pesos e sequências. */
public enum Distribution {
    EXPONENTIAL,   // Exp(1)
    NORMAL,        // N(0,1)
    POISSON,       // Poisson(1)
    UNIFORM        // Uniform[0,1]
}
