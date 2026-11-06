package dtw.kpz;

import dtw.kpz.DirectedPassage.Cost;
import dtw.kpz.DirectedPassage.Model;
import dtw.kpz.DirectedPassage.Outcome;
import dtw.kpz.Distribution;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.CRC32;


/**
 * Teste de equivalência da flutuação do DTW clássico com a percolação de última
 * passagem e teste de pertencimento à classe KPZ (expoente 1/3, geodésico 2/3,
 * forma Tracy–Widom GUE).
 *
 * A não-rejeição de um teste de igualdade não é equivalência: a decisão
 * EQUIVALENTE exige que o limite superior de confiança de W1 fique abaixo da
 * margem ε calibrada contra a Gaussiana e contra TW1.
 */
public final class KpzEquivalenceStudy {
    private static final double LPP_SPEED = 4.0;
    private static final double LPP_SCALE = Math.pow(2.0, 4.0 / 3.0);

    public static void main(String[] args) throws Exception {
        if (has(args, "--help")) {
            System.out.println(usage());
            return;
        }
        Config cfg = Config.parse(args);
        if (cfg.checkOnly) {
            int code = runCheck(cfg);
            System.exit(code);
        }
        if (cfg.refreshConcluded) {
            refreshConcluded();
            return;
        }
        runStudy(cfg);
    }

    private static int runCheck(Config cfg) {
        System.out.println("== verificacao da tabela Tracy-Widom e dos nucleos ==");
        double inv = EmpiricalStats.normalQuantile(0.975);
        System.out.printf(Locale.US, "invNorm(0.975)=%.5f (esperado 1.95996)%n", inv);
        TracyWidom tw = TracyWidom.build();
        System.out.println(tw.diagnostic());
        System.out.printf(Locale.US,
                "TW2 media=%.6f (ref %.6f) var=%.6f (ref %.6f) assim=%.4f (ref %.4f)%n",
                tw.mean2(), TracyWidom.TW2_MEAN_REF, tw.variance2(), TracyWidom.TW2_VAR_REF,
                tw.skewness2(), TracyWidom.TW2_SKEW_REF);
        System.out.printf(Locale.US,
                "TW1 media=%.6f (ref %.6f) var=%.6f (ref %.6f) erro assintota esquerda=%.3e confiavel=%s%n",
                tw.mean1(), TracyWidom.TW1_MEAN_REF, tw.variance1(), TracyWidom.TW1_VAR_REF,
                tw.leftRelError(), tw.reliable());
        int n = 48;
        int reps = 80;
        double[] lpp = new double[reps];
        long t0 = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            lpp[r] = DirectedPassage.simulate(Model.LPP, n, cfg.seed, r, true, Cost.ABS,
                    Distribution.EXPONENTIAL, Distribution.NORMAL).value;
        }
        long t1 = System.nanoTime();
        double cells = (double) n * n * reps;
        double sec = (t1 - t0) / 1e9;
        System.out.printf(Locale.US, "LPP n=%d R=%d  %.3f s  %.1f Mcelulas/s (1 fio)%n",
                n, reps, sec, cells / sec / 1e6);
        double scale = LPP_SCALE * Math.pow(n, 1.0 / 3.0);
        double zMean = 0.0;
        for (int r = 0; r < reps; r++) {
            zMean += (lpp[r] - LPP_SPEED * n) / scale;
        }
        zMean /= reps;
        System.out.printf(Locale.US, "media de Z teorico LPP (n=%d, R=%d)=%.3f (limite %.3f; vies de n pequeno e esperado)%n",
                n, reps, zMean, TracyWidom.TW2_MEAN_REF);
        Outcome dtw = DirectedPassage.simulate(Model.DTW, 32, cfg.seed, 0, true, Cost.ABS,
                cfg.weightDist, cfg.seqDist);
        Outcome fpp = DirectedPassage.simulate(Model.FPP3, 32, cfg.seed, 0, true, Cost.ABS,
                cfg.weightDist, cfg.seqDist);
        Outcome lpp3 = DirectedPassage.simulate(Model.LPP3, 32, cfg.seed, 0, true, Cost.ABS,
                cfg.weightDist, cfg.seqDist);
        System.out.printf(Locale.US, "fumaca DTW=%.6f mid=%d  FPP3=%.6f mid=%d  LPP3=%.6f mid=%d%n",
                dtw.value, dtw.midColumn, fpp.value, fpp.midColumn, lpp3.value, lpp3.midColumn);
        boolean ok = tw.reliable()
                && Math.abs(inv - 1.95996398454) < 2e-4
                && Double.isFinite(dtw.value)
                && dtw.midColumn >= 0 && dtw.midColumn < 32
                && Double.isFinite(fpp.value)
                && fpp.midColumn >= 0 && fpp.midColumn < 32
                && Double.isFinite(lpp3.value)
                && lpp3.midColumn >= 0 && lpp3.midColumn < 32;
        System.out.println(ok ? "CHECK PASSA" : "CHECK FALHA");
        return ok ? 0 : 1;
    }

    private static void runStudy(Config cfg) throws Exception {
        TracyWidom tw = TracyWidom.build();
        if (!tw.reliable()) {
            System.err.println("Tabela Tracy-Widom nao confiavel: " + tw.diagnostic());
            System.err.println("Rode com --check. O estudo foi interrompido para nao produzir p-valores falsos.");
            System.exit(2);
        }
        double dGauss = EmpiricalStats.wassersteinStudentized(
                tw::quantile2, tw.mean2(), tw.sd2(),
                EmpiricalStats::normalQuantile, 0.0, 1.0, 16001);
        double dGoe = EmpiricalStats.wassersteinStudentized(
                tw::quantile2, tw.mean2(), tw.sd2(),
                tw::quantile1, tw.mean1(), tw.sd1(), 16001);
        double epsGoe = 0.5 * dGoe;
        double epsGauss = 0.5 * dGauss;
        double epsilon = Math.min(epsGoe, epsGauss);

        int threads = cfg.threads > 0 ? cfg.threads : Runtime.getRuntime().availableProcessors();
        ExecutorService pool = Executors.newFixedThreadPool(threads, r -> {
            Thread t = new Thread(r, "kpz");
            t.setDaemon(true);
            return t;
        });
        try {
            Model[] models = Model.values();
            Bundle[][] data = new Bundle[models.length][cfg.ns.length];
            long wall0 = System.nanoTime();
            for (Model model : models) {
                for (int i = 0; i < cfg.ns.length; i++) {
                    int n = cfg.ns[i];
                    System.err.printf(Locale.US, "simulando %s n=%d pilot=%d eval=%d%n",
                            model, n, cfg.pilot, cfg.replicates);
                    data[model.id][i] = simulateBundle(model, n, cfg, pool);
                }
            }
            double wall = (System.nanoTime() - wall0) / 1e9;

            Report rep = new Report();
            rep.meta(cfg, tw, threads, wall, dGauss, dGoe, epsilon);

            Xoroshiro128 boot = new Xoroshiro128(cfg.seed ^ 0xA5A5A5A5A5A5A5A5L);
            for (Model model : models) {
                analyseModel(model, data[model.id], cfg, tw, boot, rep);
            }
            equivalence(Model.DTW, Model.LPP, data, cfg, epsilon, boot, rep);
            equivalence(Model.DTW, Model.FPP3, data, cfg, epsilon, boot, rep);
            equivalence(Model.FPP3, Model.LPP, data, cfg, epsilon, boot, rep);

            String text = rep.text.toString();
            System.out.print(text);
            Path out = Path.of(cfg.outCsv);
            if (out.getParent() != null) {
                Files.createDirectories(out.getParent());
            }
            Files.writeString(out, rep.csv.toString(), StandardCharsets.UTF_8);
            Path txt = Path.of(cfg.outCsv.replaceAll("\\.csv$", "") + ".txt");
            Files.writeString(txt, text, StandardCharsets.UTF_8);
            maybeWriteConcluded(cfg, rep.csv.toString());
            System.err.println("CSV: " + out.toAbsolutePath());
            System.err.println("TXT: " + txt.toAbsolutePath());
        } finally {
            pool.shutdown();
        }
    }

    private static void maybeWriteConcluded(Config cfg, String currentCsv) throws IOException {
        if (cfg.seqDist != Distribution.UNIFORM || !cfg.outCsv.endsWith("kpz-concluido-unif.csv")) return;
        Path exponential = Path.of("results/kpz-concluido-exp.csv");
        if (!Files.exists(exponential)) return;
        writeConcluded(Files.readString(exponential, StandardCharsets.UTF_8), currentCsv);
    }

    private static void refreshConcluded() throws IOException {
        Path exponential = Path.of("results/kpz-concluido-exp.csv");
        Path uniform = Path.of("results/kpz-concluido-unif.csv");
        if (!Files.exists(exponential) || !Files.exists(uniform)) {
            throw new IOException("faltam os CSVs de produção exponential/uniforme");
        }
        writeConcluded(Files.readString(exponential, StandardCharsets.UTF_8),
                Files.readString(uniform, StandardCharsets.UTF_8));
    }

    private static void writeConcluded(String expCsv, String uniformCsv) throws IOException {
        String combined = expCsv + uniformCsv.substring(uniformCsv.indexOf('\n') + 1);
        Files.writeString(Path.of("results/kpz-concluido.csv"), combined, StandardCharsets.UTF_8);
        Files.writeString(Path.of("results/kpz-concluido.txt"), concludedText(expCsv, uniformCsv),
                StandardCharsets.UTF_8);
    }

    private static String concludedText(String expCsv, String uniformCsv) {
        Map<String, String> exponential = parseMetrics(expCsv);
        Map<String, String> uniform = parseMetrics(uniformCsv);
        SequenceSummary e = summarizeSequence(exponential);
        SequenceSummary u = summarizeSequence(uniform);
        StringBuilder text = new StringBuilder();
        text.append("LPP exponencial é o único controle Johansson válido. Uniforme em FALHA_ESCALA não conta.\n\n");
        appendSequenceParagraph(text, "Exponencial", e);
        text.append('\n');
        appendSequenceParagraph(text, "Uniforme", u);
        text.append('\n');
        text.append(String.format(Locale.US,
                "W1 contra a lei de sinal certo, descontado o piso, está %s na sequência exponencial e %s na uniforme (plano não é equivalência).%n%n",
                e.w1Trend, u.w1Trend));
        text.append(String.format(Locale.US,
                "Resíduo marginal: sequência exponencial R² piloto final=%.4f, chi do sd residual no último par=%.4f [IC %.4f, %.4f]; uniforme R² piloto final=%.4f, chi do sd residual no último par=%.4f [IC %.4f, %.4f].%s%s%n%n",
                e.r2Last, e.residualChi, e.residualLo, e.residualHi,
                u.r2Last, u.residualChi, u.residualLo, u.residualHi,
                e.r2NotUp && e.residualChi >= e.chiC ? " Na exponencial, R² não sobe com n e chi residual não é menor que chi de C." : "",
                u.r2NotUp && u.residualChi >= u.chiC ? " Na uniforme, R² não sobe com n e chi residual não é menor que chi de C." : ""));
        String conclusion;
        if (e.hypothesis != 0 && u.hypothesis != 0 && e.hypothesis != u.hypothesis) {
            conclusion = "LEIS_DISCORDEM: as duas sequências não apontam para a mesma hipótese.";
        } else if (e.hypothesis == 1 && u.hypothesis == 1
                && e.skewCoversMinus && u.skewCoversMinus) {
            conclusion = "COMPATIVEL_COM_REFLEXAO_TW_NA_JANELA: nas duas leis, o último IC de chi cobre 1/3 e não cobre 1/2, e o IC da assimetria cobre -0.224.";
        } else if (e.hypothesis == 2 && u.hypothesis == 2
                && e.skewCoversZero && u.skewCoversZero) {
            conclusion = "COMPATIVEL_COM_CLT_NA_JANELA: nas duas leis, o último IC de chi cobre 1/2 e não cobre 1/3, e o IC da assimetria cobre 0.";
        } else {
            conclusion = "INCONCLUSIVO_ATE_1024: qualquer outro caso.";
        }
        text.append(conclusion).append('\n');
        return text.toString();
    }

    private static void appendSequenceParagraph(StringBuilder text, String name, SequenceSummary s) {
        String chiCover = s.coversThird && s.coversHalf ? "ambos"
                : s.coversThird ? "1/3" : s.coversHalf ? "1/2" : "nenhum";
        text.append(String.format(Locale.US,
                "Na sequência %s, o chi local do DTW no último par %d→%d é %.4f, IC[%.4f, %.4f], cobrindo %s (alvos 1/3 e 1/2); a assimetria é %.4f, IC[%.4f, %.4f], mais perto de %s.%n",
                name, s.pairN, 2 * s.pairN, s.chi, s.chiLo, s.chiHi, chiCover,
                s.skew, s.skewLo, s.skewHi, s.nearestSkew));
    }

    private static Map<String, String> parseMetrics(String csv) {
        Map<String, String> values = new HashMap<>();
        String[] lines = csv.split("\\R");
        for (int i = 1; i < lines.length; i++) {
            String[] fields = lines[i].split(",", -1);
            if (fields.length == 5) values.put(fields[0] + "|" + fields[2] + "|" + fields[3], fields[4]);
        }
        return values;
    }

    private static SequenceSummary summarizeSequence(Map<String, String> values) {
        SequenceSummary s = new SequenceSummary();
        int maxN = -1;
        for (String key : values.keySet()) {
            String[] parts = key.split("\\|");
            if (parts.length == 3 && parts[0].equals("DTW") && parts[2].equals("chi_local")) {
                int n = Integer.parseInt(parts[1]);
                if (n > maxN) maxN = n;
            }
        }
        s.pairN = maxN;
        s.chi = metric(values, "DTW", maxN, "chi_local");
        s.chiLo = metric(values, "DTW", maxN, "chi_local_lo");
        s.chiHi = metric(values, "DTW", maxN, "chi_local_hi");
        s.coversThird = s.chiLo <= 1.0 / 3.0 && 1.0 / 3.0 <= s.chiHi;
        s.coversHalf = s.chiLo <= 0.5 && 0.5 <= s.chiHi;
        int lastN = 0;
        for (String key : values.keySet()) {
            String[] parts = key.split("\\|");
            if (parts.length == 3 && parts[0].equals("DTW") && parts[2].equals("assimetria")) {
                lastN = Math.max(lastN, Integer.parseInt(parts[1]));
            }
        }
        s.skew = metric(values, "DTW", lastN, "assimetria");
        s.skewLo = metric(values, "DTW", lastN, "assimetria_lo");
        s.skewHi = metric(values, "DTW", lastN, "assimetria_hi");
        double d0 = Math.abs(s.skew);
        double dp = Math.abs(s.skew - 0.224);
        double dm = Math.abs(s.skew + 0.224);
        s.nearestSkew = d0 <= dp && d0 <= dm ? "0" : dp <= dm ? "+0.224" : "-0.224";
        s.skewCoversZero = s.skewLo <= 0.0 && 0.0 <= s.skewHi;
        s.skewCoversMinus = s.skewLo <= -0.224 && -0.224 <= s.skewHi;
        s.hypothesis = s.coversThird && !s.coversHalf ? 1
                : s.coversHalf && !s.coversThird ? 2 : 0;
        List<Integer> sizes = new ArrayList<>();
        for (String key : values.keySet()) {
            String[] parts = key.split("\\|");
            if (parts.length == 3 && parts[0].equals("DTW") && parts[2].equals("w1_alvo_menos_piso")) {
                sizes.add(Integer.parseInt(parts[1]));
            }
        }
        sizes.sort(Integer::compareTo);
        double firstW1 = sizes.isEmpty() ? Double.NaN
                : metric(values, "DTW", sizes.get(0), "w1_alvo_menos_piso");
        double lastW1 = sizes.isEmpty() ? Double.NaN
                : metric(values, "DTW", sizes.get(sizes.size() - 1), "w1_alvo_menos_piso");
        s.w1Trend = sizes.size() > 1 && lastW1 < firstW1 ? "descendo" : "plano";
        s.r2Last = metric(values, "DTW", lastN, "r2_piloto");
        int firstN = Integer.MAX_VALUE;
        for (String key : values.keySet()) {
            String[] parts = key.split("\\|");
            if (parts.length == 3 && parts[0].equals("DTW") && parts[2].equals("r2_piloto")) {
                firstN = Math.min(firstN, Integer.parseInt(parts[1]));
            }
        }
        s.r2NotUp = firstN < Integer.MAX_VALUE
                && s.r2Last <= metric(values, "DTW", firstN, "r2_piloto");
        s.residualChi = metric(values, "DTW", maxN, "chi_sd_residuo");
        s.residualLo = metric(values, "DTW", maxN, "chi_sd_residuo_lo");
        s.residualHi = metric(values, "DTW", maxN, "chi_sd_residuo_hi");
        s.chiC = metric(values, "DTW", maxN, "chi_sd_C");
        return s;
    }

    private static double metric(Map<String, String> values, String model, int n, String key) {
        return Double.parseDouble(values.get(model + "|" + n + "|" + key));
    }

    private static final class SequenceSummary {
        int pairN;
        double chi, chiLo, chiHi, skew, skewLo, skewHi, r2Last, residualChi, residualLo, residualHi;
        boolean coversThird, coversHalf, skewCoversZero, skewCoversMinus, r2NotUp;
        double chiC;
        int hypothesis;
        String nearestSkew, w1Trend;
    }

    private static Bundle simulateBundle(Model model, int n, Config cfg, ExecutorService pool)
            throws ExecutionException, InterruptedException, IOException {
        Bundle b = new Bundle();
        b.n = n;
        b.model = model;
        b.pilot = new double[cfg.pilot];
        b.eval = new double[cfg.replicates];
        b.pilotFeatures = model == Model.DTW ? new double[4][cfg.pilot] : null;
        b.features = model == Model.DTW ? new double[4][cfg.replicates] : null;
        b.mid = cfg.geodesic ? new double[cfg.replicates] : null;
        long cells0 = System.nanoTime();
        runRange(model, n, "pilot", cfg.seed, 0, b.pilot, null, b.pilotFeatures, false,
                cfg, pool);
        runRange(model, n, "eval", cfg.seed, 1 << 30, b.eval, b.mid, b.features, cfg.geodesic,
                cfg, pool);
        b.seconds = (System.nanoTime() - cells0) / 1e9;
        b.pilotMean = EmpiricalStats.mean(b.pilot);
        b.pilotSd = EmpiricalStats.sd(b.pilot);
        b.z = new double[b.eval.length];
        EmpiricalStats.studentize(b.eval, b.z, b.pilotMean, b.pilotSd);
        b.zSorted = EmpiricalStats.sortedCopy(b.z);
        return b;
    }

    private static void runRange(Model model, int n, String phase, long seed, int offset,
                                 double[] values, double[] mids, double[][] features,
                                 boolean geodesic, Config cfg, ExecutorService pool)
            throws ExecutionException, InterruptedException, IOException {
        int reps = values.length;
        String identity = String.join("|", model.name(), Integer.toString(n), phase,
                Integer.toString(offset), Integer.toString(reps), Long.toString(seed),
                cfg.cost.name(), cfg.weightDist.name(), cfg.seqDist.name(),
                Boolean.toString(geodesic), Boolean.toString(mids != null),
                features == null ? "0" : Integer.toString(features.length));
        Path directory = Path.of(cfg.checkpointDir);
        Files.createDirectories(directory);
        Path path = directory.resolve(model.name().toLowerCase(Locale.ROOT) + "-n" + n
                + "-" + phase + ".bin");
        try (CheckpointStore checkpoint = new CheckpointStore(path, identity, values, mids, features)) {
            int resumed = checkpoint.completedCount();
            int pending = reps - resumed;
            System.err.printf(Locale.US,
                    "checkpoint %s n=%d fase=%s recuperadas=%d/%d arquivo=%s%n",
                    model, n, phase, resumed, reps, path.toAbsolutePath());
            if (pending == 0) return;

            int batches = Math.max(1, Runtime.getRuntime().availableProcessors() * 2);
            List<Future<?>> futures = new ArrayList<>(batches);
            AtomicInteger completed = new AtomicInteger();
            for (int b = 0; b < batches; b++) {
                int start = (int) ((long) b * reps / batches);
                int end = (int) ((long) (b + 1) * reps / batches);
                futures.add(pool.submit(() -> {
                    try {
                        for (int r = start; r < end; r++) {
                            if (checkpoint.isDone(r)) continue;
                            Outcome o = DirectedPassage.simulate(model, n, seed, offset + r,
                                    geodesic, cfg.cost, cfg.weightDist, cfg.seqDist);
                            double mid = mids == null ? 0.0 : o.midColumn - 0.5 * (n - 1);
                            double[] featureRow = features == null ? null :
                                    new double[] {o.mX, o.mY, o.w1xy, o.emparelhado};
                            checkpoint.write(r, o.value, mid, featureRow);
                            int count = completed.incrementAndGet();
                            if (count % cfg.checkpointEvery == 0 || count == pending) {
                                checkpoint.force();
                                System.err.printf(Locale.US,
                                        "checkpoint %s n=%d fase=%s novas=%d/%d total=%d/%d%n",
                                        model, n, phase, count, pending, resumed + count, reps);
                            }
                        }
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
            }));
            }
            for (Future<?> f : futures) f.get();
            checkpoint.force();
            if (completed.get() % cfg.checkpointEvery != 0 && completed.get() != pending) {
                System.err.printf(Locale.US,
                        "checkpoint %s n=%d fase=%s novas=%d/%d total=%d/%d persistido%n",
                        model, n, phase, completed.get(), pending, resumed + completed.get(), reps);
            }
        }
    }

    private static void analyseModel(Model model, Bundle[] bundles, Config cfg, TracyWidom tw,
                                    Xoroshiro128 boot, Report rep) {
        int k = bundles.length;
        double[] logN = new double[k];
        double[] logS = new double[k];
        double[] w = new double[k];
        double[] meanY = new double[k];
        double[] nLin = new double[k];
        double[] nThird = new double[k];
        double[] varY = new double[k];
        double[] nTwoThird = new double[k];
        for (int i = 0; i < k; i++) {
            logN[i] = Math.log(bundles[i].n);
            logS[i] = Math.log(bundles[i].pilotSd);
            w[i] = cfg.pilot;
            meanY[i] = bundles[i].pilotMean;
            nLin[i] = bundles[i].n;
            nThird[i] = Math.pow(bundles[i].n, 1.0 / 3.0);
            varY[i] = bundles[i].pilotSd * bundles[i].pilotSd;
            nTwoThird[i] = Math.pow(bundles[i].n, 2.0 / 3.0);
        }
        double[] slope = EmpiricalStats.wlsSlope(logN, logS, w);
        double theta = slope[1];
        double[] thetaBoot = new double[cfg.bootstrap];
        double[][] scratch = new double[k][];
        for (int i = 0; i < k; i++) {
            scratch[i] = new double[bundles[i].pilot.length];
        }
        for (int b = 0; b < cfg.bootstrap; b++) {
            for (int i = 0; i < k; i++) {
                resampleInto(bundles[i].pilot, scratch[i], boot);
                logS[i] = Math.log(EmpiricalStats.sd(scratch[i]));
            }
            thetaBoot[b] = EmpiricalStats.wlsSlope(logN, logS, w)[1];
        }
        Arrays.sort(thetaBoot);
        double thetaLo = EmpiricalStats.percentile(thetaBoot, cfg.alpha / 2.0);
        double thetaHi = EmpiricalStats.percentile(thetaBoot, 1.0 - cfg.alpha / 2.0);
        String thetaDec = exponentDecision(thetaLo, thetaHi, 1.0 / 3.0, 0.5);

        rep.section(model + " expoente de flutuacao");
        rep.line(model, 0, "theta", theta);
        rep.line(model, 0, "theta_lo", thetaLo);
        rep.line(model, 0, "theta_hi", thetaHi);
        rep.line(model, 0, "theta_decisao", thetaDec);
        rep.text.append(String.format(Locale.US,
                "%s  theta=%.4f  IC[%.4f, %.4f]  decisao=%s  (KPZ 1/3, CLT 1/2)%n",
                model, theta, thetaLo, thetaHi, thetaDec));

        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                if (bundles[j].n == 2 * bundles[i].n) {
                    double ratio = bundles[j].pilotSd / bundles[i].pilotSd;
                    double[] ratios = new double[cfg.bootstrap];
                    double[] a = new double[bundles[i].pilot.length];
                    double[] c = new double[bundles[j].pilot.length];
                    for (int b = 0; b < cfg.bootstrap; b++) {
                        resampleInto(bundles[i].pilot, a, boot);
                        resampleInto(bundles[j].pilot, c, boot);
                        ratios[b] = EmpiricalStats.sd(c) / EmpiricalStats.sd(a);
                    }
                    Arrays.sort(ratios);
                    double lo = EmpiricalStats.percentile(ratios, cfg.alpha / 2.0);
                    double hi = EmpiricalStats.percentile(ratios, 1.0 - cfg.alpha / 2.0);
                    rep.line(model, bundles[i].n, "razao_sd_2n", ratio);
                    rep.line(model, bundles[i].n, "razao_sd_2n_lo", lo);
                    rep.line(model, bundles[i].n, "razao_sd_2n_hi", hi);
                    double chi = Math.log(ratio) / Math.log(2.0);
                    double chiLo = Math.log(lo) / Math.log(2.0);
                    double chiHi = Math.log(hi) / Math.log(2.0);
                    boolean coversThird = lo <= Math.pow(2.0, 1.0 / 3.0) && Math.pow(2.0, 1.0 / 3.0) <= hi;
                    boolean coversHalf = lo <= Math.sqrt(2.0) && Math.sqrt(2.0) <= hi;
                    String localDecision = coversThird && coversHalf ? "COBRE_AMBOS"
                            : coversThird ? "COBRE_1_3" : coversHalf ? "COBRE_1_2" : "NENHUM";
                    rep.line(model, bundles[i].n, "chi_local", chi);
                    rep.line(model, bundles[i].n, "chi_local_lo", chiLo);
                    rep.line(model, bundles[i].n, "chi_local_hi", chiHi);
                    rep.line(model, bundles[i].n, "chi_local_decisao", localDecision);
                    rep.text.append(String.format(Locale.US,
                            "  par %d,%d: chi=%.4f IC[%.4f, %.4f]; alvos 1/3=%.4f e 1/2=%.4f; %s%n",
                            bundles[i].n, bundles[j].n, chi, chiLo, chiHi,
                            1.0 / 3.0, 0.5, localDecision));
                }
            }
        }

        double[] hydro = EmpiricalStats.wls3(nLin, nThird, meanY, w);
        double[] varFit = EmpiricalStats.wls3(nTwoThird, nThird, varY, w);
        rep.line(model, 0, "mu_hidrodinamico", hydro[1]);
        rep.line(model, 0, "correcao_n13", hydro[2]);
        rep.line(model, 0, "intercepto_media", hydro[0]);
        rep.line(model, 0, "sigma2", varFit[1]);
        rep.text.append(String.format(Locale.US,
                "  media ≈ %.6f * n + %.6f * n^(1/3) + %.6f%n  variancia ≈ %.6f * n^(2/3) + %.6f * n^(1/3) + %.6f%n",
                hydro[1], hydro[2], hydro[0], varFit[1], varFit[2], varFit[0]));

        String shapeAll = "PASSA";
        for (Bundle bundle : bundles) {
            double skew = EmpiricalStats.skewness(bundle.z);
            double kurt = EmpiricalStats.excessKurtosis(bundle.z);
            double[] zCopy = EmpiricalStats.sortedCopy(bundle.z);
            boolean reflected = model == Model.DTW || model == Model.FPP3;
            double ad = EmpiricalStats.andersonDarling(zCopy, tw::cdf2Studentized, tw::survival2Studentized);
            double p = adPValue(ad, bundle.z.length, cfg.pilot, cfg.bootstrap, tw, boot);
            double adSign = reflected
                    ? EmpiricalStats.andersonDarling(zCopy, tw::cdf2ReflectedStudentized, tw::survival2ReflectedStudentized)
                    : ad;
            double pSign = reflected
                    ? adPValueReflected(adSign, bundle.z.length, cfg.pilot, cfg.bootstrap, tw, boot)
                    : p;
            double[] skewBoot = new double[Math.min(cfg.bootstrap, 200)];
            double[] tmp = new double[bundle.z.length];
            for (int b = 0; b < skewBoot.length; b++) {
                resampleInto(bundle.z, tmp, boot);
                skewBoot[b] = EmpiricalStats.skewness(tmp);
            }
            Arrays.sort(skewBoot);
            double skewLo = EmpiricalStats.percentile(skewBoot, cfg.alpha / 2.0);
            double skewHi = EmpiricalStats.percentile(skewBoot, 1.0 - cfg.alpha / 2.0);
            String shape = p < cfg.alpha ? "FALHA" : "PASSA";
            if ("FALHA".equals(shape)) {
                shapeAll = "FALHA";
            }
            rep.line(model, bundle.n, "assimetria", skew);
            rep.line(model, bundle.n, "assimetria_lo", skewLo);
            rep.line(model, bundle.n, "assimetria_hi", skewHi);
            rep.line(model, bundle.n, "curtose_excedente", kurt);
            rep.line(model, bundle.n, "anderson_darling", ad);
            rep.line(model, bundle.n, "ad_p", p);
            rep.line(model, bundle.n, "forma_tw2", shape);
            double skewTarget = reflected ? -TracyWidom.TW2_SKEW_REF : TracyWidom.TW2_SKEW_REF;
            rep.line(model, bundle.n, "assimetria_alvo", skewTarget);
            rep.line(model, bundle.n, "delta_assimetria", skew - skewTarget);
            rep.line(model, bundle.n, "ad_sinal", adSign);
            rep.line(model, bundle.n, "ad_p_sinal", pSign);
            String signedShape = pSign < cfg.alpha ? "FALHA" : "PASSA";
            rep.line(model, bundle.n, "forma_tw_sinal", signedShape);
            double[] lawZ = sampleStudentizedTw(tw, bundle.z.length, cfg.pilot, reflected, boot);
            double[] lawZ2 = sampleStudentizedTw(tw, bundle.z.length, cfg.pilot, reflected, boot);
            Arrays.sort(lawZ);
            Arrays.sort(lawZ2);
            double w1Target = EmpiricalStats.wasserstein1Sorted(bundle.zSorted, lawZ);
            double w1Floor = EmpiricalStats.wasserstein1Sorted(lawZ, lawZ2);
            double[] normalSorted = new double[bundle.z.length];
            for (int q = 0; q < normalSorted.length; q++) {
                normalSorted[q] = EmpiricalStats.normalQuantile((q + 0.5) / normalSorted.length);
            }
            rep.line(model, bundle.n, "w1_alvo", w1Target);
            rep.line(model, bundle.n, "w1_piso", w1Floor);
            rep.line(model, bundle.n, "w1_alvo_menos_piso", w1Target - w1Floor);
            rep.line(model, bundle.n, "w1_normal", EmpiricalStats.wasserstein1Sorted(bundle.zSorted, normalSorted));
            rep.text.append(String.format(Locale.US,
                    "  n=%d  assim=%.3f IC[%.3f, %.3f] alvo=%.3f delta=%.3f AD_sinal=%.3f p_sinal=%.4f %s%n",
                    bundle.n, skew, skewLo, skewHi, skewTarget, skew - skewTarget, ad, p, signedShape));

            if (model == Model.LPP) {
                double[] zTh = new double[bundle.eval.length];
                double scale = LPP_SCALE * Math.pow(bundle.n, 1.0 / 3.0);
                for (int i = 0; i < zTh.length; i++) {
                    zTh[i] = (bundle.eval[i] - LPP_SPEED * bundle.n) / scale;
                }
                Arrays.sort(zTh);
                double adTh = EmpiricalStats.andersonDarling(zTh, tw::cdf2, tw::survival2);
                double pTh = adPValueRaw(adTh, zTh.length, cfg.bootstrap, tw, boot);
                double m = EmpiricalStats.mean(zTh);
                double s = EmpiricalStats.sd(zTh);
                String ctrl;
                if (s < 0.45 || s > 2.2) {
                    ctrl = "FALHA_ESCALA";
                } else if (pTh < cfg.alpha) {
                    ctrl = "AFASTA_TW_FINITO";
                } else {
                    ctrl = "COMPATIVEL";
                }
                rep.line(model, bundle.n, "z_teorico_media", m);
                rep.line(model, bundle.n, "z_teorico_sd", s);
                rep.line(model, bundle.n, "z_teorico_ad", adTh);
                rep.line(model, bundle.n, "z_teorico_p", pTh);
                rep.line(model, bundle.n, "controle_johansson", ctrl);
                rep.text.append(String.format(Locale.US,
                        "  controle Johansson n=%d  mediaZ=%.3f sdZ=%.3f AD=%.3f p=%.4f %s%n",
                        bundle.n, m, s, adTh, pTh, ctrl));
            }
        }

        String geoDec = "NAO_PEDIDO";
        if (cfg.geodesic) {
            double[] gLogN = new double[k];
            double[] gLogS = new double[k];
            boolean ok = true;
            for (int i = 0; i < k; i++) {
                double s = EmpiricalStats.sd(bundles[i].mid);
                if (!(s > 0.0) || !Double.isFinite(s)) {
                    ok = false;
                    break;
                }
                gLogN[i] = Math.log(bundles[i].n);
                gLogS[i] = Math.log(s);
            }
            if (ok) {
                double gTheta = EmpiricalStats.wlsSlope(gLogN, gLogS, w)[1];
                double[] gBoot = new double[cfg.bootstrap];
                double[] gTmp = new double[bundles[0].mid.length];
                double[] gS = new double[k];
                for (int b = 0; b < cfg.bootstrap; b++) {
                    for (int i = 0; i < k; i++) {
                        if (gTmp.length != bundles[i].mid.length) {
                            gTmp = new double[bundles[i].mid.length];
                        }
                        resampleInto(bundles[i].mid, gTmp, boot);
                        gS[i] = Math.log(Math.max(EmpiricalStats.sd(gTmp), 1e-12));
                    }
                    gBoot[b] = EmpiricalStats.wlsSlope(gLogN, gS, w)[1];
                }
                Arrays.sort(gBoot);
                double lo = EmpiricalStats.percentile(gBoot, cfg.alpha / 2.0);
                double hi = EmpiricalStats.percentile(gBoot, 1.0 - cfg.alpha / 2.0);
                geoDec = exponentDecision(lo, hi, 2.0 / 3.0, 0.5);
                rep.line(model, 0, "theta_transversal", gTheta);
                rep.line(model, 0, "theta_transversal_lo", lo);
                rep.line(model, 0, "theta_transversal_hi", hi);
                rep.line(model, 0, "theta_transversal_decisao", geoDec);
                rep.text.append(String.format(Locale.US,
                        "  geodesico theta=%.4f IC[%.4f, %.4f] decisao=%s (KPZ 2/3)%n",
                        gTheta, lo, hi, geoDec));
            } else {
                geoDec = "FALHA";
                rep.line(model, 0, "theta_transversal_decisao", geoDec);
            }
        }

        if (model == Model.DTW) {
            analyseMarginalResiduals(bundles, cfg, boot, rep);
        }

        String membership;
        if ("FALHA".equals(thetaDec) || "FALHA".equals(shapeAll) || "FALHA".equals(geoDec)) {
            membership = "FORA";
        } else if ("INCONCLUSIVO".equals(thetaDec) || "INCONCLUSIVO".equals(geoDec)) {
            membership = "INCONCLUSIVO";
        } else if ("PASSA".equals(thetaDec) && "PASSA".equals(shapeAll)
                && (!cfg.geodesic || "PASSA".equals(geoDec))) {
            membership = "COMPATIVEL_MARGEM_PONTO_A_PONTO";
        } else {
            membership = "INCONCLUSIVO";
        }
        rep.line(model, 0, "pertencimento_kpz", membership);
        rep.text.append("  pertencimento KPZ (margem + expoentes): ").append(membership).append('\n');
    }

    private static void analyseMarginalResiduals(Bundle[] bundles, Config cfg, Xoroshiro128 boot, Report rep) {
        for (Bundle b : bundles) {
            double[][] x = b.pilotFeatures;
            double[][] e = b.features;
            double[] coefficients = olsMarginal(x, b.pilot);
            double[] fittedPilot = predictMarginal(x, coefficients);
            double meanC = EmpiricalStats.mean(b.pilot);
            double sst = 0.0;
            double sse = 0.0;
            for (int i = 0; i < b.pilot.length; i++) {
                double d = b.pilot[i] - meanC;
                sst += d * d;
                double r = b.pilot[i] - fittedPilot[i];
                sse += r * r;
            }
            b.r2Pilot = sst > 0.0 ? 1.0 - sse / sst : 0.0;
            b.residual = new double[b.eval.length];
            b.cEval = new double[b.eval.length];
            for (int i = 0; i < b.eval.length; i++) {
                b.cEval[i] = predictMarginalAt(e, coefficients, i);
                b.residual[i] = b.eval[i] - b.cEval[i];
            }
            b.residualSd = EmpiricalStats.sd(b.residual);
            b.cSd = EmpiricalStats.sd(b.cEval);
            rep.line(Model.DTW, b.n, "sd_residuo", b.residualSd);
            rep.line(Model.DTW, b.n, "sd_C", b.cSd);
            rep.line(Model.DTW, b.n, "r2_piloto", b.r2Pilot);
        }
        for (int i = 0; i < bundles.length; i++) {
            for (int j = i + 1; j < bundles.length; j++) {
                Bundle a = bundles[i];
                Bundle c = bundles[j];
                if (c.n != 2 * a.n) continue;
                double chiR = Math.log(c.residualSd / a.residualSd) / Math.log(2.0);
                double chiC = Math.log(c.cSd / a.cSd) / Math.log(2.0);
                double[] boots = new double[cfg.bootstrap];
                double[] cBoots = new double[cfg.bootstrap];
                double[] ra = new double[a.residual.length];
                double[] rc = new double[c.residual.length];
                double[] ca = new double[a.cEval.length];
                double[] cc = new double[c.cEval.length];
                for (int t = 0; t < boots.length; t++) {
                    resampleInto(a.residual, ra, boot);
                    resampleInto(c.residual, rc, boot);
                    resampleInto(a.cEval, ca, boot);
                    resampleInto(c.cEval, cc, boot);
                    boots[t] = Math.log(EmpiricalStats.sd(rc) / EmpiricalStats.sd(ra)) / Math.log(2.0);
                    cBoots[t] = Math.log(EmpiricalStats.sd(cc) / EmpiricalStats.sd(ca)) / Math.log(2.0);
                }
                Arrays.sort(boots);
                Arrays.sort(cBoots);
                double lo = EmpiricalStats.percentile(boots, cfg.alpha / 2.0);
                double hi = EmpiricalStats.percentile(boots, 1.0 - cfg.alpha / 2.0);
                double cLo = EmpiricalStats.percentile(cBoots, cfg.alpha / 2.0);
                double cHi = EmpiricalStats.percentile(cBoots, 1.0 - cfg.alpha / 2.0);
                rep.line(Model.DTW, a.n, "chi_sd_residuo", chiR);
                rep.line(Model.DTW, a.n, "chi_sd_residuo_lo", lo);
                rep.line(Model.DTW, a.n, "chi_sd_residuo_hi", hi);
                rep.line(Model.DTW, a.n, "chi_sd_C", chiC);
                rep.line(Model.DTW, a.n, "chi_sd_C_lo", cLo);
                rep.line(Model.DTW, a.n, "chi_sd_C_hi", cHi);
                rep.line(Model.DTW, a.n, "chi_residuo_menor_chi_C", chiR < chiC ? "SIM" : "NAO");
                boolean r2NotUp = c.r2Pilot <= a.r2Pilot;
                if (r2NotUp && chiR >= chiC) {
                    rep.line(Model.DTW, a.n, "r2_nao_sobe_chi_residuo_nao_menor", "SIM");
                }
                rep.text.append(String.format(Locale.US,
                        "  residuo DTW %d->%d: R2 piloto %.4f -> %.4f; chi sd residual=%.4f IC[%.4f, %.4f], chi sd C=%.4f%n",
                        a.n, c.n, a.r2Pilot, c.r2Pilot, chiR, lo, hi, chiC));
                if (r2NotUp && chiR >= chiC) {
                    rep.text.append("  R2 nao sobe com n e chi do residuo nao e menor que chi de C.\n");
                }
            }
        }
    }

    private static double[] olsMarginal(double[][] features, double[] response) {
        double[][] a = new double[4][5];
        for (int i = 0; i < response.length; i++) {
            double[] x = {1.0, features[2][i], Math.abs(features[0][i] - features[1][i]), features[3][i]};
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) a[r][c] += x[r] * x[c];
                a[r][4] += x[r] * response[i];
            }
        }
        for (int col = 0; col < 4; col++) {
            int pivot = col;
            for (int r = col + 1; r < 4; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
            double[] tmp = a[col]; a[col] = a[pivot]; a[pivot] = tmp;
            double d = a[col][col];
            if (Math.abs(d) < 1e-18) return new double[4];
            for (int c = col; c < 5; c++) a[col][c] /= d;
            for (int r = 0; r < 4; r++) if (r != col) {
                double f = a[r][col];
                for (int c = col; c < 5; c++) a[r][c] -= f * a[col][c];
            }
        }
        return new double[] {a[0][4], a[1][4], a[2][4], a[3][4]};
    }

    private static double[] predictMarginal(double[][] features, double[] beta) {
        double[] fitted = new double[features[0].length];
        for (int i = 0; i < fitted.length; i++) fitted[i] = predictMarginalAt(features, beta, i);
        return fitted;
    }

    private static double predictMarginalAt(double[][] features, double[] beta, int i) {
        return beta[0] + beta[1] * features[2][i]
                + beta[2] * Math.abs(features[0][i] - features[1][i]) + beta[3] * features[3][i];
    }

    private static void equivalence(Model a, Model b, Bundle[][] data, Config cfg,
                                   double epsilon, Xoroshiro128 boot, Report rep) {
        int k = cfg.ns.length;
        double[] w1 = new double[k];
        double[] ucb = new double[k];
        double[] x = new double[k];
        double[] weights = new double[k];
        rep.section("equivalencia " + a + " vs " + b);
        for (int i = 0; i < k; i++) {
            Bundle ba = data[a.id][i];
            Bundle bb = data[b.id][i];
            w1[i] = EmpiricalStats.wasserstein1Sorted(ba.zSorted, bb.zSorted);
            if (a == Model.DTW && b == Model.LPP) {
                double[] negativeLpp = new double[bb.z.length];
                for (int q = 0; q < negativeLpp.length; q++) negativeLpp[q] = -bb.z[q];
                Arrays.sort(negativeLpp);
                rep.linePair(a, b, ba.n, "w1_dtw_menos_lpp", EmpiricalStats.wasserstein1Sorted(ba.zSorted, negativeLpp));
            }
            double[] boots = new double[cfg.bootstrap];
            double[] sa = new double[ba.z.length];
            double[] sb = new double[bb.z.length];
            for (int t = 0; t < cfg.bootstrap; t++) {
                resampleInto(ba.z, sa, boot);
                resampleInto(bb.z, sb, boot);
                Arrays.sort(sa);
                Arrays.sort(sb);
                boots[t] = EmpiricalStats.wasserstein1Sorted(sa, sb);
            }
            Arrays.sort(boots);
            ucb[i] = EmpiricalStats.percentile(boots, 1.0 - cfg.alpha);
            x[i] = Math.pow(ba.n, -1.0 / 3.0);
            weights[i] = cfg.replicates;
            String decN = ucb[i] < epsilon ? "EQUIVALENTE" : "NAO_EQUIVALENTE";
            rep.linePair(a, b, ba.n, "w1", w1[i]);
            rep.linePair(a, b, ba.n, "w1_ucb", ucb[i]);
            rep.linePair(a, b, ba.n, "decisao_n", decN);
            rep.text.append(String.format(Locale.US,
                    "  n=%d  W1=%.4f  UCB=%.4f  eps=%.4f  %s%n",
                    ba.n, w1[i], ucb[i], epsilon, decN));
        }
        double[] fit = EmpiricalStats.wlsSlope(x, w1, weights);
        double wInf = fit[0];
        double[] infBoot = new double[cfg.bootstrap];
        double[] wBoot = new double[k];
        double[] sa = new double[data[a.id][0].eval.length];
        double[] sb = new double[data[b.id][0].eval.length];
        for (int t = 0; t < cfg.bootstrap; t++) {
            for (int i = 0; i < k; i++) {
                if (sa.length != data[a.id][i].z.length) {
                    sa = new double[data[a.id][i].z.length];
                    sb = new double[data[b.id][i].z.length];
                }
                resampleInto(data[a.id][i].z, sa, boot);
                resampleInto(data[b.id][i].z, sb, boot);
                Arrays.sort(sa);
                Arrays.sort(sb);
                wBoot[i] = EmpiricalStats.wasserstein1Sorted(sa, sb);
            }
            infBoot[t] = EmpiricalStats.wlsSlope(x, wBoot, weights)[0];
        }
        Arrays.sort(infBoot);
        double ucbInf = EmpiricalStats.percentile(infBoot, 1.0 - cfg.alpha);
        String limitDec = ucbInf < epsilon ? "EQUIVALENTE" : "NAO_EQUIVALENTE";
        rep.linePair(a, b, 0, "w1_infinito", wInf);
        rep.linePair(a, b, 0, "w1_infinito_ucb", ucbInf);
        rep.linePair(a, b, 0, "epsilon", epsilon);
        rep.linePair(a, b, 0, "decisao_limite", limitDec);
        rep.text.append(String.format(Locale.US,
                "  extrapolacao W1(inf)=%.4f UCB=%.4f eps=%.4f  %s%n",
                wInf, ucbInf, epsilon, limitDec));
        rep.text.append("  (W1(inf) e o intercepto de W1(n)=W1(inf)+c n^(-1/3); UCB bootstrap percentil)\n");
    }

    private static double adPValue(double adObs, int evalN, int pilotN, int boot, TracyWidom tw, Xoroshiro128 rng) {
        int bUse = Math.max(49, Math.min(boot, 199));
        int ge = 0;
        double[] pilot = new double[pilotN];
        double[] eval = new double[evalN];
        for (int b = 0; b < bUse; b++) {
            for (int i = 0; i < pilotN; i++) {
                pilot[i] = tw.sample2(rng);
            }
            double m = EmpiricalStats.mean(pilot);
            double s = EmpiricalStats.sd(pilot);
            for (int i = 0; i < evalN; i++) {
                eval[i] = (tw.sample2(rng) - m) / s;
            }
            Arrays.sort(eval);
            double ad = EmpiricalStats.andersonDarling(eval, tw::cdf2Studentized, tw::survival2Studentized);
            if (ad >= adObs) {
                ge++;
            }
        }
        return (ge + 1.0) / (bUse + 1.0);
    }

    private static double adPValueReflected(double adObs, int evalN, int pilotN, int boot, TracyWidom tw, Xoroshiro128 rng) {
        int bUse = Math.max(49, Math.min(boot, 199));
        int ge = 0;
        double[] pilot = new double[pilotN];
        double[] eval = new double[evalN];
        for (int b = 0; b < bUse; b++) {
            for (int i = 0; i < pilotN; i++) pilot[i] = -tw.sample2(rng);
            double m = EmpiricalStats.mean(pilot);
            double s = EmpiricalStats.sd(pilot);
            for (int i = 0; i < evalN; i++) eval[i] = (-tw.sample2(rng) - m) / s;
            Arrays.sort(eval);
            double ad = EmpiricalStats.andersonDarling(eval, tw::cdf2ReflectedStudentized,
                    tw::survival2ReflectedStudentized);
            if (ad >= adObs) ge++;
        }
        return (ge + 1.0) / (bUse + 1.0);
    }

    private static double[] sampleStudentizedTw(TracyWidom tw, int n, int pilotN,
                                                 boolean reflected, Xoroshiro128 rng) {
        double[] pilot = new double[pilotN];
        double[] sample = new double[n];
        for (int i = 0; i < pilotN; i++) {
            double value = tw.sample2(rng);
            pilot[i] = reflected ? -value : value;
        }
        double mean = EmpiricalStats.mean(pilot);
        double sd = EmpiricalStats.sd(pilot);
        for (int i = 0; i < n; i++) {
            double value = tw.sample2(rng);
            sample[i] = ((reflected ? -value : value) - mean) / sd;
        }
        return sample;
    }

    private static double adPValueRaw(double adObs, int n, int boot, TracyWidom tw, Xoroshiro128 rng) {
        int bUse = Math.max(49, Math.min(boot, 199));
        int ge = 0;
        double[] eval = new double[n];
        for (int b = 0; b < bUse; b++) {
            for (int i = 0; i < n; i++) {
                eval[i] = tw.sample2(rng);
            }
            Arrays.sort(eval);
            double ad = EmpiricalStats.andersonDarling(eval, tw::cdf2, tw::survival2);
            if (ad >= adObs) {
                ge++;
            }
        }
        return (ge + 1.0) / (bUse + 1.0);
    }

    private static void resampleInto(double[] src, double[] dst, Xoroshiro128 rng) {
        int n = src.length;
        for (int i = 0; i < dst.length; i++) {
            dst[i] = src[rng.nextInt(n)];
        }
    }

    /** Reamostra a amostra bruta e studentiza com a própria média e desvio (forma). */
    private static void resampleStudentized(double[] raw, double[] dst, Xoroshiro128 rng) {
        int n = raw.length;
        for (int i = 0; i < n; i++) {
            dst[i] = raw[rng.nextInt(n)];
        }
        double m = EmpiricalStats.mean(dst);
        double s = EmpiricalStats.sd(dst);
        double inv = 1.0 / s;
        for (int i = 0; i < n; i++) {
            dst[i] = (dst[i] - m) * inv;
        }
    }

    private static String exponentDecision(double lo, double hi, double target, double alternative) {
        boolean hasTarget = lo <= target && target <= hi;
        boolean hasAlt = lo <= alternative && alternative <= hi;
        if (!hasTarget) {
            return "FALHA";
        }
        if (hasAlt) {
            return "INCONCLUSIVO";
        }
        return "PASSA";
    }

    private static boolean has(String[] args, String flag) {
        for (String a : args) {
            if (flag.equals(a)) {
                return true;
            }
        }
        return false;
    }

    private static String usage() {
        return String.join("\n",
                "Uso: java -cp build dtw.kpz.KpzEquivalenceStudy [opcoes]",
                "  --n 64,128,256          grades (default 64,128,256)",
                "  --replicates 800        amostra de avaliacao por n",
                "  --pilot 400             amostra independente para media/desvio",
                "  --bootstrap 199         replicas bootstrap dos IC/UCB",
                "  --alpha 0.01",
                "  --seed 20260325",
                "  --threads 0             0 = todos os processadores",
                "  --cost abs|square       custo local do DTW",
                "  --weight-dist EXPONENTIAL|NORMAL|POISSON|UNIFORM",
                "                          pesos de LPP/LPP3/FPP3 (default EXPONENTIAL)",
                "  --seq-dist EXPONENTIAL|NORMAL|POISSON|UNIFORM",
                "                          sequencias do DTW (default NORMAL)",
                "  --no-geodesic           nao estima o expoente transversal",
                "  --checkpoint-dir DIR    diretorio para checkpoints retomaveis",
                "  --checkpoint-every N    forca progresso a cada N realizacoes (default 100)",
                "  --out results/kpz-equivalence.csv",
                "  --check                 valida TW2/TW1 e os nucleos; sai em seguida",
                "",
                "Producao sugerida:",
                "  --n 64,128,256,512,1024 --replicates 5000 --pilot 2000 --bootstrap 399",
                "",
                "Decisao EQUIVALENTE: UCB de W1(inf) < eps, com eps = metade da distancia",
                "de TW2 studentizado ate a normal ou ate TW1, o que for menor.",
                "Decisao KPZ: expoente 1/3 (e 2/3 se geodesico) no IC, e AD contra TW2 nao rejeita.");
    }

    private static final class Config {
        int[] ns = {64, 128, 256};
        int replicates = 400;
        int pilot = 200;
        int bootstrap = 99;
        double alpha = 0.01;
        long seed = 20260325L;
        int threads = 0;
        boolean geodesic = true;
        boolean checkOnly = false;
        boolean refreshConcluded = false;
        String checkpointDir = "results/kpz-checkpoints";
        int checkpointEvery = 100;
        Cost cost = Cost.ABS;
        Distribution weightDist = Distribution.EXPONENTIAL;
        Distribution seqDist = Distribution.NORMAL;
        String outCsv = "results/kpz-equivalence.csv";

        static Config parse(String[] args) {
            Config c = new Config();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                switch (a) {
                    case "--n":
                        c.ns = parseNs(next(args, ++i));
                        break;
                    case "--replicates":
                        c.replicates = Integer.parseInt(next(args, ++i));
                        break;
                    case "--pilot":
                        c.pilot = Integer.parseInt(next(args, ++i));
                        break;
                    case "--bootstrap":
                        c.bootstrap = Integer.parseInt(next(args, ++i));
                        break;
                    case "--alpha":
                        c.alpha = Double.parseDouble(next(args, ++i));
                        break;
                    case "--seed":
                        c.seed = Long.parseLong(next(args, ++i));
                        break;
                    case "--threads":
                        c.threads = Integer.parseInt(next(args, ++i));
                        break;
                    case "--cost":
                        c.cost = Cost.valueOf(next(args, ++i).toUpperCase(Locale.ROOT));
                        break;
                    case "--weight-dist":
                        c.weightDist = parseDist(next(args, ++i));
                        break;
                    case "--seq-dist":
                        c.seqDist = parseDist(next(args, ++i));
                        break;
                    case "--out":
                        c.outCsv = next(args, ++i);
                        break;
                    case "--checkpoint-dir":
                        c.checkpointDir = next(args, ++i);
                        break;
                    case "--checkpoint-every":
                        c.checkpointEvery = Integer.parseInt(next(args, ++i));
                        break;
                    case "--no-geodesic":
                        c.geodesic = false;
                        break;
                    case "--check":
                        c.checkOnly = true;
                        break;
                    case "--refresh-concluded":
                        c.refreshConcluded = true;
                        break;
                    default:
                        throw new IllegalArgumentException("opcao desconhecida: " + a);
                }
            }
            if (c.ns.length < 3) {
                throw new IllegalArgumentException("pelo menos 3 valores de n para extrapolar");
            }
            for (int n : c.ns) {
//                if (n < 8 || n > 1024) {
//                    throw new IllegalArgumentException("8 <= n <= 1024");
//                }
                if (n < 8 || n > 32768) {
                    throw new IllegalArgumentException("8 <= n <= 32768");
                }
            }
            if (c.replicates < 30 || c.pilot < 30) {
                throw new IllegalArgumentException("replicates e pilot >= 30");
            }
            if (c.checkpointEvery < 1) {
                throw new IllegalArgumentException("checkpoint-every >= 1");
            }
            if (!(c.alpha > 0.0 && c.alpha < 0.5)) {
                throw new IllegalArgumentException("alpha em (0, 0.5)");
            }
            Arrays.sort(c.ns);
            return c;
        }

        private static String next(String[] args, int i) {
            if (i >= args.length) {
                throw new IllegalArgumentException("faltou valor para " + args[i - 1]);
            }
            return args[i];
        }

        private static Distribution parseDist(String s) {
            try {
                return Distribution.valueOf(s.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "distribuicao invalida: " + s + " (use EXPONENTIAL, NORMAL, POISSON ou UNIFORM)");
            }
        }

        private static int[] parseNs(String s) {
            String[] p = s.split(",");
            int[] ns = new int[p.length];
            for (int i = 0; i < p.length; i++) {
                ns[i] = Integer.parseInt(p[i].trim());
            }
            return ns;
        }
    }

    private static final class CheckpointStore implements AutoCloseable {
        private static final int MAGIC = 0x4B505A43;
        private static final int VERSION = 2;
        private final RandomAccessFile file;
        private final boolean[] done;
        private final double[] values;
        private final double[] mids;
        private final double[][] features;
        private final int featureCount;
        private final boolean hasMids;
        private final long recordsStart;
        private final int recordBytes;
        private int completed;

        CheckpointStore(Path path, String identity, double[] targetValues, double[] targetMids,
                        double[][] targetFeatures) throws IOException {
            values = targetValues;
            mids = targetMids;
            features = targetFeatures;
            featureCount = features == null ? 0 : features.length;
            hasMids = mids != null;
            done = new boolean[values.length];
            recordBytes = 8 + 8 + featureCount * 8 + 4 + 1;
            file = new RandomAccessFile(path.toFile(), "rw");
            if (file.length() == 0) {
                file.writeInt(MAGIC);
                file.writeInt(VERSION);
                file.writeUTF(identity);
                file.writeInt(values.length);
                file.writeInt(featureCount);
                file.writeBoolean(hasMids);
                recordsStart = file.getFilePointer();
                file.setLength(recordsStart + (long) recordBytes * values.length);
                file.getChannel().force(true);
            } else {
                if (file.readInt() != MAGIC || file.readInt() != VERSION) {
                    file.close();
                    throw new IOException("checkpoint invalido: " + path);
                }
                String savedIdentity = file.readUTF();
                int savedReps = file.readInt();
                int savedFeatures = file.readInt();
                boolean savedMids = file.readBoolean();
                recordsStart = file.getFilePointer();
                long expectedLength = recordsStart + (long) recordBytes * values.length;
                if (!identity.equals(savedIdentity) || savedReps != values.length
                        || savedFeatures != featureCount || savedMids != hasMids
                        || file.length() != expectedLength) {
                    file.close();
                    throw new IOException("parametros/checkpoint incompativeis: " + path
                            + " (use outro --checkpoint-dir ou remova o checkpoint)");
                }
                for (int i = 0; i < values.length; i++) {
                    file.seek(recordsStart + (long) i * recordBytes);
                    double value = file.readDouble();
                    double mid = file.readDouble();
                    double[] row = featureCount == 0 ? null : new double[featureCount];
                    for (int f = 0; f < featureCount; f++) row[f] = file.readDouble();
                    int savedCrc = file.readInt();
                    boolean saved = file.readBoolean();
                    if (saved && savedCrc == checksum(value, mid, row)) {
                        values[i] = value;
                        if (mids != null) mids[i] = mid;
                        if (features != null) {
                            for (int f = 0; f < featureCount; f++) features[f][i] = row[f];
                        }
                        done[i] = true;
                        completed++;
                    }
                }
            }
        }

        synchronized boolean isDone(int index) {
            return done[index];
        }

        synchronized int completedCount() {
            return completed;
        }

        synchronized void write(int index, double value, double mid, double[] row) throws IOException {
            if (done[index]) return;
            file.seek(recordsStart + (long) index * recordBytes);
            file.writeDouble(value);
            file.writeDouble(mid);
            for (int f = 0; f < featureCount; f++) file.writeDouble(row[f]);
            file.writeInt(checksum(value, mid, row));
            file.writeBoolean(true);
            values[index] = value;
            if (mids != null) mids[index] = mid;
            if (features != null) {
                for (int f = 0; f < featureCount; f++) features[f][index] = row[f];
            }
            done[index] = true;
            completed++;
        }

        private static int checksum(double value, double mid, double[] row) {
            CRC32 crc = new CRC32();
            updateLong(crc, Double.doubleToLongBits(value));
            updateLong(crc, Double.doubleToLongBits(mid));
            if (row != null) {
                for (double feature : row) updateLong(crc, Double.doubleToLongBits(feature));
            }
            return (int) crc.getValue();
        }

        private static void updateLong(CRC32 crc, long value) {
            for (int shift = 56; shift >= 0; shift -= 8) crc.update((int) (value >>> shift) & 0xff);
        }

        synchronized void force() throws IOException {
            file.getChannel().force(true);
        }

        @Override
        public synchronized void close() throws IOException {
            file.getChannel().force(true);
            file.close();
        }
    }

    private static final class Bundle {
        Model model;
        int n;
        double[] pilot;
        double[] eval;
        double[] z;
        double[] zSorted;
        double[] mid;
        double[][] pilotFeatures;
        double[][] features;
        double[] residual;
        double[] cEval;
        double residualSd;
        double cSd;
        double r2Pilot;
        double pilotMean;
        double pilotSd;
        double seconds;
    }

    private static final class Report {
        final StringBuilder text = new StringBuilder(8192);
        final StringBuilder csv = new StringBuilder(8192);

        Report() {
            csv.append("modelo,contra,n,chave,valor\n");
        }

        void meta(Config cfg, TracyWidom tw, int threads, double seconds,
                  double dGauss, double dGoe, double epsilon) {
            text.append("Teste de equivalencia DTW / LPP e pertencimento KPZ\n");
            text.append(String.format(Locale.US,
                    "n=%s  eval=%d  pilot=%d  bootstrap=%d  alpha=%.4f  custo=%s  pesos=%s  seq=%s  fios=%d  tempo=%.1fs%n",
                    Arrays.toString(cfg.ns), cfg.replicates, cfg.pilot, cfg.bootstrap, cfg.alpha,
                    cfg.cost, cfg.weightDist, cfg.seqDist, threads, seconds));
            text.append(String.format(Locale.US,
                    "TW2 numerico media=%.4f var=%.4f assim=%.4f | W1(TW2*,N)=%.4f W1(TW2*,TW1*)=%.4f eps=%.4f%n",
                    tw.mean2(), tw.variance2(), tw.skewness2(), dGauss, dGoe, epsilon));
            text.append("Escala teorica LPP: (L-4n)/(2^(4/3) n^(1/3)) -> TW2. DTW usa studentizacao independente do piloto.\n\n");
            line(Model.LPP, 0, "threads", threads);
            line(Model.LPP, 0, "segundos", seconds);
            line(Model.LPP, 0, "epsilon", epsilon);
            line(Model.LPP, 0, "w1_tw2_normal", dGauss);
            line(Model.LPP, 0, "w1_tw2_tw1", dGoe);
            line(Model.LPP, 0, "tw2_media_numerica", tw.mean2());
            line(Model.LPP, 0, "tw2_var_numerica", tw.variance2());
            line(Model.LPP, 0, "weight_dist", cfg.weightDist.name());
            line(Model.LPP, 0, "seq_dist", cfg.seqDist.name());
        }

        void section(String title) {
            text.append('\n').append(title).append('\n');
        }

        void line(Model model, int n, String key, double value) {
            csv.append(model.name()).append(",,")
                    .append(n).append(',').append(key).append(',')
                    .append(String.format(Locale.US, "%.10g", value)).append('\n');
        }

        void line(Model model, int n, String key, String value) {
            csv.append(model.name()).append(",,")
                    .append(n).append(',').append(key).append(',')
                    .append(value).append('\n');
        }

        void linePair(Model a, Model b, int n, String key, double value) {
            csv.append(a.name()).append(',').append(b.name()).append(',')
                    .append(n).append(',').append(key).append(',')
                    .append(String.format(Locale.US, "%.10g", value)).append('\n');
        }

        void linePair(Model a, Model b, int n, String key, String value) {
            csv.append(a.name()).append(',').append(b.name()).append(',')
                    .append(n).append(',').append(key).append(',')
                    .append(value).append('\n');
        }
    }
}
