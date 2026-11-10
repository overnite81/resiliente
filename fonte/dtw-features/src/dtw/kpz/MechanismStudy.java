package dtw.kpz;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/** Experimentos adicionais sobre modo comum, geometria e ambiente separável. */
public final class MechanismStudy {
    private static final int MODEL_DT = DirectedPassage.Model.DTW.id;
    private static final String HEADER = "phase,dist,n,replicate,seed,model,opt,diag,gain,residual,rms_offset,max_offset,mean_visits,mid_column,control_lpp3,control_fpp3,lpp3_rms_offset,lpp3_max_offset";
    private MechanismStudy() {}

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Config c = Config.parse(args);
        Files.createDirectories(c.out);
        switch (c.phase) {
            case "check": check(); break;
            case "reanalysis": reanalysis(c); break;
            case "probe": probe(c); break;
            case "interpolation": interpolation(c); break;
            case "ranks": ranks(c); break;
            case "permutations": permutations(c); break;
            default: throw new IllegalArgumentException("phase: check|reanalysis|probe|interpolation|ranks|permutations");
        }
    }

    private static void check() {
        TracyWidom tw = TracyWidom.build();
        require(tw.reliable(), "Tracy-Widom: " + tw.diagnostic());
        int n = 24; long seed = 318273L; int rep = 9;
        DirectedPassage.Outcome expected = DirectedPassage.simulate(DirectedPassage.Model.DTW, n, seed, rep,
                true, DirectedPassage.Cost.ABS, Distribution.EXPONENTIAL, Distribution.NORMAL);
        Xoroshiro128 rng = new Xoroshiro128(Xoroshiro128.mix(seed, MODEL_DT, n, rep));
        double[] x = SeparablePath.sequence(n, rng, Distribution.NORMAL);
        double[] y = SeparablePath.sequence(n, rng, Distribution.NORMAL);
        double[] a = new double[n], b = new double[n]; byte[] pred = new byte[n*n];
        SeparablePath.Metrics m = SeparablePath.dtw(x,y,a,b,pred,true);
        close(m.cost, expected.value, "DTW ядро");
        close(m.diagonal/n, expected.emparelhado, "DIAG");
        require(m.midColumn == expected.midColumn, "midColumn");
        Xoroshiro128 r2 = new Xoroshiro128(Xoroshiro128.mix(seed, MODEL_DT, n, rep));
        double[] x2 = SeparablePath.sequence(n,r2,Distribution.NORMAL), y2 = SeparablePath.sequence(n,r2,Distribution.NORMAL);
        double[] ranksX=SeparablePath.ranks(x2), ranksY=SeparablePath.ranks(y2);
        double raw=0, rankCost=0;
        for(int i=0;i<n;i++){raw+=Math.abs(x2[i]-y2[i]);rankCost+=Math.abs(ranksX[i]-ranksY[i]);}
        require(Math.abs(raw-rankCost)>1e-6,"rank transform must be a distinct cost");
        Xoroshiro128 lr = new Xoroshiro128(Xoroshiro128.mix(seed, DirectedPassage.Model.LPP3.id,n,rep));
        double lpp3 = DirectedPassage.lpp3(n,lr,new double[n],new double[n],null,Distribution.EXPONENTIAL);
        close(lpp3, DirectedPassage.simulate(DirectedPassage.Model.LPP3,n,seed,rep,false,
                DirectedPassage.Cost.ABS,Distribution.EXPONENTIAL,Distribution.NORMAL).value,"LPP3");
        Xoroshiro128 fr = new Xoroshiro128(Xoroshiro128.mix(seed, DirectedPassage.Model.FPP3.id,n,rep));
        double fpp3 = DirectedPassage.fpp3(n,fr,new double[n],new double[n],null,Distribution.EXPONENTIAL);
        close(fpp3, DirectedPassage.simulate(DirectedPassage.Model.FPP3,n,seed,rep,false,
                DirectedPassage.Cost.ABS,Distribution.EXPONENTIAL,Distribution.NORMAL).value,"FPP3");
        double[] omega = new double[n*n];
        Xoroshiro128 weightRng=new Xoroshiro128(Xoroshiro128.mix(seed,DirectedPassage.Model.LPP3.id,n,rep));
        for(int i=0;i<omega.length;i++)omega[i]=weightRng.nextExponential();
        byte[] maxPred=new byte[n*n];
        SeparablePath.Metrics max0=SeparablePath.mixed(x,y,omega,0,new double[n],new double[n],maxPred,false,true);
        close(max0.cost,lpp3,"interpolation lambda=0 maximum / LPP3");
        DirectedPassage.Outcome lppPath=DirectedPassage.simulate(DirectedPassage.Model.LPP3,n,seed,rep,true,
                DirectedPassage.Cost.ABS,Distribution.EXPONENTIAL,Distribution.NORMAL);
        require(max0.midColumn==lppPath.midColumn,"interpolation lambda=0 maximum midColumn");

        Xoroshiro128 minWeightRng=new Xoroshiro128(Xoroshiro128.mix(seed,DirectedPassage.Model.FPP3.id,n,rep));
        for(int i=0;i<omega.length;i++)omega[i]=minWeightRng.nextExponential();
        byte[] minPred=new byte[n*n];
        SeparablePath.Metrics min0=SeparablePath.mixed(x,y,omega,0,new double[n],new double[n],minPred,true,true);
        close(min0.cost,fpp3,"interpolation lambda=0 minimum / FPP3");
        DirectedPassage.Outcome fppPath=DirectedPassage.simulate(DirectedPassage.Model.FPP3,n,seed,rep,true,
                DirectedPassage.Cost.ABS,Distribution.EXPONENTIAL,Distribution.NORMAL);
        require(min0.midColumn==fppPath.midColumn,"interpolation lambda=0 minimum midColumn");

        SeparablePath.Metrics min1=SeparablePath.mixed(x,y,omega,1,new double[n],new double[n],null,true,false);
        close(min1.cost,m.cost,"interpolation lambda=1 minimum / DTW");
        System.out.println("CHECK_OK " + tw.diagnostic());
    }

    private static void probe(Config c) throws Exception {
        Path raw = c.out.resolve("probe-"+c.dist.name().toLowerCase(Locale.ROOT)+"-raw.csv");
        Set<String> done = completed(raw, c.signature());
        try (BufferedWriter w = Files.newBufferedWriter(raw, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (Files.size(raw)==0) { w.write("#config="+c.signature()); w.newLine(); w.write(HEADER); w.newLine(); w.flush(); }
            for (int n : c.ns) {
                runProbePhase(c,n,"pilot",c.pilot,0,done,w);
                w.flush();
                // Ajuste independente lido do journal; evita recalcular o piloto ao retomar.
                List<String> journal=Files.readAllLines(raw);
                double[] px=column(journal,"pilot",n,7,c.dist), py=column(journal,"pilot",n,6,c.dist);
                if(px.length!=c.pilot||py.length!=c.pilot)throw new IOException("piloto incompleto no journal para n="+n);
                double[] ab=EmpiricalStats.wlsSlope(px,py,null);
                runProbePhase(c,n,"eval",c.replicates,c.pilot,done,w,ab);
                w.flush();
            }
        }
        summarizeProbe(c,raw);
    }

    private static void runProbePhase(Config c,int n,String phase,int count,int offset,Set<String> done,BufferedWriter w) throws Exception {
        runProbePhase(c,n,phase,count,offset,done,w,new double[]{0,0});
    }
    private static void runProbePhase(Config c,int n,String phase,int count,int offset,Set<String> done,BufferedWriter w,double[] ab) throws Exception {
        List<Integer> todo=new ArrayList<>();
        for(int z=0;z<count;z++){int r=offset+z; if(!done.contains(phase+"|"+c.dist+"|"+n+"|"+r)) todo.add(r);}
        Object lock=new Object(); AtomicInteger completed=new AtomicInteger();
        IntStream.range(0,todo.size()).parallel().forEach(ix->{
            int r=todo.get(ix); Observation o=observe(c,n,r,phase.equals("eval"));
            double residual=phase.equals("pilot")?Double.NaN:o.opt-(ab[0]+ab[1]*o.diag);
            String line=String.format(Locale.US,"%s,%s,%d,%d,%d,DTW,%.12g,%.12g,%.12g,%.12g,%.8g,%d,%.8g,%d,%.12g,%.12g,%.8g,%d", 
                    phase,c.dist,n,r,c.seed,o.opt,o.diag,o.diag-o.opt,residual,o.rms,o.max,o.visits,o.mid,o.lpp3,o.fpp3,o.lpp3Rms,o.lpp3Max);
            synchronized(lock){try{w.write(line);w.newLine();int q=completed.incrementAndGet();if(q%100==0)w.flush();}catch(IOException e){throw new RuntimeException(e);}}
        });
        System.out.printf("probe %s %s n=%d: %d novos%n",phase,c.dist,n,todo.size());
        // Existing records are read before run; update set so subsequent phase logic sees these.
        for(int r:todo) done.add(phase+"|"+c.dist+"|"+n+"|"+r);
    }

    private static Observation observe(Config c,int n,int replicate,boolean path) {
        Xoroshiro128 rng=new Xoroshiro128(Xoroshiro128.mix(c.seed,MODEL_DT,n,replicate));
        double[] x=SeparablePath.sequence(n,rng,c.dist), y=SeparablePath.sequence(n,rng,c.dist);
        byte[] pred=path?new byte[n*n]:null;
        SeparablePath.Metrics m=SeparablePath.dtw(x,y,new double[n],new double[n],pred,path);
        Xoroshiro128 lppRng=new Xoroshiro128(Xoroshiro128.mix(c.seed,DirectedPassage.Model.LPP3.id,n,replicate));
        byte[] lppPred=path?new byte[n*n]:null;
        double lpp3=DirectedPassage.lpp3(n,lppRng,new double[n],new double[n],lppPred,Distribution.EXPONENTIAL);
        SeparablePath.Metrics lppGeometry=SeparablePath.metrics(lpp3,0,lppPred,n,path);
        Xoroshiro128 fppRng=new Xoroshiro128(Xoroshiro128.mix(c.seed,DirectedPassage.Model.FPP3.id,n,replicate));
        double fpp3=DirectedPassage.fpp3(n,fppRng,new double[n],new double[n],null,Distribution.EXPONENTIAL);
        return new Observation(m.cost,m.diagonal,m.rmsOffset,m.maxOffset,m.meanVisits,m.midColumn,lpp3,fpp3,lppGeometry.rmsOffset,lppGeometry.maxOffset);
    }
    private static final class Observation {
        final double opt,diag,rms,visits,lpp3,fpp3,lpp3Rms; final int max,mid,lpp3Max;
        Observation(double a,double b,double c,int d,double e,int f,double g,double h,double q,int z){opt=a;diag=b;rms=c;max=d;visits=e;mid=f;lpp3=g;fpp3=h;lpp3Rms=q;lpp3Max=z;}
    }

    private static void summarizeProbe(Config c,Path raw) throws IOException {
        List<String> lines=Files.readAllLines(raw); List<String> out=new ArrayList<>();
        out.add("phase,dist,n,measure,count,mean,sd,skewness,excess_kurtosis,w1_normal,w1_reflected_tw2");
        TracyWidom tw=TracyWidom.build();
        for(int n:c.ns) {
            List<String[]> pilotRows=rowsFor(lines,"pilot",c.dist,n);
            for(String phase:new String[]{"pilot","eval"}) {
                List<String[]> rows=phase.equals("pilot")?pilotRows:rowsFor(lines,phase,c.dist,n);
                for(int col=6;col<=9;col++) {
                    if(phase.equals("pilot")&&col==9) continue;
                    double[] v=columnRows(rows,col);
                    double[] ref=phase.equals("pilot")?new double[]{EmpiricalStats.mean(v),EmpiricalStats.sd(v)}:pilotReference(pilotRows,col);
                    if(col==9&&phase.equals("eval"))ref=pilotResidualReference(pilotRows);
                    appendSummary(out,phase,c.dist,n,new String[]{"opt","diag","gain","residual"}[col-6],v,tw,ref[0],ref[1],true);
                }
                if(phase.equals("eval")&&!rows.isEmpty()) for(int col=10;col<=17;col++) {
                    double[] v=columnRows(rows,col);
                    String name=new String[]{"rms_offset","max_offset","mean_visits","mid_column","control_lpp3","control_fpp3","lpp3_rms_offset","lpp3_max_offset"}[col-10];
                    boolean distributional=col==14||col==15;
                    double[] ref=distributional?pilotReference(pilotRows,col):new double[]{EmpiricalStats.mean(v),EmpiricalStats.sd(v)};
                    appendSummary(out,phase,c.dist,n,name,v,tw,ref[0],ref[1],distributional);
                }
            }
        }
        Files.write(c.out.resolve("probe-summary.csv"),out);
        writeScaling(c,out);
        writeAssociation(c,lines);
        System.out.println("Resumo: "+c.out.resolve("probe-summary.csv"));
    }
    private static void writeScaling(Config c,List<String> summary) throws IOException {
        List<String> rows=new ArrayList<>(); rows.add("dist,measure,n1,n2,sd1,sd2,chi_local,var_over_n");
        List<Integer> ns=Arrays.stream(c.ns).sorted().boxed().toList();
        for(String measure:new String[]{"opt","diag","gain","residual","control_lpp3","control_fpp3","lpp3_rms_offset","lpp3_max_offset"}) {
            double previous=Double.NaN; int prior=0;
            for(int n:ns) {
                double sd=Double.NaN;
                for(String line:summary){String[] p=line.split(",");if(p.length>=7&&p[0].equals("eval")&&p[2].equals(Integer.toString(n))&&p[3].equals(measure)){sd=Double.parseDouble(p[6]);break;}}
                if(Double.isFinite(sd)&&sd>0&&Double.isFinite(previous)) rows.add(String.format(Locale.US,"%s,%s,%d,%d,%.10g,%.10g,%.10g,%.10g",c.dist,measure,prior,n,previous,sd,Math.log(sd/previous)/Math.log((double)n/prior),sd*sd/n));
                prior=n; previous=sd;
            }
        }
        Files.write(c.out.resolve("probe-scaling.csv"),rows);
    }
    private static void writeAssociation(Config c,List<String> lines) throws IOException {
        List<String> out=new ArrayList<>();out.add("dist,n,pilot_r2,eval_corr_opt_diag,eval_r2_from_pilot,eval_var_over_n");
        for(int n:c.ns) {
            double[] px=column(lines,"pilot",n,7,c.dist),py=column(lines,"pilot",n,6,c.dist),ex=column(lines,"eval",n,7,c.dist),ey=column(lines,"eval",n,6,c.dist);
            if(px.length<3||ex.length<3)continue;
            double[] ab=EmpiricalStats.wlsSlope(px,py,null);double pm=EmpiricalStats.mean(py),ss=0,se=0;
            for(double v:py)ss+=(v-pm)*(v-pm);for(int i=0;i<px.length;i++){double e=py[i]-ab[0]-ab[1]*px[i];se+=e*e;}
            double xm=EmpiricalStats.mean(ex),ym=EmpiricalStats.mean(ey),cov=0,vx=0,vy=0,evalSe=0,evalSs=0;
            for(int i=0;i<ex.length;i++){double dx=ex[i]-xm,dy=ey[i]-ym;cov+=dx*dy;vx+=dx*dx;vy+=dy*dy;double e=ey[i]-ab[0]-ab[1]*ex[i];evalSe+=e*e;evalSs+=dy*dy;}
            out.add(String.format(Locale.US,"%s,%d,%.10g,%.10g,%.10g,%.10g",c.dist,n,1-se/ss,cov/Math.sqrt(vx*vy),1-evalSe/evalSs,EmpiricalStats.variance(ey)/n));
        }
        Files.write(c.out.resolve("probe-association.csv"),out);
    }
    private static double[] column(List<String> lines,String phase,int n,int col,Distribution dist){
        List<Double> v=new ArrayList<>();for(String line:lines){if(line.startsWith("#")||line.startsWith("phase,"))continue;String[] p=line.split(",");if(p.length>col&&p[0].equals(phase)&&p[1].equals(dist.name())&&Integer.parseInt(p[2])==n)v.add(Double.parseDouble(p[col]));}
        return v.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private static List<String[]> rowsFor(List<String> lines,String phase,Distribution dist,int n){
        List<String[]> rows=new ArrayList<>();for(String line:lines){if(line.startsWith("#")||line.startsWith("phase,"))continue;String[] p=line.split(",");if(p.length>=18&&p[0].equals(phase)&&p[1].equals(dist.name())&&Integer.parseInt(p[2])==n)rows.add(p);}return rows;
    }
    private static double[] columnRows(List<String[]> rows,int col){double[] v=new double[rows.size()];for(int i=0;i<v.length;i++)v[i]=Double.parseDouble(rows.get(i)[col]);return v;}
    private static double[] pilotReference(List<String[]> rows,int col){double[] v=columnRows(rows,col);return new double[]{EmpiricalStats.mean(v),EmpiricalStats.sd(v)};}
    private static double[] pilotResidualReference(List<String[]> rows){
        double[] x=columnRows(rows,7),y=columnRows(rows,6),ab=EmpiricalStats.wlsSlope(x,y,null),r=new double[x.length];
        for(int i=0;i<r.length;i++)r[i]=y[i]-ab[0]-ab[1]*x[i];return new double[]{EmpiricalStats.mean(r),EmpiricalStats.sd(r)};
    }
    private static void appendSummary(List<String> out,String phase,Distribution dist,int n,String name,double[] v,TracyWidom tw,double refMean,double refSd,boolean compare){
        if(v.length<2)return; double mean=EmpiricalStats.mean(v),sd=EmpiricalStats.sd(v);double wN=Double.NaN,wTw=Double.NaN;
        if(compare&&refSd>0&&Double.isFinite(refSd)){
            double[] z=new double[v.length];EmpiricalStats.studentize(v,z,refMean,refSd);Arrays.sort(z);
            wN=EmpiricalStats.wasserstein1Sorted(z,normalQuantiles(v.length));
            wTw=EmpiricalStats.wasserstein1Sorted(z,twQuantiles(tw,v.length));
        }
        out.add(String.format(Locale.US,"%s,%s,%d,%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g",phase,dist,n,name,v.length,mean,sd,EmpiricalStats.skewness(v),EmpiricalStats.excessKurtosis(v),wN,wTw));
    }
    private static double[] normalQuantiles(int n){double[] q=new double[n];for(int i=0;i<n;i++)q[i]=EmpiricalStats.normalQuantile((i+.5)/n);return q;}
    private static double[] twQuantiles(TracyWidom tw,int n){double[] q=new double[n];for(int i=0;i<n;i++)q[i]=tw.quantile2ReflectedStudentized((i+.5)/n);return q;}

    private static void interpolation(Config c) throws Exception {
        Path out=c.out.resolve("interpolation.csv");
        try(BufferedWriter w=Files.newBufferedWriter(out,StandardOpenOption.CREATE,StandardOpenOption.APPEND)){
            if(Files.size(out)==0){w.write("dist,n,replicate,lambda,mode,cost,rms_offset,max_offset,mean_visits");w.newLine();}
            double[] fixed={0,.25,.5,.75,1};
            for(int n:c.ns) {
                List<Double> lambdas=new ArrayList<>();for(double l:fixed)lambdas.add(l);
                double scale=Math.pow(n,1.0/3.0-c.chi);
                for(double u:new double[]{.25,.5,1,2,4})lambdas.add(Math.min(1,u*scale));
                for(int r=0;r<c.replicates;r++) {
                    Xoroshiro128 rng=new Xoroshiro128(Xoroshiro128.mix(c.seed,MODEL_DT,n,r));
                    double[] x=SeparablePath.sequence(n,rng,c.dist),y=SeparablePath.sequence(n,rng,c.dist),omega=new double[n*n];
                    for(int k=0;k<omega.length;k++)omega[k]=rng.nextExponential();
                    for(double lambda:lambdas) for(boolean minimum:new boolean[]{true,false}) {
                        byte[] pred=(lambda==0||lambda==1)?new byte[n*n]:null;
                        SeparablePath.Metrics m=SeparablePath.mixed(x,y,omega,lambda,new double[n],new double[n],pred,minimum,pred!=null);
                        w.write(String.format(Locale.US,"%s,%d,%d,%.8g,%s,%.12g,%.8g,%d,%.8g%n",c.dist,n,r,lambda,minimum?"min":"max",m.cost,m.rmsOffset,m.maxOffset,m.meanVisits));
                    }
                    if(r%50==0)w.flush();
                }
            }
        }
    }

    private static void ranks(Config c) throws Exception {
        Path out=c.out.resolve("ranks.csv");
        try(BufferedWriter w=Files.newBufferedWriter(out,StandardOpenOption.CREATE,StandardOpenOption.APPEND)){
            if(Files.size(out)==0){w.write("dist,n,replicate,rank_dtw,raw_dtw,diag_rank");w.newLine();}
            for(int n:c.ns)for(int r=0;r<c.replicates;r++){
                Xoroshiro128 rng=new Xoroshiro128(Xoroshiro128.mix(c.seed,MODEL_DT,n,r));
                double[] x=SeparablePath.sequence(n,rng,c.dist),y=SeparablePath.sequence(n,rng,c.dist);
                double[] rx=SeparablePath.ranks(x),ry=SeparablePath.ranks(y);double[][] a=new double[2][n];
                double rank=DirectedPassage.dtw(n,rx,ry,a[0],a[1],null,true);
                double raw=DirectedPassage.dtw(n,x,y,a[0],a[1],null,true),diag=0;
                for(int i=0;i<n;i++)diag+=Math.abs(rx[i]-ry[i]);
                w.write(String.format(Locale.US,"%s,%d,%d,%.12g,%.12g,%.12g%n",c.dist,n,r,rank,raw,diag));
            }
        }
    }

    private static void permutations(Config c) throws IOException {
        Path out=c.out.resolve("permutations.csv");
        try(BufferedWriter w=Files.newBufferedWriter(out,StandardOpenOption.CREATE,StandardOpenOption.APPEND)){
            if(Files.size(out)==0){w.write("n,replicate,dtw_perm,diagonal_perm");w.newLine();}
            for(int n:c.ns)for(int r=0;r<c.replicates;r++){
                Xoroshiro128 rng=new Xoroshiro128(Xoroshiro128.mix(c.seed,MODEL_DT,n,r));
                double[] x=new double[n],y=new double[n];for(int i=0;i<n;i++){x[i]=i;y[i]=i;}
                shuffle(x,rng);shuffle(y,rng);double diag=0;
                for(int i=0;i<n;i++)diag+=Math.abs(x[i]-y[i]);
                double dtw=DirectedPassage.dtw(n,x,y,new double[n],new double[n],null,true);
                w.write(String.format(Locale.US,"%d,%d,%.12g,%.12g%n",n,r,dtw,diag));
            }
        }
    }
    private static void shuffle(double[] x,Xoroshiro128 rng){for(int i=x.length-1;i>0;i--){int j=rng.nextInt(i+1);double t=x[i];x[i]=x[j];x[j]=t;}}

    private static void reanalysis(Config c) throws IOException {
        Path[] inputs={Path.of("results/kpz-equivalence.csv"),Path.of("results/kpz-concluido.csv"),Path.of("results/kpz-concluido-exp.csv"),Path.of("results/kpz-concluido-unif.csv")};
        List<String> out=new ArrayList<>();out.add("file,model,n,key,value,derived");
        for(Path f:inputs){if(!Files.exists(f))continue;for(String line:Files.readAllLines(f)){String[] p=line.split(",");if(p.length<5||p[0].equals("modelo"))continue;
            try{int n=Integer.parseInt(p[2]);double v=Double.parseDouble(p[4]);String d="";
                if(p[3].equals("assimetria"))d=String.format(Locale.US,"abs_skew_sqrt_n=%.10g;dist_to_TW2=%.10g;dist_to_reflected_TW2=%.10g",Math.abs(v)*Math.sqrt(n),Math.abs(v+TracyWidom.TW2_SKEW_REF),Math.abs(v-TracyWidom.TW2_SKEW_REF));
                else if(p[3].equals("razao_sd_2n"))d=String.format(Locale.US,"chi_local=%.10g",Math.log(v)/Math.log(2));
                else if(p[3].equals("curtose_excedente"))d=String.format(Locale.US,"kurtosis_times_n=%.10g",v*n);
                out.add(String.format(Locale.US,"%s,%s,%d,%s,%.12g,%s",f,p[0],n,p[3],v,d));
            }catch(NumberFormatException ignored){}
        }}
        Files.write(c.out.resolve("reanalysis.csv"),out);System.out.println("Reanálise: "+c.out.resolve("reanalysis.csv"));
    }
    private static Set<String> completed(Path raw,String signature)throws IOException{
        Set<String> done=new HashSet<>();if(!Files.exists(raw))return done;
        List<String> lines=Files.readAllLines(raw);if(!lines.isEmpty()&&lines.get(0).startsWith("#config=")&&!lines.get(0).equals("#config="+signature))throw new IOException("checkpoint com configuração diferente: "+raw);
        for(String line:lines){if(line.startsWith("#")||line.startsWith("phase,"))continue;String[] p=line.split(",");if(p.length>=4)done.add(p[0]+"|"+p[1]+"|"+p[2]+"|"+p[3]);}return done;
    }
    private static void close(double a,double b,String name){if(Math.abs(a-b)>1e-10*Math.max(1,Math.abs(b)))throw new IllegalStateException(name+": "+a+" != "+b);}
    private static void require(boolean b,String message){if(!b)throw new IllegalStateException(message);}

    private static final class Config {
        String phase="probe"; int[] ns={64,128,256,512,1024,2048}; int replicates=5000,pilot=2000; long seed=20260325L; int threads=Runtime.getRuntime().availableProcessors(); double chi=.4;
        Distribution dist=Distribution.NORMAL; Path out=Path.of("results/mechanism");
        String signature(){return phase+";"+Arrays.toString(ns)+";"+replicates+";"+pilot+";"+seed+";"+dist;}
        static Config parse(String[] args){Config c=new Config();for(int i=0;i<args.length;i++){String a=args[i];switch(a){case "--phase":c.phase=args[++i];break;case "--n":c.ns=Arrays.stream(args[++i].split(",")).mapToInt(Integer::parseInt).toArray();break;case "--replicates":c.replicates=Integer.parseInt(args[++i]);break;case "--pilot":c.pilot=Integer.parseInt(args[++i]);break;case "--seed":c.seed=Long.parseLong(args[++i]);break;case "--seq-dist":c.dist=Distribution.valueOf(args[++i].toUpperCase(Locale.ROOT));break;case "--chi":c.chi=Double.parseDouble(args[++i]);break;case "--out":c.out=Path.of(args[++i]);break;default:throw new IllegalArgumentException("opção desconhecida: "+a);}}
            if(c.ns.length==0||c.replicates<2||c.pilot<3)throw new IllegalArgumentException("n/repetições inválidos");return c;}
    }
}
