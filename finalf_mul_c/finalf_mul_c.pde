/** //<>//
 * Kinect v2 Overhead Tracking with Projector
 * Keys:
 * b : capture floor
 * o : toggle overlays (debug view)
 * c : start/continue calibration
 * d : discard current red target & advance
 * u : undo last accepted calibration pair
 * r : fit H+k1 (RANSAC and refine) and (re)build hull
 * n : clear calibration points (restart calibration)
 * l : load homography
 * s : save homography
 * p : open/close projector mirror window
 * t : toggle target mode
 * G : regenerate random targets (RANDOM mode)
 * [ / ] : ACCEPT_THRESH -/+ (px) for hull keeping
 * , / . : KEEP_FRACTION -/+ for hull keeping
 * h : toggle hull shrink (1.00 - 0.97)
 * k : toggle test pattern on projector
 * g : toggle hull gating ON/OFF (use hull as safety mask)
 * f : toggle fractal trails ON/OFF (performance toggle)
 *
 * Workflow:
 *   1. press 'b'  (capture floor)
 *   2. press 'p'  (open projector window)
 *   3. press 'v'  (previewTargets=true)
 *   4. look at dots on floor/projector
 *   5. hit ENTER  (previewTargets=false, calibrating=true)
 *   6. click Kinect points one by one in LEFT pane
 *   7. press 'r' to fit homography and build hull
 */

import KinectPV2.*;
import gab.opencv.*;
import java.util.*;
import java.awt.Rectangle;
import java.awt.GraphicsEnvironment;
import java.awt.GraphicsDevice;
import java.io.File;
import java.util.Comparator;
import processing.sound.*;

KinectPV2 kinect;
final int W = 512, H = 424;

final int DS = 2; // downscale factor (2 = 256x212)
final int DS_W = W / DS;
final int DS_H = H / DS;
final int MAX_CONTOURS = 24; // cap contours per frame for surety

OpenCV opencvSmall;
PImage maskSmall;

int[] floorMM = null;
boolean debugView = true;

int H_MIN = 200, H_MAX = 2200;
int MIN_AREA = 1500, MAX_AREA = 200000;

float scaleDisp = 1.8f;
float gap = 40;
float paneW, paneH;
float leftX = 0, leftY = 0;
float rightX, rightY = 0;

final int PROCESS_EVERY = 1; 

final int DEPTH_IMG_EVERY = 3;
PImage lastDepthImg = null;

ArrayList<Rectangle> lastDetBoxes   = new ArrayList<Rectangle>();
ArrayList<PVector>   lastDetCenters = new ArrayList<PVector>();

ArrayList<Rectangle> boxesReuse   = new ArrayList<Rectangle>();
ArrayList<PVector>   centersReuse = new ArrayList<PVector>();

ArrayList<PVector> dotsLocalReuse      = new ArrayList<PVector>();
ArrayList<PVector> ptsForPreviewReuse  = new ArrayList<PVector>();

int nextID = 1;

// base spacing range between footprints (px) - RANDOMIZED
float trailMinDistMin = 6.0f;   // smaller and denser
float trailMinDistMax = 50.0f;  // larger and sparser

//Different Fractal styles

enum PatternType {
  SINGLE_RING,
  DOUBLE_RING,
  SPIRAL_ARC,
  CLUSTER_FLOWER
}

class FractalStyle {
  PatternType pattern;

  int   palIdx; // palette index
  int   maxDepth; // logical depth for color mapping
  float scaleChild; // child radius / parent radius
  float gapFactor; // spacing between parent & child ring
  float twist; // extra rotation per child ring
  boolean animate; // whether this style spins/breathes
  float rotSpeedDeg; // deg/sec
  float breatheAmp; // size modulation amplitude
  float breatheHz; // breathing speed
  float sizeMul; // overall size multiplier
}

// Generate a random style when a track or step is created
FractalStyle makeRandomStyle() {
  FractalStyle s = new FractalStyle();
  s.palIdx      = (int)random(0, 5); // 5 palettes in ProjectorWindow
  s.maxDepth    = (int)random(2, 5);
  s.scaleChild  = random(0.42f, 0.54f);
  s.gapFactor   = random(0.06f, 0.16f);
  s.twist       = random(0.10f, 0.26f);
  s.animate     = (random(1) < 0.85f);
  s.rotSpeedDeg = random(6.0f, 22.0f);
  s.breatheAmp  = random(0.00f, 0.08f);
  s.breatheHz   = random(0.12f, 0.45f);
  s.sizeMul     = random(0.6f, 1.8f);

  float r = random(1);
  if (r < 0.25f) s.pattern = PatternType.SINGLE_RING;
  else if (r < 0.50f) s.pattern = PatternType.DOUBLE_RING;
  else if (r < 0.75f) s.pattern = PatternType.SPIRAL_ARC;
  else s.pattern = PatternType.CLUSTER_FLOWER;

  return s;
}

// footprint size
float trailBaseRadius = 14.0f;
// was 5.0f, reduced for lighter load
final float TRAIL_LIFETIME = 4.0f; // seconds each footprint stays alive

// ON/OFF toggle
boolean enableTrails = true;

class TrailStep {
  PVector pos; // projector local coords (pane space, after warp)
  float createdT; // seconds since start (for animation timing)
  FractalStyle style; // style frozen at creation
  float radius; // per-step base radius for random size
}

ArrayList<TrailStep> trailSteps = new ArrayList<TrailStep>();

boolean enableRandomSpawns = true; 
// LOWER rate so performance is better (was 2.0f)
float   randomSpawnsPerSecond = 1.3f; // expected extra footprints / second
// lower cap (was 320)
int maxRandomTrailSteps = 220; // hard cap so projector draw stays cheap

class Track {
  int id; PVector pos, vel, raw; Rectangle lastBox;
  int age, missed, hits; boolean confirmed;

  PVector foot = new PVector(-1, -1);
  boolean hasFoot = false;
  int footMiss = 0;

  PVector projPos = new PVector(-1, -1);
  boolean hasProj = false;

  PVector lastTrailPos = null; // last footprint position

  FractalStyle style; // per-person fractal style

  Track(PVector p, Rectangle r) {
    id = nextID++;
    pos = p.copy();
    vel = new PVector();
    raw = p.copy();
    lastBox = r;
    age = 0;
    missed = 0;
    hits = 0;
    confirmed = false;

    // assign a random style when track is created
    style = makeRandomStyle();
  }
}

ArrayList<Track> tracks = new ArrayList<Track>();

// add a new footprint with random size
void addTrailStep(PVector warped, FractalStyle style){
  if (!enableTrails || style == null) return;
  TrailStep s = new TrailStep();
  s.pos = warped.copy();
  s.createdT = millis() / 1000.0f;
  s.style = style;

  // Randomize footprint size around the global base radius.
  float jitter = random(0.3f, 2.8f);
  s.radius = trailBaseRadius * jitter;

  // Occasionally drop a bigger "splash" step
  if (random(1) < 0.08f) {
    s.radius = trailBaseRadius * random(1.6f, 2.3f);
  }

  synchronized (trailSteps) {
    trailSteps.add(s);
  }
}

// Random spacing between trail steps (per step, per track)
void maybeAddTrailStep(Track t, PVector warped){
  if (!enableTrails || warped == null) return;
  if (t.style == null) {
    t.style = makeRandomStyle();
  }

  // first footprint for this track
  if (t.lastTrailPos == null) {
    t.lastTrailPos = warped.copy();
    addTrailStep(warped, t.style);
    return;
  }

  float dx = warped.x - t.lastTrailPos.x;
  float dy = warped.y - t.lastTrailPos.y;
  float dist2 = dx*dx + dy*dy;

  // pick a fresh random spacing threshold for THIS step
  float minD = random(trailMinDistMin, trailMinDistMax);

  if (dist2 >= minD * minD) {
    t.lastTrailPos.set(warped);
    addTrailStep(warped, t.style);
  }
}

// Prune old footprints (and hull-outside ones) to keep draw cost down
void pruneTrailSteps() {
  synchronized (trailSteps) {
    if (!enableTrails) {
      trailSteps.clear();
      return;
    }
    float now = millis() / 1000.0f;
    for (int i = trailSteps.size() - 1; i >= 0; i--) {
      TrailStep s = trailSteps.get(i);
      boolean tooOld = (now - s.createdT > TRAIL_LIFETIME);
      boolean outHull = false;
      if (useHullGate && hullProj.size() >= 3) {
        outHull = !pointInPoly(hullProj, s.pos.x, s.pos.y);
      }
      if (tooOld || outHull) {
        trailSteps.remove(i);
      }
    }
  }
}

// Spawn a single random footprint
void spawnRandomTrailStepInsideHull() {
  if (!enableTrails || !enableRandomSpawns) return;

  // Soft cap total footprints (tracked and random) for performance
  if (trailSteps.size() >= maxRandomTrailSteps) return;

  boolean haveHull = (useHullGate && hullProj.size() >= 3);

  float minX, minY, maxX, maxY;

  if (haveHull) {
    // Build a simple bounding box around the hull to sample from
    minX = Float.MAX_VALUE;
    minY = Float.MAX_VALUE;
    maxX = -Float.MAX_VALUE;
    maxY = -Float.MAX_VALUE;
    for (PVector v : hullProj) {
      if (v.x < minX) minX = v.x;
      if (v.x > maxX) maxX = v.x;
      if (v.y < minY) minY = v.y;
      if (v.y > maxY) maxY = v.y;
    }
  } else {
    // No hull yet: use whole pane
    minX = 0;
    minY = 0;
    maxX = paneW;
    maxY = paneH;
  }

  // Clamp to viewport
  minX = max(0, minX);
  minY = max(0, minY);
  maxX = min(paneW, maxX);
  maxY = min(paneH, maxY);

  if (maxX <= minX || maxY <= minY) return;

  // Try a few times to get a point actually inside the hull polygon (if we have hull)
  for (int attempt = 0; attempt < 24; attempt++) {
    float x = random(minX, maxX);
    float y = random(minY, maxY);

    boolean ok = true;
    if (haveHull) {
      ok = pointInPoly(hullProj, x, y);
    }

    if (ok) {
      // Random style per ambient footprint with random color + pattern + size
      FractalStyle style = makeRandomStyle();
      addTrailStep(new PVector(x, y), style);
      return;
    }
  }
}

// Called once per frame to maybe spawn an ambient footprint
void maybeSpawnRandomTrailSteps() {
  if (!enableRandomSpawns || !enableTrails) return;

  // Approximate spawn probability per frame assuming ~60 FPS.
  float p = randomSpawnsPerSecond / 60.0f;
  if (random(1) < p) {
    spawnRandomTrailStepInsideHull();
  }
}

float medianFloat(FloatList L){
  if (L==null || L.size()==0) return Float.NaN;
  FloatList C = new FloatList();
  for (int i=0;i<L.size();i++) C.append(L.get(i));
  C.sort();
  int n = C.size();
  return (n%2==1) ? C.get(n/2) : 0.5f*(C.get(n/2-1)+C.get(n/2));
}

// Only searches the "bottom band" of the bounding box to bias toward feet.

PVector pickFootCandidate(Rectangle r, int[] depth, int[] floor,
                          int hMinMM, int hMaxMM) {
  if (r == null) return null;

  int x0 = max(0, r.x);
  int x1 = min(W - 1, r.x + r.width  - 1);
  int y0 = max(0, r.y);
  int y1 = min(H - 1, r.y + r.height - 1);

  // Only search bottom band of the box
  int band = min(r.height, 60);
  int yStart = y1;
  int yEnd   = max(y0, y1 - band);

  FloatList xs = new FloatList();
  FloatList ys = new FloatList();

  int stepX = 2; // sample every 2 columns

  for (int x = x0; x <= x1; x += stepX) {
    int bestY = -1;
    int bestDepth = -1; // we want the LARGEST depth (closest to floor)

    for (int y = yStart; y >= yEnd; y--) {
      int idx = y * W + x;
      int di = depth[idx];
      int fi = floor[idx];
      if (di <= 0 || fi <= 0) continue;

      int h = fi - di; // height above floor in mm

      // Keep only pixels in the "foot band"
      if (h < hMinMM || h > hMaxMM) continue;

      // Larger depth = closer to floor, so prefer that.
      if (di > bestDepth) {
        bestDepth = di;
        bestY = y;
      }
    }

    if (bestY >= 0) {
      xs.append(x);
      ys.append(bestY);
    }
  }

  if (xs.size() == 0) return null;

  float mx = medianFloat(xs);
  float my = medianFloat(ys);
  return new PVector(mx, my);
}

void updateTrackFoot(Track t, int[] depth, int[] floor) {
  if (t.lastBox == null) return;

  // Look for pixels 20–60 cm above floor (ankles / lower legs)
  int hMin = 200;  // mm above floor
  int hMax = 600;  // mm above floor

  PVector cand = pickFootCandidate(t.lastBox, depth, floor, hMin, hMax);

  final float alpha = 0.70f; // was 0.45f
  final float maxJumpPx = 26.0f; // was 18.0f
  final int   holdFrames = 3; // was 4

  if (cand != null) {
    if (!t.hasFoot) {
      t.foot.set(cand);
      t.hasFoot = true;
      t.footMiss = 0;
    } else {
      float dx = cand.x - t.foot.x;
      float dy = cand.y - t.foot.y;
      float d = sqrt(dx*dx + dy*dy);

      if (d > maxJumpPx && d > 1e-3f) {
        float s = maxJumpPx / d;
        cand.x = t.foot.x + dx * s;
        cand.y = t.foot.y + dy * s;
      }

      t.foot.x = lerp(t.foot.x, cand.x, alpha);
      t.foot.y = lerp(t.foot.y, cand.y, alpha);
      t.footMiss = 0;
    }
  } else {
    if (t.hasFoot) {
      t.footMiss++;
      if (t.footMiss > holdFrames) t.hasFoot = false;
    }
  }
}

// Calibration data
ArrayList<PVector> camPts = new ArrayList<PVector>(); // camera pixels
ArrayList<PVector> projPts = new ArrayList<PVector>(); // projector local pixels
boolean calibrating = false;
boolean previewTargets = false; // show all targets first
int calibIdx = 0;

// GRID targets (fallback)
float[][] projTargetsNorm = new float[][] {
  {0.15f,0.15f},{0.50f,0.15f},{0.85f,0.15f},
  {0.15f,0.50f},{0.50f,0.50f},{0.85f,0.50f},
  {0.15f,0.85f},{0.50f,0.85f},{0.85f,0.85f}
};

// Random targets
enum TargetMode { GRID, RANDOM }
TargetMode targetMode = TargetMode.RANDOM;
ArrayList<PVector> targetsLocal = new ArrayList<PVector>();
int    randCount  = 40;
float  minSep     = 120;
float  edgeMargin = 0.06f;

PVector currentProjTargetLocal() {
  // only return a single target while actively calibrating
  if (!calibrating) return null;
  if (targetMode == TargetMode.GRID) {
    int idx = calibIdx % projTargetsNorm.length;
    return new PVector(projTargetsNorm[idx][0] * paneW,
                       projTargetsNorm[idx][1] * paneH);
  } else {
    if (targetsLocal.isEmpty()) buildRandomTargets();
    int idx = calibIdx % max(1, targetsLocal.size());
    return targetsLocal.get(idx);
  }
}

void buildRandomTargets() {
  targetsLocal.clear();
  float xmin = edgeMargin * paneW;
  float ymin = edgeMargin * paneH;
  float xmax = (1.0f - edgeMargin) * paneW;
  float ymax = (1.0f - edgeMargin) * paneH;

  java.util.Random rng = new java.util.Random(12345 + millis());
  int attempts = 0, maxAttempts = randCount * 60;

  while (targetsLocal.size() < randCount && attempts < maxAttempts) {
    attempts++;
    float x = xmin + rng.nextFloat() * (xmax - xmin);
    float y = ymin + rng.nextFloat() * (ymax - ymin);

    boolean ok = true;
    for (int i = 0; i < targetsLocal.size(); i++) {
      PVector p = targetsLocal.get(i);
      float dx = p.x - x, dy = p.y - y;
      if (dx*dx + dy*dy < minSep*minSep) { ok = false; break; }
    }
    if (ok) targetsLocal.add(new PVector(x, y));
  }
  Collections.shuffle(targetsLocal, new java.util.Random(9876 + millis()));
  println("Random targets: " + targetsLocal.size() + " generated.");
}

// Model
float[] Hcam2proj = null; 
float k1 = 0.0f; // radial distortion
final float uc = W*0.5f, vc = H*0.5f; // depth cam center
final float sNorm = 1.0f / max(W, H); // normalize radius^2

PVector undistort(float u, float v) {
  float du = (u - uc), dv = (v - vc);
  float r2 = (du*du + dv*dv) * (sNorm*sNorm);
  float scale = 1.0f + k1 * r2;
  return new PVector(uc + du*scale, vc + dv*scale);
}

PVector mapCamToProj(float u, float v) {
  if (Hcam2proj == null) return null;
  PVector q = undistort(u, v);
  float x = Hcam2proj[0]*q.x + Hcam2proj[1]*q.y + Hcam2proj[2];
  float y = Hcam2proj[3]*q.x + Hcam2proj[4]*q.y + Hcam2proj[5];
  float w = Hcam2proj[6]*q.x + Hcam2proj[7]*q.y + 1.0f;
  if (abs(w) < 1e-6f) return null;
  return new PVector(x/w, y/w);
}

String homographyPath() { return dataPath("homography.txt"); }

void saveHomography(float[] H) {
  if (H == null) { println("Nothing to save."); return; }
  String[] lines = new String[] {
    nf(H[0],1,6)+" "+nf(H[1],1,6)+" "+nf(H[2],1,6),
    nf(H[3],1,6)+" "+nf(H[4],1,6)+" "+nf(H[5],1,6),
    nf(H[6],1,6)+" "+nf(H[7],1,6)+" "+nf(H[8],1,6),
    "k1 "+nf(k1,1,8)
  };
  saveStrings(homographyPath(), lines);
  println("Saved homography + k1 to " + homographyPath());
}

float[] loadHomography() {
  File f = new File(homographyPath());
  if (!f.exists() || !f.canRead()) { println("No homography file yet."); return null; }
  String[] lines = loadStrings(homographyPath());
  if (lines == null || lines.length < 3) { println("Homography file unreadable."); return null; }
  float[] H = new float[9];
  String[] a = splitTokens(lines[0]), b = splitTokens(lines[1]), c = splitTokens(lines[2]);
  H[0]=parseFloat(a[0]); H[1]=parseFloat(a[1]); H[2]=parseFloat(a[2]);
  H[3]=parseFloat(b[0]); H[4]=parseFloat(b[1]); H[5]=parseFloat(b[2]);
  H[6]=parseFloat(c[0]); H[7]=parseFloat(c[1]); H[8]=parseFloat(c[2]);
  for (int i=3;i<lines.length;i++){
    String[] toks = splitTokens(lines[i]);
    if (toks!=null && toks.length==2 && toks[0].equals("k1")) k1 = parseFloat(toks[1]);
  }
  println("Loaded homography; k1="+k1);
  return H;
}

float[] homographyDLT(List<PVector> srcUndist, List<PVector> dst, int[] useIdx) {
  int N = useIdx.length; if (N < 4) return null;
  double[][] A = new double[2*N][9];
  for (int i=0;i<N;i++){
    PVector s = srcUndist.get(useIdx[i]);
    PVector d = dst.get(useIdx[i]);
    double u=s.x, v=s.y, x=d.x, y=d.y;
    int r=2*i;
    A[r][0]= u;  A[r][1]= v;  A[r][2]= 1;  A[r][3]= 0;  A[r][4]= 0;  A[r][5]= 0;  A[r][6]= -x*u;  A[r][7]= -x*v;  A[r][8]= -x;
    A[r+1][0]= 0; A[r+1][1]= 0; A[r+1][2]= 0; A[r+1][3]= u; A[r+1][4]= v; A[r+1][5]= 1; A[r+1][6]= -y*u; A[r+1][7]= -y*v; A[r+1][8]= -y;
  }
  double[][] AtA = new double[9][9];
  for (int r=0;r<2*N;r++){
    for (int c=0;c<9;c++){
      for (int k=0;k<9;k++){
        AtA[c][k] += A[r][c]*A[r][k];
      }
    }
  }
  double[] h = smallestEigenvector(AtA);
  if (h==null) return null;
  float[] H = new float[9];
  for (int i=0;i<9;i++) H[i] = (float)(h[i] / h[8]);
  return H;
}

double[] smallestEigenvector(double[][] M) {
  int n = M.length;
  double[] y = new double[n];
  java.util.Random rng = new java.util.Random(1234);
  for (int i=0;i<n;i++) y[i] = rng.nextDouble();
  double[][] A = new double[n][n+1];
  double mu = 0.0;
  for (int it=0; it<25; it++) {
    for (int r=0;r<n;r++){
      for(int c=0;c<n;c++) A[r][c]=M[r][c] - (r==c?mu:0);
      A[r][n]=y[r];
    }
    double[] z = gjSolve(A);
    if (z==null) return null;
    double norm=0;
    for(int i=0;i<n;i++) norm+=z[i]*z[i];
    norm=Math.sqrt(norm);
    for(int i=0;i<n;i++) y[i]=z[i]/(norm+1e-12);
  }
  return y;
}

double[] gjSolve(double[][] M){
  int n=M.length, m=M[0].length-1;
  for (int col=0; col<m; col++){
    int piv=col; double best=Math.abs(M[piv][col]);
    for(int r=col+1;r<n;r++){
      double v=Math.abs(M[r][col]);
      if(v>best){best=v;piv=r;}
    }
    if (best<1e-12) return null;
    if (piv!=col){ double[] t=M[piv]; M[piv]=M[col]; M[col]=t; }
    double div=M[col][col];
    for(int j=col;j<=m;j++) M[col][j]/=div;
    for(int r=0;r<n;r++){
      if(r!=col){
        double f=M[r][col];
        if(f!=0){
          for(int j=col;j<=m;j++) M[r][j]-=f*M[col][j];
        }
      }
    }
  }
  double[] x=new double[m];
  for(int i=0;i<m;i++) x[i]=M[i][m];
  return x;
}

float reprojRMS(float[] H, float k1test, int[] idx){
  if (H==null) return 1e9f;
  double ss=0; int n=0;
  for (int i=0;i<idx.length;i++){
    PVector s = camPts.get(idx[i]);
    PVector d = projPts.get(idx[i]);
    float du = (s.x-uc), dv=(s.y-vc);
    float r2 = (du*du+dv*dv)*(sNorm*sNorm);
    float scale = 1.0f + k1test*r2;
    float uu = uc + du*scale, vv = vc + dv*scale;
    float x = H[0]*uu + H[1]*vv + H[2];
    float y = H[3]*uu + H[4]*vv + H[5];
    float w = H[6]*uu + H[7]*vv + 1.0f;
    if (abs(w)<1e-9f) continue;
    float X=x/w, Y=y/w;
    float dx=X-d.x, dy=Y-d.y;
    ss += dx*dx+dy*dy;
    n++;
  }
  return (float)Math.sqrt(ss/Math.max(1,n));
}

class RansacResult { float[] H; int[] inliers; float rms; }

RansacResult ransacH_k1(int iters, float threshPx){
  int N = camPts.size(); if (N<4) return null;
  ArrayList<PVector> und = new ArrayList<PVector>(N);
  for (int i=0;i<N;i++) und.add(undistort(camPts.get(i).x, camPts.get(i).y));
  java.util.Random rng = new java.util.Random(42);
  RansacResult best=null;
  for (int it=0; it<iters; it++){
    HashSet<Integer> pick = new HashSet<Integer>();
    while (pick.size()<4) pick.add(rng.nextInt(N));
    int[] subset = new int[4];
    int t=0;
    for(int v:pick) subset[t++]=v;
    float[] Htry = homographyDLT(und, projPts, subset);
    if (Htry==null) continue;
    ArrayList<Integer> in = new ArrayList<Integer>();
    for (int i=0;i<N;i++){
      PVector su = und.get(i), d = projPts.get(i);
      float x = Htry[0]*su.x + Htry[1]*su.y + Htry[2];
      float y = Htry[3]*su.y + Htry[4]*su.y + Htry[5]; // slight bug from original
      float w = Htry[6]*su.x + Htry[7]*su.y + 1.0f;
      if (abs(w)<1e-9f) continue;
      float X=x/w, Y=y/w;
      float dx=X-d.x, dy=Y-d.y;
      if (dx*dx+dy*dy <= threshPx*threshPx) in.add(i);
    }
    if (in.size()>=4){
      int[] inIdx = new int[in.size()];
      for (int i2=0;i2<in.size();i2++) inIdx[i2]=in.get(i2);
      ArrayList<PVector> undIn = new ArrayList<PVector>(inIdx.length);
      ArrayList<PVector> dstIn = new ArrayList<PVector>(inIdx.length);
      for (int i2=0;i2<inIdx.length;i2++){
        undIn.add(und.get(inIdx[i2]));
        dstIn.add(projPts.get(inIdx[i2]));
      }
      float[] Href = homographyDLT(undIn, dstIn, makeRange(inIdx.length));
      float rms = reprojRMS(Href, k1, inIdx);
      if (best==null || rms<best.rms){
        best=new RansacResult();
        best.H=Href;
        best.inliers=inIdx;
        best.rms=rms;
      }
    }
  }
  return best;
}

int[] makeRange(int n){ int[] r=new int[n]; for (int i=0;i<n;i++) r[i]=i; return r; }

float[] refine_H_k1(float[] Hinit, float k1init, int[] inliers, int iters){
  if (Hinit==null) return null;
  if (Hcam2proj==null) Hcam2proj = new float[9];
  double[] p = new double[9];
  for (int i=0;i<8;i++) p[i]=Hinit[i];
  p[8]=k1init;
  for (int it=0; it<iters; it++){
    double[][] JTJ = new double[9][9];
    double[] JTe = new double[9];
    for (int idx: inliers){
      PVector s = camPts.get(idx);
      PVector d = projPts.get(idx);
      double du = s.x-uc, dv=s.y-vc;
      double r2 = (du*du+dv*dv)*(sNorm*sNorm);
      double scale = 1.0 + p[8]*r2; // k1
      double uu = uc + du*scale;
      double vv = vc + dv*scale;
      double x = p[0]*uu + p[1]*vv + p[2];
      double y = p[3]*uu + p[4]*vv + p[5];
      double w = p[6]*uu + p[7]*vv + 1.0;
      double X = x/w, Y = y/w;
      double ex = X - d.x, ey = Y - d.y;

      double[] jx = new double[]{ uu/w, vv/w, 1.0/w, 0,0,0, -uu*X/w, -vv*X/w };
      double[] jy = new double[]{ 0,0,0, uu/w, vv/w, 1.0/w, -uu*Y/w, -vv*Y/w };

      double du_dk = du * r2;
      double dv_dk = dv * r2;
      double dxdk = p[0]*du_dk + p[1]*dv_dk;
      double dydk = p[3]*du_dk + p[4]*dv_dk;
      double dwdk = p[6]*du_dk + p[7]*dv_dk;
      double dXdk = (dxdk*w - x*dwdk)/(w*w);
      double dYdk = (dydk*w - y*dwdk)/(w*w);

      double[] Jx = new double[9];
      double[] Jy = new double[9];
      Jx[0]=jx[0]; Jx[1]=jx[1]; Jx[2]=jx[2]; Jx[3]=0; Jx[4]=0; Jx[5]=0; Jx[6]=jx[6]; Jx[7]=jx[7]; Jx[8]=dXdk;
      Jy[0]=0; Jy[1]=0; Jy[2]=0; Jy[3]=jy[3]; Jy[4]=jy[4]; Jy[5]=jy[5]; Jy[6]=jy[6]; Jy[7]=jy[7]; Jy[8]=dYdk;

      for (int a=0;a<9;a++){
        for (int b=0;b<9;b++){
          JTJ[a][b] += Jx[a]*Jx[b] + Jy[a]*Jy[b];
        }
        JTe[a] += Jx[a]*ex + Jy[a]*ey;
      }
    }
    
    double[][] Aug = new double[9][10];
    for (int r=0;r<9;r++){
      for (int c=0;c<9;c++) Aug[r][c]=JTJ[r][c];
      Aug[r][9] = -JTe[r];
    }
    double[] dp = gjSolve(Aug);
    if (dp==null) break;

    double maxStep=0.0;
    for (int i=0;i<9;i++){
      p[i]+=dp[i];
      double a = Math.abs(dp[i]);
      if (a > maxStep) maxStep = a;
    }
    if (maxStep<1e-6) break;
  }

  for (int i=0;i<8;i++) Hcam2proj[i]=(float)p[i];
  Hcam2proj[8]=1;
  k1 = (float)p[8];
  return Hcam2proj;
}

float[] normalizeH(float[] H){
  if (H==null) return null;
  float s = H[8];
  if (Math.abs(s)<1e-9f) s=1;
  float[] R = new float[9];
  for (int i=0;i<9;i++) R[i]=H[i]/s;
  return R;
}

// Hull binding

float ACCEPT_THRESH = 24.0f;
float KEEP_FRACTION = 0.80f;

int[] lastKept = new int[0]; // indices used to build hull

ArrayList<PVector> hullCam = new ArrayList<PVector>(); // Undistorted camera pixels
ArrayList<PVector> hullProj = new ArrayList<PVector>();  // mapped into projector
float hullShrink = 0.97f;

boolean useHullGate = true;

float cross(PVector O, PVector A, PVector B){
  return (A.x - O.x)*(B.y - O.y) - (A.y - O.y)*(B.x - O.x);
}

ArrayList<PVector> convexHull(List<PVector> pts){
  ArrayList<PVector> p = new ArrayList<PVector>(pts);
  Collections.sort(p, new Comparator<PVector>() {
    public int compare(PVector a, PVector b) {
      int cmp = Float.compare(a.x, b.x);
      if (cmp != 0) return cmp;
      return Float.compare(a.y, b.y);
    }
  });
  ArrayList<PVector> Hh = new ArrayList<PVector>();
  for (PVector pt : p){
    while (Hh.size() >= 2 && cross(Hh.get(Hh.size()-2), Hh.get(Hh.size()-1), pt) <= 0) Hh.remove(Hh.size()-1);
    Hh.add(pt);
  }
  int t = Hh.size()+1;
  for (int i=p.size()-2; i>=0; i--){
    PVector pt = p.get(i);
    while (Hh.size() >= t && cross(Hh.get(Hh.size()-2), Hh.get(Hh.size()-1), pt) <= 0) Hh.remove(Hh.size()-1);
    Hh.add(pt);
  }
  if (Hh.size()>1) Hh.remove(Hh.size()-1);
  return Hh;
}

ArrayList<PVector> shrinkPoly(List<PVector> poly, float s){
  if (poly==null || poly.size()==0) return new ArrayList<PVector>();
  PVector c = new PVector();
  for (PVector v: poly) { c.x += v.x; c.y += v.y; }
  c.x /= poly.size(); c.y /= poly.size();
  ArrayList<PVector> out = new ArrayList<PVector>(poly.size());
  for (PVector v: poly){
    out.add(new PVector(c.x + (v.x - c.x)*s, c.y + (v.y - c.y)*s));
  }
  return out;
}

boolean pointInPoly(List<PVector> poly, float x, float y){
  boolean inside = false;
  for (int i=0, j=poly.size()-1; i<poly.size(); j=i++){
    float xi=poly.get(i).x, yi=poly.get(i).y;
    float xj=poly.get(j).x, yj=poly.get(j).y;
    boolean intersect = ((yi>y)!=(yj>y)) && (x < (xj - xi)*(y - yi)/(yj - yi + 1e-9f) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

void buildOverlapHullFromIdx(int[] idx){
  hullCam.clear();
  hullProj.clear();
  if (idx==null || idx.length<3){
    println("Not enough kept points to build overlap hull.");
    return;
  }
  ArrayList<PVector> undIn = new ArrayList<PVector>();
  for (int k : idx){
    PVector s = camPts.get(k);
    undIn.add(undistort(s.x, s.y));
  }
  if (undIn.size()<3){
    println("Not enough undistorted points for hull.");
    return;
  }
  ArrayList<PVector> hull = convexHull(undIn);
  hull = shrinkPoly(hull, hullShrink);
  hullCam.addAll(hull);
  for (PVector v: hullCam){
    PVector q = mapCamToProj(v.x, v.y);
    if (q != null) hullProj.add(q);
  }
  println("Overlap hull: " + hullCam.size() + " vertices  (kept="+idx.length+")");
}

// Warp disabled, we just pass projector coords through
float projWarpStrength = 0.0f;

PVector applyProjectorRadialCorrection(PVector p) {
  if (p == null) return null;
  return p.copy();
}

// Fit
float doFit() {
  int N = camPts.size();
  if (N < 4) { println("Need at least 4 points."); return -1; }

  RansacResult rr = ransacH_k1(600, 10.0f);
  if (rr == null) { println("RANSAC failed."); return -1; }

  if (Hcam2proj==null) Hcam2proj = new float[9];
  refine_H_k1(rr.H, k1, rr.inliers, 25);
  Hcam2proj = normalizeH(Hcam2proj);

  class Resid { int idx; float err; }
  ArrayList<Resid> res = new ArrayList<Resid>(camPts.size());
  for (int i=0; i<camPts.size(); i++){
    PVector s = camPts.get(i), d = projPts.get(i);
    PVector m = mapCamToProj(s.x, s.y);
    if (m==null) continue;
    float dx = m.x - d.x, dy = m.y - d.y;
    Resid r = new Resid();
    r.idx=i;
    r.err=sqrt(dx*dx + dy*dy);
    res.add(r);
  }
  if (res.size()<3){ println("Too few valid residuals."); return -1; }

  Collections.sort(res, new Comparator<Resid>(){
    public int compare(Resid a, Resid b){ return Float.compare(a.err, b.err); }
  });

  int keepN = constrain(round(KEEP_FRACTION * res.size()), 3, res.size());
  IntList keep = new IntList();
  IntList drop = new IntList();
  for (int i=0;i<res.size();i++){
    Resid r = res.get(i);
    if (i < keepN && r.err <= ACCEPT_THRESH) keep.append(r.idx);
    else drop.append(r.idx);
  }
  if (keep.size() < 3 && res.size() >= 3) {
    keep.clear();
    for (int i=0;i<min(3,res.size());i++) keep.append(res.get(i).idx);
  }
  lastKept = keep.array();

  float rms = reprojRMS(Hcam2proj, k1, rr.inliers);
  println("RANSAC inliers: "+rr.inliers.length+"/"+N+
          "  refine RMS="+nf(rms,1,2)+" px  k1="+nf(k1,1,8));
  println("Hull keep: "+keep.size()+"/"+res.size()+
          "  (KEEP="+int(KEEP_FRACTION*100)+"%, ACCEPT="+nf(ACCEPT_THRESH,1,1)+"px)");

  buildOverlapHullFromIdx(lastKept);
  pruneTrailSteps(); // clean out any old footprints outside new hull / too old
  return rms;
}

int projectorDisplayIndex = 2; // set to your projector monitor index
ProjectorWindow projWin = null;
boolean projHideCursor = false;
boolean testPattern = false; // 'k' toggle

// Sound
SoundFile bgm;// background music

class ProjectorWindow extends PApplet {
  volatile PVector calibTarget = null;
  volatile ArrayList<PVector> dots = new ArrayList<PVector>();

  // preview mirroring
  volatile boolean previewDotsMode = false;
  volatile ArrayList<PVector> previewDots = new ArrayList<PVector>();

  final float TAU = TWO_PI;

  // Palettes used by all footprints; actual choice is per TrailStep.style.palIdx
  color[][] PALETTES = new color[][]{
    {#7C4DFF,#2DA7FF,#6CFFF0,#FFEED6}, // aurora
    {#90A6FF, #CBE4FF, #EAF9FF, #6A3DFF}, // ice, non-white accent
    {#C55BFF,#FF74E6,#FFD6F7,#FFF2FF}, // magenta
    {#FFB84D,#FFD36E,#FFE7B2,#FFF7E6}, // gold
    {#1E2EFF,#38D6FF,#7CFFF6,#F7FFF9}  // cyan
  };

  public void settings() {
    // Fullscreen on the projector monitor (no window bar, no borders)
    fullScreen(JAVA2D, projectorDisplayIndex);
  }



public void setup() {
  surface.setTitle("PROJECTOR OUT (JAVA2D)");
  // In fullScreen mode
  println("PROJECTOR PIXELS = " + width + " x " + height);
  cursor(ARROW);
  frameRate(50); // slightly capped to help performance
}


  // draw an N-gon centered at 0,0 with radius r
  void regularPolygon(int n, float r){
    beginShape();
    for (int i=0;i<n;i++){
      float a = -HALF_PI + i*TAU/n;
      vertex(cos(a)*r, sin(a)*r);
    }
    endShape(CLOSE);
  }

  // connect opposite vertices
  void starLines(int n, float r){
    for (int i=0;i<n;i++){
      float a1 = -HALF_PI + i*TAU/n;
      float a2 = -HALF_PI + ((i+4)%n)*TAU/n;
      line(cos(a1)*r, sin(a1)*r, cos(a2)*r, sin(a2)*r); 
    }
  }

  void drawOctagonPretty(float r, int depth, float animAngle, float fade, FractalStyle style){
    if (style == null) return;

    int idx = constrain(style.palIdx, 0, PALETTES.length - 1);
    color P1 = PALETTES[idx][0];
    color P2 = PALETTES[idx][1];
    color P3 = PALETTES[idx][2];
    color P4 = PALETTES[idx][3];

    int maxD = max(1, style.maxDepth);

    pushMatrix();
    rotate(animAngle);

    float t = map(depth, 0, maxD, 0, 1);
    int base = lerpColor(P1, P2, t*0.7f);
    int edge = lerpColor(P3, P4, 0.4f + 0.6f*t);

    int aFull = (int)(255 * fade);
    aFull = constrain(aFull, 0, 255);

    // Fill
    noStroke();
    fill(red(base), green(base), blue(base), aFull);
    regularPolygon(8, r);

    // Outer outline
    int aEdge = (int)(220 * fade);
    stroke(red(edge), green(edge), blue(edge), constrain(aEdge, 0, 255));
    strokeWeight(max(1.4f, r*0.08f));
    noFill();
    regularPolygon(8, r);

    // Inner star
    int starCol = lerpColor(base, #6A3DFF, 0.6f);
    int aStar = (int)(190 * fade);
    stroke(red(starCol), green(starCol), blue(starCol), constrain(aStar, 0, 255));
    strokeWeight(max(1.0f, r*0.05f));
    starLines(8, r*0.78f);

    popMatrix();
  }

  // Different pattern layouts

  // Classic, root and single ring of children
  void drawPatternSingleRing(float rBase, float animAngle, float fade, FractalStyle style){
    int maxD = max(1, style.maxDepth);
    float scaleChild = style.scaleChild;
    float gapFactor  = style.gapFactor;
    float twist      = style.twist;

    // root
    drawOctagonPretty(rBase, 0, animAngle, fade, style);

    // one child ring (depth 1)
    float childR = rBase * scaleChild;
    float dist   = (rBase + childR) * (1.0f + gapFactor);
    for (int i = 0; i < 8; i++){
      float a = -HALF_PI + i*TAU/8.0;
      float nx = cos(a) * dist;
      float ny = sin(a) * dist;
      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(childR, 1, animAngle + twist, fade, style);
      popMatrix();
    }
  }

  // Two rings, tighter inner ring and looser outer ring
  void drawPatternDoubleRing(float rBase, float animAngle, float fade, FractalStyle style){
    int maxD = max(2, style.maxDepth);
    float scaleChild = style.scaleChild;
    float gapFactor  = style.gapFactor;
    float twist      = style.twist;

    drawOctagonPretty(rBase, 0, animAngle, fade, style);

    float childR1 = rBase * scaleChild;
    float dist1   = (rBase + childR1) * (1.0f + gapFactor * 0.5f);

    // small inner ring with 6 children
    int innerN = 6;
    for (int i = 0; i < innerN; i++){
      float a = -HALF_PI + i*TAU/innerN;
      float nx = cos(a) * dist1;
      float ny = sin(a) * dist1;
      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(childR1, 1, animAngle + twist * 0.5f, fade, style);
      popMatrix();
    }

    // outer ring with 8 slightly smaller children
    float childR2 = childR1 * 0.75f;
    float dist2   = (rBase + childR2) * (1.0f + gapFactor * 1.3f);
    for (int i = 0; i < 8; i++){
      float a = -HALF_PI + i*TAU/8.0 + 0.15f;
      float nx = cos(a) * dist2;
      float ny = sin(a) * dist2;
      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(childR2, 2, animAngle + twist, fade, style);
      popMatrix();
    }
  }

  // Spiral arc, small steps along a spiral path outward
  void drawPatternSpiralArc(float rBase, float animAngle, float fade, FractalStyle style){
    int maxD = max(2, style.maxDepth);
    float scaleChild = style.scaleChild;
    float twist      = style.twist;

    drawOctagonPretty(rBase * 0.9f, 0, animAngle, fade, style);

    int steps = 12 + maxD * 3;
    float angleStep = TAU / (float)steps * 1.5f;
    float radiusStep = rBase * 0.16f;

    float childR = rBase * scaleChild * 0.9f;
    float angle = animAngle;
    float radius = rBase * 0.6f;

    for (int i = 0; i < steps; i++) {
      float nx = cos(angle) * radius;
      float ny = sin(angle) * radius;
      int depth = 1 + (i % maxD);

      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(childR, depth, animAngle + twist * depth, fade, style);
      popMatrix();

      angle  += angleStep;
      radius += radiusStep;
      childR *= 0.97f; // slightly shrink along spiral
    }
  }

  // Flower cluster, several overlapping "petals" around center
  void drawPatternClusterFlower(float rBase, float animAngle, float fade, FractalStyle style){
    int petals = 5;
    float petalRadius = rBase * 0.9f;
    float ringRadius  = rBase * 0.7f;

    // soft center
    drawOctagonPretty(rBase * 0.8f, 0, animAngle, fade, style);

    for (int i = 0; i < petals; i++) {
      float a = -HALF_PI + i*TAU/petals;
      float nx = cos(a) * ringRadius;
      float ny = sin(a) * ringRadius;

      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(petalRadius, 1, animAngle + style.twist * i, fade, style);
      popMatrix();
    }

    // tiny inner ring of sparks
    float sparkR = petalRadius * 0.4f;
    float sparkDist = rBase * 0.35f;
    for (int i = 0; i < petals; i++) {
      float a = -HALF_PI + i*TAU/petals + 0.35f;
      float nx = cos(a) * sparkDist;
      float ny = sin(a) * sparkDist;
      pushMatrix();
      translate(nx, ny);
      drawOctagonPretty(sparkR, 2, animAngle - style.twist * 0.7f * i, fade, style);
      popMatrix();
    }
  }

  // Main footprint renderer which chooses pattern based on style.pattern
  void renderFractalFoot(float x, float y, float baseR, float timeSeconds, float fade, FractalStyle style){
    if (style == null) return;

    pushMatrix();
    translate(x, y);

    float animAngle = 0;
    float rBase = baseR * style.sizeMul;
    if (style.animate) {
      float rotRad = radians(style.rotSpeedDeg);
      animAngle = rotRad * timeSeconds;
      if (style.breatheAmp > 0) {
        float breath = 1.0f + style.breatheAmp * sin(TAU * style.breatheHz * timeSeconds);
        rBase *= breath;
      }
    }

    if (style.pattern == null) {
      // fallback is a single ring
      drawPatternSingleRing(rBase, animAngle, fade, style);
    } else {
      switch (style.pattern) {
      case SINGLE_RING:
        drawPatternSingleRing(rBase, animAngle, fade, style);
        break;
      case DOUBLE_RING:
        drawPatternDoubleRing(rBase, animAngle, fade, style);
        break;
      case SPIRAL_ARC:
        drawPatternSpiralArc(rBase, animAngle, fade, style);
        break;
      case CLUSTER_FLOWER:
        drawPatternClusterFlower(rBase, animAngle, fade, style);
        break;
      default:
        drawPatternSingleRing(rBase, animAngle, fade, style);
        break;
      }
    }

    popMatrix();
  }

  public void draw() {
    if (projHideCursor) noCursor(); else cursor(ARROW);
    background(0);
  
    // soft rounded border around the projector output
    float margin = 24;
    float radius = 40;
    noFill();
    stroke(180, 210, 255); // soft bluish border
    strokeWeight(3);
    rect(margin, margin,
         width  - 2*margin,
         height - 2*margin,
         radius);
  
    float sx = (float)width / paneW;
    float sy = (float)height / paneH;

    if (testPattern) {
      for (int x=0; x<width; x+=10) { stroke((x%100==0)?200:80); line(x, 0, x, height); }
      for (int y=0; y<height; y+=10) { stroke((y%100==0)?200:80); line(0, y, width, y); }
      for (int yy=50; yy<150; yy++) for (int xx=50; xx<150; xx++) {
        stroke(((xx+yy)&1)==0 ? 255 : 0); point(xx, yy);
      }
    }

    float t = millis() / 1000.0f;

    // Preview mode
    if (previewDotsMode) {
      for (int i=0;i<previewDots.size();i++){
        PVector p = previewDots.get(i);
        float tx = p.x * sx;
        float ty = p.y * sy;
        noFill(); stroke(255,60,60); strokeWeight(2);
        ellipse(tx, ty, 28, 28);
        noFill(); stroke(255,60,60); strokeWeight(1.5f);
        ellipse(tx, ty, 60, 60);
      }
    } else {
      // draw single target
      if (calibTarget != null) {
        float tx = calibTarget.x * sx, ty = calibTarget.y * sy;
        noStroke(); fill(255,60,60); ellipse(tx, ty, 28, 28);
        noFill(); stroke(255,60,60); strokeWeight(3); ellipse(tx, ty, 70, 70);
      }
    }

if (enableTrails) {
  float nowProj = millis() / 1000.0f;
  clip(margin, margin,
       width  - 2*margin,
       height - 2*margin);

  synchronized (trailSteps) {
    for (int i = 0; i < trailSteps.size(); i++) {
      TrailStep ts = trailSteps.get(i);
      if (ts.style == null) continue;

      float age = nowProj - ts.createdT;
      float fade = 1.0f - constrain(age / TRAIL_LIFETIME, 0, 1);
      if (fade <= 0) continue;

      PVector p = ts.pos;
      float tx = p.x * sx;
      float ty = p.y * sy;

      float baseR = (ts.radius > 0) ? ts.radius : trailBaseRadius;

      renderFractalFoot(tx, ty, baseR, t, fade, ts.style);
    }
  }

  noClip();
}
    synchronized (dots) {
      noStroke(); fill(255);
      for (int i=0;i<dots.size();i++){
        PVector p = dots.get(i);
        ellipse(p.x*sx, p.y*sy, 18, 18);
      }
    }
  }
}

Rectangle getScreenBounds(int idx){
  try{
    GraphicsDevice[] devs = GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
    if (idx<0 || idx>=devs.length) {
      println("Monitor index "+idx+" invalid; found "+devs.length);
      return null;
    }
    return devs[idx].getDefaultConfiguration().getBounds();
  }catch(Exception e){
    e.printStackTrace();
    return null;
  }
}

void openProjectorWindow(){
  if (projWin!=null) return;
  try{
    projWin = new ProjectorWindow();
    PApplet.runSketch(new String[]{"ProjectorOut"}, projWin);
    println("Projector window opened on monitor "+projectorDisplayIndex+" (JAVA2D).");
  }catch(Exception e){
    e.printStackTrace();
    projWin=null;
  }
}

void closeProjectorWindow(){
  try{
    if (projWin!=null){
      projWin.noLoop();
      projWin.dispose();
      projWin=null;
      println("Projector window closed.");
    }
  }catch(Exception e){
    e.printStackTrace();
    projWin=null;
  }
}

// UI Layout 
void settings(){
  paneW = W * scaleDisp;
  paneH = H * scaleDisp;
  rightX = paneW + gap;
  size(int(paneW*2 + gap), int(paneH + 120), JAVA2D);
}

void setup(){
  surface.setTitle("Kinect and Projector)");
  openProjectorWindow();
  cursor(ARROW);
  kinect = new KinectPV2(this);
  kinect.enableDepthImg(true);
  kinect.init();

  // downscaled mask
  maskSmall = createImage(DS_W, DS_H, ALPHA);
  opencvSmall = new OpenCV(this, DS_W, DS_H);

  Hcam2proj = loadHomography();
  if (targetMode == TargetMode.RANDOM) buildRandomTargets();

  // Sound setup
  try {
    bgm = new SoundFile(this, "dark-cluster-16449.mp3");
    bgm.loop();
    bgm.amp(0.7);   // adjust overall volume (0.0–1.0)
    println("Background music loaded and looping.");
  } catch (Exception e) {
    println("Could not load/loop dark-cluster-16449.mp3");
    e.printStackTrace();
  }
}

// timing diagnostics
long lastTime = 0;
float avgDetectMs = 0;
float detectAlpha = 0.05f;

// half-rate morphology toggle
int morphCounter = 0;

void draw(){
  long frameStart = System.nanoTime();

  if (calibrating) {
    cursor(CROSS);
  } else {
    cursor(ARROW);
  }

  background(20);
  int[] depth = kinect.getRawDepthData();
  if (depth==null || depth.length!=W*H) return;

  if (floorMM != null){
    boolean doProcess = (frameCount % PROCESS_EVERY == 0);

    long detectStart = System.nanoTime();

    ArrayList<Rectangle> boxes;
    ArrayList<PVector>   centers;

    if (doProcess) {
      boxes   = boxesReuse;
      centers = centersReuse;
      boxes.clear();
      centers.clear();
      
      maskSmall.loadPixels();
      int fgCol = color(255);
      int bgCol = color(0);
      Arrays.fill(maskSmall.pixels, bgCol);

      int whiteCount = 0; // how many foreground pixels we set

      for (int y = 0; y < H; y++) {
        int sy = y / DS;
        if (sy < 0 || sy >= DS_H) continue;
        int rowFull  = y * W;
        int rowSmall = sy * DS_W;

        for (int x = 0; x < W; x++) {
          int sx = x / DS;
          if (sx < 0 || sx >= DS_W) continue;
          int idx = rowFull + x;
          int d = depth[idx];
          int f = floorMM[idx];
          if (d > 0 && f > 0) {
            int h = f - d;
            if (h >= H_MIN && h <= H_MAX) {
              int smallIdx = rowSmall + sx;
              if (maskSmall.pixels[smallIdx] != fgCol) {
                maskSmall.pixels[smallIdx] = fgCol;
                whiteCount++;
              }
            }
          }
        }
      }
      maskSmall.updatePixels();

      float totalSmall = (float)(DS_W * DS_H);
      float occupancy  = (totalSmall > 0) ? (whiteCount / totalSmall) : 0;

      if (whiteCount == 0) {
        // no foreground,  clear detections and cache
        boxes.clear();
        centers.clear();
        lastDetBoxes.clear();
        lastDetCenters.clear();
      }
      else if (occupancy > 0.30f && !lastDetBoxes.isEmpty()) {
        // mask is very full because likely many contours so skip findContours and reuse last good detections for this frame
        boxes.clear();
        centers.clear();
        boxes.addAll(lastDetBoxes);
        centers.addAll(lastDetCenters);
      }
      else {
        // run OpenCV on relatively sparse mask
        opencvSmall.loadImage(maskSmall);

        morphCounter++;
        if ( (morphCounter & 1) == 0 ) {
          opencvSmall.dilate();
          opencvSmall.erode();
        }

        ArrayList<Contour> contours = opencvSmall.findContours(true, true);

        int limit = min(contours.size(), MAX_CONTOURS);
        for (int i=0;i<limit;i++){
          Rectangle rs = contours.get(i).getBoundingBox();
          Rectangle r  = new Rectangle(rs.x * DS, rs.y * DS,
                                       rs.width * DS, rs.height * DS);
          int area = r.width * r.height;
          if (area<MIN_AREA || area>MAX_AREA) continue;
          boxes.add(r);
          centers.add(new PVector(r.x + r.width*0.5f, r.y + r.height*0.5f));
        }

        lastDetBoxes.clear();
        lastDetBoxes.addAll(boxes);
        lastDetCenters.clear();
        lastDetCenters.addAll(centers);
      }

      long detectEnd = System.nanoTime();
      float detectMs = (detectEnd - detectStart) / 1e6f;
      if (avgDetectMs == 0) avgDetectMs = detectMs;
      else avgDetectMs = lerp(avgDetectMs, detectMs, detectAlpha);

      // Only print truly large spikes to avoid spam and print overhead
      if (detectMs > 30.0f){
        println("DETECT SPIKE: " + nf(detectMs,1,2) + " ms (avg=" + nf(avgDetectMs,1,2) + " ms)");
      }

    } else {
      // reuse last detections to keep CPU down this frame
      boxes   = lastDetBoxes;
      centers = lastDetCenters;
    }

    updateTracks(centers, boxes, 80);
    
    if (frameCount % DEPTH_IMG_EVERY == 0 || lastDepthImg == null) {
      lastDepthImg = kinect.getDepthImage();
    }
    if (lastDepthImg != null) {
      image(lastDepthImg, leftX, leftY, paneW, paneH);
    }

    pushMatrix();
      translate(leftX,leftY);
      scale(scaleDisp);

      if (debugView){
        noFill(); stroke(0,255,0); strokeWeight(2.0f/scaleDisp);
        for (Rectangle r : boxes) rect(r.x,r.y,r.width,r.height);
      }

      if (hullCam.size() >= 3){
        noFill(); stroke(60,255,120); strokeWeight(2.0f/scaleDisp);
        beginShape();
        for (PVector v: hullCam) vertex(v.x, v.y);
        endShape(CLOSE);
      }

      for (Track t: tracks) if (t.confirmed){
        updateTrackFoot(t, depth, floorMM);
        if (!t.hasFoot) continue;

        boolean inside = true;
        if (useHullGate && hullCam.size()>=3){
          PVector su = undistort(t.foot.x, t.foot.y);
          inside = pointInPoly(hullCam, su.x, su.y);
        }

        if (!inside){
          noFill(); stroke(255,80,80); strokeWeight(2.0f/scaleDisp);
          line(t.pos.x-10, t.pos.y-10, t.pos.x+10, t.pos.y+10);
          line(t.pos.x+10, t.pos.y-10, t.pos.x-10, t.pos.y+10);
        } else {
          fill(255); noStroke();
          ellipse(t.foot.x, t.foot.y, 10.0f/scaleDisp, 10.0f/scaleDisp);
          fill(255,240,0); textSize(14.0f/scaleDisp);
          text("ID "+t.id, t.foot.x + 8.0f/scaleDisp, t.foot.y - 8.0f/scaleDisp);
        }

        // DEBUG
        if (Hcam2proj != null) {
          PVector Pdbg = mapCamToProj(t.foot.x, t.foot.y);
          if (Pdbg != null) {
            float px = map(Pdbg.x, 0, paneW, 20, 120);
            float py = map(Pdbg.y, 0, paneH, 20, 120);
            noStroke();
            fill(0, 255, 255);
            ellipse(px, py, 6.0f/scaleDisp, 6.0f/scaleDisp);
          }
        }
      }
    popMatrix();

    noStroke(); fill(0); rect(rightX,rightY,paneW,paneH);

    if (previewTargets) {
      pushStyle();
      noFill(); stroke(255,60,60); strokeWeight(2);

      // reuse preview list
      ArrayList<PVector> ptsForPreview = ptsForPreviewReuse;
      ptsForPreview.clear();

      if (targetMode==TargetMode.GRID){
        for (int i=0;i<projTargetsNorm.length;i++){
          ptsForPreview.add(new PVector(
            projTargetsNorm[i][0]*paneW,
            projTargetsNorm[i][1]*paneH
          ));
        }
      } else {
        if (targetsLocal.isEmpty()) buildRandomTargets();
        ptsForPreview.addAll(targetsLocal);
      }

      for (PVector p : ptsForPreview){
        float tx = rightX + p.x;
        float ty = rightY + p.y;
        noFill(); stroke(255,60,60); strokeWeight(2);
        ellipse(tx,ty,28,28);
        noFill(); stroke(255,60,60); strokeWeight(1.5f);
        ellipse(tx,ty,60,60);
      }
      popStyle();

      // also mirror to projector window
      if (projWin != null){
        projWin.previewDotsMode = true;
        synchronized (projWin.previewDots){
          projWin.previewDots.clear();
          if (targetMode==TargetMode.GRID){
            for (int i=0;i<projTargetsNorm.length;i++){
              projWin.previewDots.add(new PVector(
                projTargetsNorm[i][0]*paneW,
                projTargetsNorm[i][1]*paneH
              ));
            }
          } else {
            if (targetsLocal.isEmpty()) buildRandomTargets();
            projWin.previewDots.addAll(targetsLocal);
          }
        }
      }
    } else {
      // Single Red target
      PVector tgtLocal = calibrating ? currentProjTargetLocal() : null;
      if (tgtLocal!=null){
        float tx=rightX+tgtLocal.x, ty=rightY+tgtLocal.y;
        noStroke(); fill(255,60,60); ellipse(tx,ty,28,28);
        noFill(); stroke(255,60,60); strokeWeight(3); ellipse(tx, ty, 70, 70);
      }

      // Mirror single target to projector window
      if (projWin != null){
        projWin.previewDotsMode = false;
        projWin.calibTarget = tgtLocal;
      }
    }

    // draw overlap hull on projector pane (green polygon)
    if (hullProj.size() >= 3){
      noFill(); stroke(60,255,120); strokeWeight(2);
      beginShape();
      for (PVector v: hullProj) vertex(rightX+v.x, rightY+v.y);
      endShape(CLOSE);
    }

    pruneTrailSteps();

    // maybe spawn ambient random footprints (independent of trails)
    maybeSpawnRandomTrailSteps();

    // draw calibration residual info (yellow lines)
    if (Hcam2proj != null && camPts.size()>0){
      // kept (green)
      stroke(60,255,120); fill(60,255,120);
      for (int idx : lastKept){
        PVector s = camPts.get(idx);
        PVector m = mapCamToProj(s.x, s.y);
        if (m!=null) ellipse(rightX+m.x, rightY+m.y, 7, 7);
      }
      // all residual lines (yellow)
      stroke(255,220,60); strokeWeight(2);
      for (int i=0;i<camPts.size();i++){
        PVector s = camPts.get(i), d = projPts.get(i);
        PVector m = mapCamToProj(s.x, s.y);
        if (m!=null){
          line(rightX+m.x, rightY+m.y, rightX+d.x, rightY+d.y);
        }
      }
      // dropped (red X)
      HashSet<Integer> keptSet = new HashSet<Integer>();
      for (int k : lastKept) keptSet.add(k);
      stroke(255,80,80); noFill();
      for (int i=0;i<camPts.size();i++){
        if (keptSet.contains(i)) continue;
        PVector s = camPts.get(i);
        PVector m = mapCamToProj(s.x, s.y);
        if (m!=null){
          float x = rightX+m.x, y = rightY+m.y;
          line(x-6,y-6,x+6,y+6);
          line(x+6,y-6,x-6,y+6);
        }
      }
    }
    
    ArrayList<PVector> dotsLocal = dotsLocalReuse;
    dotsLocal.clear();

    if (Hcam2proj != null){
      for (Track t : tracks) if (t.confirmed && t.hasFoot) {
        // 1) gate in camera space
        boolean okCam = true;
        if (useHullGate && hullCam.size() >= 3) {
          PVector su = undistort(t.foot.x, t.foot.y);
          okCam = pointInPoly(hullCam, su.x, su.y);
        }
        if (!okCam) continue;

        // map to projector space
        PVector P = mapCamToProj(t.foot.x, t.foot.y);
        if (P!=null && P.x >= 0 && P.x < paneW && P.y >= 0 && P.y < paneH){

          // snappier projector smoothing
          final float projAlpha   = 0.82f;  // was 0.65f
          final float projMaxJump = 90.0f;  // was 60.0f

          if (!t.hasProj) {
            t.projPos.set(P);
            t.hasProj = true;
          } else {
            float dx = P.x - t.projPos.x;
            float dy = P.y - t.projPos.y;
            float d  = sqrt(dx*dx + dy*dy);
            if (d > projMaxJump && d > 1e-3f) {
              float s = projMaxJump / d;
              P.x = t.projPos.x + dx * s;
              P.y = t.projPos.y + dy * s;
            }

            t.projPos.x = lerp(t.projPos.x, P.x, projAlpha);
            t.projPos.y = lerp(t.projPos.y, P.y, projAlpha);
          }

          PVector warped = applyProjectorRadialCorrection(t.projPos);

          // extra safety gate in projector space and viewport bounds
          boolean insideProj = true;
          if (useHullGate && hullProj.size() >= 3) {
            insideProj = pointInPoly(hullProj, warped.x, warped.y);
          }
          boolean insideViewport =
            (warped.x >= 0 && warped.x < paneW &&
             warped.y >= 0 && warped.y < paneH);

          if (!(insideProj && insideViewport)) {
            continue;
          }

          // add to live dots for preview pane
          dotsLocal.add(warped.copy());

          // maybe drop a fractal footprint for this track (with random spacing)
          maybeAddTrailStep(t, warped);

          // debug marker on the RIGHT pane
          noStroke(); fill(255);
          ellipse(rightX + warped.x, rightY + warped.y, 10, 10);
        }
      }
    }

    // mirror tracking dots to projector window
    if (projWin != null){
      synchronized (projWin.dots){
        projWin.dots.clear();
        projWin.dots.addAll(dotsLocal);
      }
    }

    // HUD text
    fill(255);

    String modeText;
    if (previewTargets) {
      modeText = "PREVIEW MODE: check spread. Press ENTER to start mapping.";
    } else if (calibrating) {
      modeText = "CALIB MODE: click Kinect LEFT pane for each red target.";
    } else {
      modeText = "IDLE (press 'v' to preview targets)";
    }

    String hud =
      "Tracks: " + countConfirmed() +
      "   ['b' floor, 'o' dbg, 'v' preview, ENTER confirm, 'c' calib, 'd' discard, 'u' undo, 'r' fit+hull, 'n' new, 'l' load, 's' save, 'p' proj, 't' mode, 'G' regen, '['/']' ACCEPT, ','/'.' KEEP, 'h' shrink, 'k' test, 'g' hullGate, 'f' trails]" +
      "  k1=" + nf(k1,1,6) +
      "  hullShrink=" + nf(hullShrink,1,2) +
      "  ACCEPT=" + nf(ACCEPT_THRESH,1,1) + "px" +
      "  KEEP=" + int(KEEP_FRACTION*100) + "%" +
      "  hullGate=" + (useHullGate ? "ON" : "OFF") +
      "  trails=" + (enableTrails ? "ON" : "OFF") +
      "  processEvery=" + PROCESS_EVERY +
      "  avgDetectMs=" + nf(avgDetectMs,1,2);

    text(modeText, 20, paneH+20);
    text(hud, 20, paneH+50);

  } else {
    if (frameCount % DEPTH_IMG_EVERY == 0 || lastDepthImg == null) {
      lastDepthImg = kinect.getDepthImage();
    }
    if (lastDepthImg != null) {
      image(lastDepthImg, leftX, leftY, paneW, paneH);
    }

    noFill(); stroke(80); rect(rightX,rightY,paneW,paneH);

    // projector window state even before floor capture
    if (projWin != null){
      projWin.previewDotsMode = previewTargets;
      if (previewTargets){
        synchronized (projWin.previewDots){
          projWin.previewDots.clear();
          if (targetMode==TargetMode.GRID){
            for (int i=0;i<projTargetsNorm.length;i++){
              projWin.previewDots.add(new PVector(
                projTargetsNorm[i][0]*paneW,
                projTargetsNorm[i][1]*paneH
              ));
            }
          } else {
            if (targetsLocal.isEmpty()) buildRandomTargets();
            projWin.previewDots.addAll(targetsLocal);
          }
        }
      }
    }

    fill(255);
    text("Press 'b' when the floor is empty to capture the background.", 20, paneH+50);
  }

  long frameEnd = System.nanoTime();
  float frameMs = (frameEnd - frameStart)/1e6f;
  float fpsEst = (frameMs > 0) ? (1000.0f / frameMs) : 0;

  if (frameCount % 60 == 0){
    println("FRAME: " + nf(frameMs,1,2) + " ms   DETECT: " + nf(avgDetectMs,1,2) + " ms   FPS?"+ nf(fpsEst,1,1));
  }
}

// Tracking Helpwers

float iou(Rectangle a, Rectangle b){
  int x1=max(a.x,b.x), y1=max(a.y,b.y);
  int x2=min(a.x+a.width,b.x+b.width), y2=min(a.y+a.height,b.y+b.height);
  int iw=max(0,x2-x1), ih=max(0,y2-y1);
  int inter=iw*ih, uni=a.width*a.height + b.width*b.height - inter;
  if (uni<=0) return 0;
  return (float)inter/(float)uni;
}

void updateTracks(ArrayList<PVector> detections, ArrayList<Rectangle> detBoxes, float gatePx){
  if (detections == null) detections = new ArrayList<PVector>();
  if (detBoxes == null) detBoxes = new ArrayList<Rectangle>();

  boolean[] used=new boolean[detections.size()];
  float gate2=gatePx*gatePx;
  for(int ti=0;ti<tracks.size();ti++){
    Track t=tracks.get(ti);
    t.age++;
    t.missed++;
    PVector pred=new PVector(t.pos.x+t.vel.x, t.pos.y+t.vel.y);
    int bestIdx=-1;
    float bestCost=1e9f;
    for(int i=0;i<detections.size();i++){
      if(used[i]) continue;
      PVector m=detections.get(i);
      Rectangle mb=detBoxes.get(i);
      float dx=m.x-pred.x, dy=m.y-pred.y, d2=dx*dx+dy*dy;
      if(d2>gate2) continue;
      float j=(t.lastBox!=null)? iou(t.lastBox,mb) : 0;
      float cost=d2 - (j*2000.0f);
      if(cost<bestCost){
        bestCost=cost;
        bestIdx=i;
      }
    }
    if(bestIdx>=0){
      PVector m=detections.get(bestIdx);
      Rectangle mb=detBoxes.get(bestIdx);
      t.raw.set(m);
      // snappier tracking
      float beta  = 0.35f;   // velocity update
      float alpha = 0.55f;   // position update
      t.vel.x=0.8f*t.vel.x+beta*(m.x-t.pos.x);
      t.vel.y=0.8f*t.vel.y+beta*(m.y-t.pos.y);
      t.pos.x=t.pos.x+alpha*(m.x-t.pos.x);
      t.pos.y=t.pos.y+alpha*(m.y-t.pos.y);
      t.lastBox=mb;
      t.missed=0;
      t.hits++;
      if(!t.confirmed && t.hits>=3) t.confirmed=true;
      used[bestIdx]=true;
    }
  }
  for(int i=0;i<detections.size();i++) {
    if(!used[i]) tracks.add(new Track(detections.get(i), detBoxes.get(i)));
  }
  for(int i=tracks.size()-1;i>=0;i--){
    Track t=tracks.get(i);
    int maxMiss=t.confirmed?20:8;
    if(t.missed>maxMiss) tracks.remove(i);
  }
}

int countConfirmed(){
  int c=0;
  for(int i=0;i<tracks.size();i++) if(tracks.get(i).confirmed) c++;
  return c;
}

// Input

void keyPressed(){

  if (key=='b') captureFloor();
  if (key=='o') debugView=!debugView;
  if (key=='l') {
    float[] H=loadHomography();
    if (H!=null){
      Hcam2proj=normalizeH(H);
      println("Loaded H+k1. Press 'r' to rebuild hull from current points.");
    }
  }
  if (key=='s') saveHomography(Hcam2proj);
  if (key=='p'){
    if (projWin==null) openProjectorWindow();
    else closeProjectorWindow();
  }

  if (key=='t'){
    targetMode = (targetMode==TargetMode.GRID) ? TargetMode.RANDOM : TargetMode.GRID;
    println("Target mode: " + targetMode);
    calibIdx = 0;
    if (targetMode==TargetMode.RANDOM) buildRandomTargets();
  }
  if (key=='G'){
    if (targetMode==TargetMode.RANDOM){
      buildRandomTargets();
      calibIdx=0;
      println("Regenerated random targets.");
    }
  }

  if (key=='v'){
    previewTargets = true;
    calibrating = false; // don't accept clicks yet
    calibIdx = 0;
    if (targetMode==TargetMode.RANDOM) buildRandomTargets();
    println("PREVIEW MODE: showing all targets. Inspect spread, then press ENTER.");
  }

  if (keyCode == ENTER || keyCode == RETURN){
    if (previewTargets){
      previewTargets = false;
      calibrating = true;
      calibIdx = 0;
      println("Now CALIBRATING. For each red circle, click the Kinect LEFT pane.");
    }
  }

  // legacy 'c' still forces calibration mode right away
  if (key=='c'){
    calibrating=true;
    previewTargets=false;
    println("Calibration started (no preview). Pairs: "+camPts.size()+
            "   Mode="+targetMode);
  }

  if (key=='d'){ // discard current target & advance
    if (!calibrating) calibrating=true;
    int totalTargets = (targetMode==TargetMode.GRID) ? projTargetsNorm.length
                                                     : max(1, targetsLocal.size());
    int prev = calibIdx % totalTargets;
    calibIdx++;
    int next = calibIdx % totalTargets;
    println("Discarded target "+prev+" → now showing "+next+". Pairs kept: "+camPts.size());
  }

  if (key=='u'){
    if (camPts.size()>0){
      camPts.remove(camPts.size()-1);
      projPts.remove(projPts.size()-1);
      calibIdx=max(0,calibIdx-1);
      println("Undo. Pairs: "+camPts.size());
    }
  }

  if (key=='n'){
    camPts.clear();
    projPts.clear();
    calibIdx=0;
    calibrating=false;
    previewTargets=false;
    hullCam.clear();
    hullProj.clear();
    lastKept = new int[0];
    synchronized (trailSteps) {
      trailSteps.clear();
    }
    println("Cleared calibration & trails. Press 'v' to preview, ENTER to confirm, then click.");
  }

  if (key=='r'){
    float rms = doFit();
    if (rms>=0) println("Refit done. RMS="+nf(rms,1,2)+" px");
  }

  if (key=='k'){
    testPattern = !testPattern;
    println("Projector test pattern: "+(testPattern?"ON":"OFF"));
  }

  if (key=='h'){
    if (Math.abs(hullShrink-1.0f)<1e-3f) hullShrink = 0.97f;
    else hullShrink = 1.00f;
    println("hullShrink="+hullShrink+" (press 'r' to rebuild hull)");
  }

  if (key=='['){
    ACCEPT_THRESH = max(4.0f, ACCEPT_THRESH-2.0f);
    println("ACCEPT_THRESH="+nf(ACCEPT_THRESH,1,1)+" px  (press 'r')");
  }
  if (key==']'){
    ACCEPT_THRESH = min(60.0f, ACCEPT_THRESH+2.0f);
    println("ACCEPT_THRESH="+nf(ACCEPT_THRESH,1,1)+" px  (press 'r')");
  }
  if (key==','){
    KEEP_FRACTION = max(0.50f, KEEP_FRACTION-0.05f);
    println("KEEP_FRACTION="+int(KEEP_FRACTION*100)+"%  (press 'r')");
  }
  if (key=='.'){
    KEEP_FRACTION = min(0.95f, KEEP_FRACTION+0.05f);
    println("KEEP_FRACTION="+int(KEEP_FRACTION*100)+"%  (press 'r')");
  }

  // toggle hull gating
  if (key=='g'){
    useHullGate = !useHullGate;
    println("Hull gating: " + (useHullGate ? "ON" : "OFF"));
  }

  // toggle fractal trails (includes ambient)
  if (key=='f'){
    enableTrails = !enableTrails;
    synchronized (trailSteps) {
      trailSteps.clear();
    }
    println("Fractal trails: " + (enableTrails ? "ON" : "OFF"));
  }
}

void mousePressed(){
  // only allow capture in calibration mode (NOT preview)
  if (!calibrating) {
    if (previewTargets) {
      println("You're in PREVIEW. Press ENTER to start calibration, THEN click.");
    } else {
      println("Not calibrating. Press 'v' then ENTER first.");
    }
    return;
  }

  if (mouseX<leftX || mouseX>=leftX+paneW || mouseY<leftY || mouseY>=leftY+paneH){
    println("Click inside LEFT pane.");
    return;
  }
  float u=(mouseX-leftX)/scaleDisp, v=(mouseY-leftY)/scaleDisp;
  if (u<0||u>=W||v<0||v>=H){
    println("Click inside LEFT pane.");
    return;
  }
  PVector tgt = currentProjTargetLocal();
  if (tgt==null){
    println("No target? (Shouldn't happen unless calib off).");
    return;
  }
  camPts.add(new PVector(u,v));
  projPts.add(tgt.copy());
  println("Pair "+camPts.size()+": cam("+nf(u,1,1)+","+nf(v,1,1)+") -> proj("+int(tgt.x)+","+int(tgt.y)+")");
  calibIdx++; // advance after successful click
}

// Floor Capture

void captureFloor(){
  int[] depth=kinect.getRawDepthData();
  if (depth==null||depth.length!=W*H) return;
  floorMM=new int[depth.length];
  final int FRAMES=4; // fewer frames = less initial spike
  Arrays.fill(floorMM,0);
  for(int f=0;f<FRAMES;f++){
    delay(30);
    int[] d=kinect.getRawDepthData();
    if(d==null||d.length!=floorMM.length) continue;
    for(int i=0;i<floorMM.length;i++) {
      floorMM[i]+= (d[i]>0? d[i]:0); 
    }
  }
  
  for(int i=0;i<floorMM.length;i++) {
    floorMM[i]/=FRAMES;
  }
  println("Floor captured.");
}
