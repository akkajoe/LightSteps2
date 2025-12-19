# README

## What this is

This Processing sketch uses a Kinect v2 (depth camera) mounted overhead to detect people on the floor and track them. It opens a second fullscreen window on the projector and draws animated fractal footprints where it thinks people are stepping.

It also has a calibration mode that maps Kinect camera pixels to projector pixels using a homography (and a saved k1 value).

Optional: it plays background music if the mp3 file is present.

---

## What you need

Hardware

* Kinect v2 connected to your computer
* Kinect mounted overhead, pointing down at the floor
* A projector connected as a second monitor

Processing libraries

* KinectPV2
* OpenCV for Processing (gab.opencv)
* Sound (processing.sound)

Files

* Optional audio file in the sketch “data” folder:

  * dark-cluster-16449.mp3

---

## Important settings to check

* projectorDisplayIndex (monitor index for the projector window)

  * Example: 0 is your main monitor, 1 is the next, etc.
  * If the projector window shows on the wrong screen, change this number.

* W and H are Kinect depth resolution (512 x 424). Do not change unless you know what you are doing.

* DS is the downscale factor for the OpenCV mask. DS = 2 is faster.

---

## How it works (simple)

1. Capture the empty floor depth as the background.
2. Every frame, compare current depth to the floor depth.
3. Keep pixels that are between H_MIN and H_MAX millimeters above the floor.
4. Downscale the mask and use OpenCV contours to find person-sized blobs.
5. Track blobs over time and assign IDs.
6. For each tracked person, pick a “foot point” near the bottom of their detection.
7. Map that point into projector space (after calibration).
8. Draw fractal footprints at that projector position, and fade them out over time.

---

## Basic workflow (recommended)

1. Clear the floor (nobody in view).
2. Press b to capture the floor background.
3. Press p to open the projector fullscreen window (if it is not already open).
4. Press v to preview all calibration targets on the floor.
5. Look at the target dots and confirm they are spread out well.
6. Press ENTER to start calibration (one target at a time).
7. For each red target:

   * Click the matching point in the Kinect LEFT pane (camera view).
8. After collecting enough pairs, press r to fit and build the hull.
9. Press s to save the homography to data/homography.txt.
10. Now walking should create footprints on the projector.

Tip: you can press l later to load the saved homography.

---

## Controls (keys)

Floor and windows

* b : capture floor (do this when the floor is empty)
* p : open/close projector window

Calibration

* v : preview all calibration targets (no clicking yet)
* ENTER : switch from preview to calibration (single target)
* c : start calibration immediately (skips preview)
* d : discard current target and move to the next one
* u : undo last accepted calibration pair
* r : fit homography and rebuild hull (RANSAC + refine)
* n : clear all calibration points and hull (restart)
* l : load homography from file
* s : save homography to file

Target mode

* t : toggle target mode (GRID or RANDOM)
* G : regenerate random targets (RANDOM mode)

Hull settings

* g : toggle hull gating on/off (safety mask)
* h : toggle hull shrink (1.00 vs 0.97)
* [ / ] : decrease/increase ACCEPT_THRESH
* , / . : decrease/increase KEEP_FRACTION

Debug and tests

* o : toggle debug overlays
* k : toggle projector test pattern

Trails

* f : toggle fractal trails on/off (also clears existing trails)

---

## Mouse input

* In calibration mode only:

  * Click inside the LEFT pane (Kinect view) to record the camera point for the current red target.
* If you click while in preview mode, the sketch will tell you to press ENTER first.

---

## Output files

* data/homography.txt

  * 3 lines for the 3x3 homography matrix
  * 1 line for k1

---

## Troubleshooting

Projector window is on the wrong screen

* Change projectorDisplayIndex and restart the sketch.

Nothing is detected

* Make sure you pressed b with an empty floor.
* Check H_MIN and H_MAX (height above floor in mm).
* Check MIN_AREA and MAX_AREA (filters out too-small or too-big blobs).

Footprints show outside the usable floor area

* Press r after calibration to rebuild the hull.
* Make sure hull gating is ON (press g).
* Try hullShrink 0.97 (press h then r).

Performance is slow

* Press f to disable trails.
* Increase DS (more downscaling) if needed.
* Reduce MAX_CONTOURS or randomSpawnsPerSecond.

Audio does not play

* Make sure dark-cluster-16449.mp3 is inside the sketch data folder.
* If the file is missing, the sketch will still run.

---

## Notes

* The LEFT pane shows the Kinect depth image and debug overlays.
* The RIGHT pane mirrors projector space in the main window.
* The projector window is a separate fullscreen PApplet used for real projection.
