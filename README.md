
# LightSteps

LightSteps is an interactive installation that uses an overhead Kinect v2 depth camera to detect and track people moving across a floor in real time. The system maps tracked positions into projector space and generates animated fractal footprints at each step.

The work is inspired by the idea of exploration from the Viraat Roop in the Mahabharata. Instead of showing this literally, the installation allows each participant’s movement to generate evolving visual trails where motion creates shifting patterns and paths over time.

## What this is

This Processing sketch uses a Kinect v2 depth camera mounted overhead to detect people on the floor and track them. It opens a second fullscreen window on the projector and draws animated fractal footprints where it detects stepping motion.

It includes a calibration mode that maps Kinect camera pixels to projector pixels using a homography and saved radial distortion parameter.

## Hardware needed

Kinect v2 connected to your computer  
Kinect mounted overhead and pointing down at the floor  
A projector connected as a second monitor  

## Required Processing libraries

KinectPV2  
OpenCV for Processing (gab.opencv)  
Sound (processing.sound)  

Optional audio file in the sketch data folder  
dark-cluster-16449.mp3  

## Important settings

projectorDisplayIndex  
0 is your main monitor  
1 is the next monitor  

If the projector window opens on the wrong screen change this number.

W and H are Kinect depth resolution values 512 x 424  
Do not change these unless needed.

DS is the downscale factor for the OpenCV mask  
DS = 2 improves performance.

## How it works

1. Capture the empty floor depth as the background  
2. Each frame compares the current depth to the floor depth  
3. Pixels between H_MIN and H_MAX millimeters above the floor are kept  
4. The mask is downscaled and OpenCV contours are used to find person sized blobs  
5. Blobs are tracked across frames and assigned IDs  
6. A foot point is selected near the bottom of each tracked blob  
7. This point is mapped into projector space after calibration  
8. Fractal footprints are drawn at the mapped projector position and fade out over time  

## Basic workflow

Clear the floor so nobody is in view  
Press b to capture the floor background  
Press p to open the projector window  
Press v to preview calibration targets  
Press ENTER to start calibration  

For each red target  
Click the matching point in the Kinect LEFT pane  

After collecting enough points  
Press r to fit the homography  
Press s to save to data/homography.txt  

Walking in view should now create footprints

Press l later to load the saved homography

## Output

data/homography.txt  
3 lines for the 3x3 homography matrix  
1 line for k1  

## Notes

The LEFT pane shows the Kinect depth image and debug overlays  
The RIGHT pane mirrors projector space  
The projector window is a separate fullscreen PApplet used for projection
