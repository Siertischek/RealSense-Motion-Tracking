# -----------------------------------------+
# Team 24                                  |
# Brendan Verbrugge, Sam Ertischek         |
# CSCI 442, Assignment 3                   |
# Last Updated: 3/10, 2023                 |
# -----------------------------------------|
# This program captures video from a       |
# RealSense Depth Camera and displays the  |
# depth as a colormap in an                |
# attached window. It also tracks a        |
# user selected object and displays the    |
# distance of that tracked object from the |
# camera in another attached window.       |
# The vstack function doesn't seem to work |
# correctly so we have the color and depth |
# map in one window, the distance from     |
# camera in another window. The combined   |
# with vstack is in another window.        |
# -----------------------------------------+

# ---------------------------------------------------------------------

# IMPORTS

# ---------------------------------------------------------------------


import pyrealsense2 as rs
import numpy as np
import cv2

# ---------------------------------------------------------------------

# CAMERA SETUP

# ---------------------------------------------------------------------

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Tracker
tracker = cv2.TrackerKCF_create()
bbox_flag = 0
ok = False

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# align object
align_to = rs.stream.color
align = rs.align(align_to)

# ---------------------------------------------------------------------------

# CAPTURE LOOP

# ---------------------------------------------------------------------------

cv2.namedWindow('Depth')
while True:

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    if bbox_flag > 0:
        # Tracking
        ok, bbox = tracker.update(color_image)
        if ok:
            # Tracking success
            bbox_flag += 1
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)

        else :
            # Tracking failure
            cv2.putText(color_image, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    elif bbox_flag == 0 and color_frame:
        # Tracking Setup
        bbox = cv2.selectROI(color_image, False)
        ok = tracker.init(color_image, bbox)
        bbox_flag += 1
        

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HSV)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    
    # Proximity Image 
    img2 = np.zeros([depth_colormap_dim[0],depth_colormap_dim[1]*2,depth_colormap_dim[2]])

    depth = depth_image[479,635].astype(float)
    distance = depth * depth_scale

    d1 = (int(bbox[0] + bbox[2])+245, 230-int(bbox[1]))
    d2 = (int(bbox[0] + bbox[2])+295, 230-int(bbox[1]))
    cv2.rectangle(img2, d1, d2, (255,0,0), 2, 1)
    cv2.rectangle(img2, (630,330), (650,350), (0,0,255), 2, 1)
    
    if ok and bbox_flag > 1:
        # Proximity detection will only work if an object is being tracked
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        # Capture depth image of bounding box
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = depth_image[p1[0]:p2[0],p1[1]:p2[1]]
        
        # Find minimum distance within bounding box
        minimum_distance = depth_image.min()
        
    

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        img1 = np.hstack((resized_color_image, depth_colormap))
        images=np.vstack((img1, img2))
    else:
        img1 = np.hstack((color_image, depth_colormap))
        images = np.vstack((img1, img2))

    # Show images
    cv2.imshow('Depth', img1)
    cv2.imshow('Proximity', img2)
    #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()