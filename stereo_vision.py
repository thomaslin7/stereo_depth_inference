import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from imutils.video import VideoStream
import time
import open3d as o3d
from datetime import datetime

# Filtering
kernel= np.ones((3,3),np.uint8)

def create_point_cloud(disparity_map, Q_matrix, color_image=None):
    """
    Convert disparity map to 3D point cloud using Q matrix from stereo rectification
    
    Args:
        disparity_map: Normalized disparity map (0-1 range)
        Q_matrix: 4x4 reprojection matrix from stereo rectification
        color_image: Optional color image for colored point cloud
    
    Returns:
        open3d.geometry.PointCloud: 3D point cloud
    """
    # Convert disparity to depth
    # For proper depth calculation, we need to use the Q matrix
    # But first, let's create a proper disparity map for reprojection
    
    # Create a disparity map suitable for reprojectImageTo3D
    # The disparity should be in the format expected by cv2.reprojectImageTo3D
    disparity_for_3d = (disparity_map * 128).astype(np.float32)  # Scale to typical disparity range
    
    # Use OpenCV's reprojectImageTo3D to convert disparity to 3D points
    points_3d = cv2.reprojectImageTo3D(disparity_for_3d, Q_matrix)
    
    # Create mask for valid points (non-zero disparity)
    mask = disparity_map > 0.1  # Threshold to remove invalid points
    
    # Extract valid 3D points
    points = points_3d[mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if color image is provided
    if color_image is not None:
        colors = color_image[mask] / 255.0  # Normalize to 0-1
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def save_point_cloud(disparity_map, Q_matrix, color_image=None, filename=None):
    """
    Generate and save point cloud to file
    
    Args:
        disparity_map: Normalized disparity map
        Q_matrix: 4x4 reprojection matrix
        color_image: Optional color image
        filename: Output filename (if None, auto-generate with timestamp)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"point_cloud_{timestamp}.ply"
    
    # Create point cloud
    pcd = create_point_cloud(disparity_map, Q_matrix, color_image)
    
    # Save to file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved as: {filename}")
    
    return filename

def coords_mouse_disp(event,x,y,flags,param):
    print("Mouse event:", event)
    if event == cv2.EVENT_LBUTTONDOWN:  # single click
        # print(x,y,disp[y,x],filteredImg[y,x])
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')
        
# This section has to be uncommented if you want to take mesurements and store them in excel
    # ws.append([counterdist, average])
    # print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
    # if (counterdist <= 85):
    #     counterdist += 3
    # elif(counterdist <= 120):
    #     counterdist += 5
    # else:
    #     counterdist += 10
    # print('Next distance to measure: '+str(counterdist)+'cm')

# Mouseclick callback
wb=Workbook()
ws=wb.active  

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,50):   # Put the number of pictures you have taken for the calibration in range(0,?), starting from the image number 0
    t= str(i)
    ChessImaR= cv2.imread('calibration_images/chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('calibration_images/chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,(9,6),None)  # Define the number of chess corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,(9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
#flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# Print intrinsic matrix (using left camera as reference)
print("Intrinsic Matrix (Left Camera):")
print(mtxL)

# Print extrinsic matrix (rotation and translation between cameras)
print("\nExtrinsic Matrix - Rotation Matrix (R):")
print(R)
print("\nExtrinsic Matrix - Translation Vector (T):")
print(T)

###################

# Intrinsic Matrix:
# [[1.42945992e+03 0.00000000e+00 6.83046151e+02]
#  [0.00000000e+00 1.42434397e+03 4.81538797e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

# Extrinsic Matrix - Rotation Matrix (R):
# [[ 0.99795576 -0.0120479  -0.06276256]
#  [ 0.01112565  0.99982525 -0.01502309]
#  [ 0.06293259  0.01429411  0.99791541]]

# Extrinsic Matrix - Translation Vector (T):
# [[-3.2806174 ]
#  [ 0.03283476]
#  [-0.15686016]]

 ###################

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables the program to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)
#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
video_stream_R = VideoStream(src=0).start()
video_stream_L = VideoStream(src=1).start()
time.sleep(2.0)

# Create temporary VideoCapture objects just to set exposure/WB
capR = cv2.VideoCapture(0)
capL = cv2.VideoCapture(1)

# Disable auto-exposure (0.25 = manual mode in OpenCV)
capR.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
capL.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# Set manual exposure (adjust value depending on lighting)
capR.set(cv2.CAP_PROP_EXPOSURE, -6)
capL.set(cv2.CAP_PROP_EXPOSURE, -6)

# Disable auto white balance
capR.set(cv2.CAP_PROP_AUTO_WB, 0)
capL.set(cv2.CAP_PROP_AUTO_WB, 0)

while True:
    # Start reading camera images
    frameR= video_stream_R.read()
    frameL= video_stream_L.read()

    # Rectify the images on rotation and alignement
    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

##    # Draw red lines
##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the lines on the images Then numer of line is defines by the image Size/20
##        Left_nice[line*20,:]= (0,0,255)
##        Right_nice[line*20,:]= (0,0,255)
##
##    for line in range(0, int(frameR.shape[0]/20)): # Draw the lines on the images Then numer of line is defines by the image Size/20
##        frameL[line*20,:]= (0,255,0)
##        frameR[line*20,:]= (0,255,0)    
        
    # Show the undistorted images
    #cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
    #cv2.imshow('Normal', np.hstack([frameL, frameR]))

    # Convert from color(BGR) to gray
    grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)   #.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

##    # Resize the image for faster executions
##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

    # Show the result for the Depth_image
    #cv2.imshow('Disparity', disp)
    #cv2.imshow('Closing',closing)
    #cv2.imshow('Color Depth',disp_Color)

    # Draw a red dot at the measurement location
    # cv2.circle(filt_Color, (730, 480), 10, (0, 0, 255), -1)  # Red circle with radius 10

    # Add instructions to the display
    # cv2.putText(filt_Color, "Press 's' to save point cloud", (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(filt_Color, "Press 'q' to quit", (10, 60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Depth Map',filt_Color)

    # Mouse click
    # cv2.setMouseCallback("Depth Map",coords_mouse_disp,filt_Color)

    # Measure the distance result
    # average=0
    # x = 730
    # y = 480
    # for u in range (-1,2):
    #     for v in range (-1,2):
    #         average += disp[y+u,x+v]
    # average=average/9
    # Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    # Distance= np.around(Distance*0.01,decimals=2)
    # print('Distance: '+ str(Distance)+' m')
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # elif key == ord('s'):  # Press 's' to save point cloud
    #     print("Saving point cloud...")
    #     try:
    #         # Use the current disparity map and Q matrix to create point cloud
    #         # Use the left rectified image for colors
    #         save_point_cloud(disp, Q, Left_nice)
    #     except Exception as e:
    #         print(f"Error saving point cloud: {e}")
    
# Save excel
##wb.save("data.xlsx")

# Release the cameras
video_stream_R.stop()
video_stream_L.stop()
capR.release()
capL.release()
cv2.destroyAllWindows()
