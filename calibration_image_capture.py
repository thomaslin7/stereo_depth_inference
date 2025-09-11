import numpy as np
import cv2
from imutils.video import VideoStream
import time

print('Starting the Calibration. Press and maintain the space bar to exit the script\n')
print('Push (s) to save the image you want and push (c) to see next frame without saving the image')

id_image=33 # change this back to 0 (if calibration image count is 0)

# termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
video_stream_R = VideoStream(src=0).start()
video_stream_L = VideoStream(src=1).start()
time.sleep(2.0)

while True:
    frameR= video_stream_R.read()
    frameL= video_stream_L.read()
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(9,6),None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
    retL, cornersL = cv2.findChessboardCorners(grayL,(9,6),None)  # Same with the left camera
    cv2.imshow('imgR',frameR)
    cv2.imshow('imgL',frameL)

    # If found, add object points, image points (after refining them)
    if (retR == True) & (retL == True):

        print("Found chessboard corners")

        corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)    # Refining the Position
        corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(grayR,(9,6),corners2R,retR)
        cv2.drawChessboardCorners(grayL,(9,6),corners2L,retL)
        cv2.imshow('VideoR',grayR)
        cv2.imshow('VideoL',grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):   # Press "s" to save the images and "c" if you don't want to
            str_id_image= str(id_image)
            print('Images ' + str_id_image + ' saved for right and left cameras')
            cv2.imwrite('calibration_images/chessboard-R'+str_id_image+'.png',frameR) # Save the image in the calibration_images folder
            cv2.imwrite('calibration_images/chessboard-L'+str_id_image+'.png',frameL)
            id_image=id_image+1
        else:
            print('Images not saved')

    # End the program
    if cv2.waitKey(0) & 0xFF == ord('q'):   # Press "q" to exit the program
        break

# Release the cameras
video_stream_R.stop()
video_stream_L.stop()
cv2.destroyAllWindows()    
