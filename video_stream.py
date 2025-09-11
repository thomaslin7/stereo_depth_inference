import cv2
from datetime import datetime
from imutils.video import VideoStream
import time

vs1 = VideoStream(src=0).start()
time.sleep(2.0)
vs2 = VideoStream(src=1).start()
time.sleep(2.0)

while(True):
    frame1 = vs1.read()
    frame2 = vs2.read()
    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2',frame2)

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break
    # if key == ord('c'):
    #     cv2.imwrite(f"calibrate01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", frame1)
    #     cv2.imwrite(f"calibrate02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", frame2)

vs1.stop()
vs2.stop()
cv2.destroyAllWindows()
