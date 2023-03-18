import cv2
import pyrealsense2 as rs
import numpy as np
import datetime



# Define the minimum and maximum depth values for the detection
MIN_DEPTH = 3
MAX_DEPTH = 10
DEPTH_LOW = 0


# Read the video from specified path
cam = cv2.VideoCapture(0)
ret,frame = cam.read()
cv2.imwrite('BGImg.jpg', frame)
cam.release()
cv2.destroyAllWindows()


# Initialize the value of the variable
width_min = 20
height_min = 50
detect = []
check_status = 0
text_default = "No Detect"


# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


# Set the variable for Background Subtraction
subtraction = cv2.createBackgroundSubtractorMOG2()


# Start camera
pipeline.start(config)

try:
    while True:
        #Imread Bankground Img
        BGImg = cv2.imread('BGImg.jpg')

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        #convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #-----------------------------------------------------------
        # Image Preprocessing processes
        HumanDetectRGB = abs(BGImg-color_image)
        
        # Background Subtraction
        sub = subtraction.apply(HumanDetectRGB) 

        # Morphological image processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.morphologyEx(sub, cv2. MORPH_CLOSE, kernel)  

        # Find the contours in the binary image
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # The boundary line of the outside leave zone
        cv2.line(color_image, (443,328), (443,0), (255, 0, 0), 2)

        # Find the centroid of the largest contour
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                 # Get the depth value at the centroid
                depth = depth_frame.get_distance(cx, cy)

                # The condition for detecting depth
                if MIN_DEPTH <= depth <= MAX_DEPTH:
                    
                    # Point indicating the centroid position of the contour.
                    cv2.circle(color_image, (cx,cy), 4, (0, 0, 255), -1)

                    x, y, w, h = cv2.boundingRect(c)
                    validate_contour = (w >= width_min) and (h >= height_min)
                    if not validate_contour:
                        continue

                    # Bounding rectangle that fully covers the contour area
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            
                    # Start the detection process and condition status
                    if (cx >= 102 and cx <= 307 and cy >= 0 and cy <= 282):
                        text = "Human stay on the bed"
                        check_status = 1
                        if check_status is 1:
                            check_status != 2 and check_status != 3
                    elif (cx >= 185 and cx <= 444 and cy >= 290 and cy <= 322):
                        text = "Human exited the bed zone"
                        check_status = 2
                        if check_status is 2:
                            check_status != 1 and check_status != 3
                    elif (cx >= 311 and cx <= 444 and cy >= 0 and cy <= 282):
                        text = "Human exited the bed zone"
                        check_status = 2
                        if check_status is 2:
                            check_status != 1 and check_status != 3
                    elif (cx >= 9 and cx <= 444 and cy >= 328 and cy <= 466):
                        text = "Human exited the bed zone"
                        check_status = 2
                        if check_status is 2:
                            check_status != 1 and check_status != 3
                    elif (cx >= 9 and cx <= 185 and cy >= 0 and cy <= 460):
                        text = "Human exited the bed zone"
                        check_status = 2
                        if check_status is 2:
                            check_status != 1 and check_status != 3
                    else:
                        if (cx >= 448 and cx <= 640 and cy >= 0 and cy <= 322):
                            text = "Human exited the leave zone"
                            check_status = 3
                            if check_status is 3:
                                check_status != 1 and check_status != 2
                        elif (cx >= 447 and cx <= 640 and cy >= 328 and cy <= 466):
                            text = "Human exited the leave zone"
                            check_status = 3
                            if check_status is 3:
                                check_status != 1 and check_status != 2
              
                    cv2.putText(color_image, "Status: {}".format(text), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)               
                else:
                    cv2.putText(color_image, text_default, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(color_image, "No Human", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)

        
        # Current date and time
        cv2.putText(color_image, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the monitor
        cv2.imshow('Moniter', color_image)


        if cv2.waitKey(1) == ord("q"):
            break
        

finally:
    pipeline.stop()
    cv2.destroyAllWindows()