import cv2  # For dealing with the cameras and basic image related operations..
import mediapipe  # For dealing with the hand-detection and hand-tracking
import time  # For FPS calculation..

############################################ Setting up the things ##############################################
capture = cv2.VideoCapture(0)  # For reading the frames from camera..
mediaPipeHands = mediapipe.solutions.hands  # To get the hand detection. Its the formality..
handsDetector = mediaPipeHands.Hands(static_image_mode=False,  # False:(Faster) Detects and Tracks based on confidence levels, when falls < confidence levels - starts detecting. True:(Bit slower) Only detection
                                     max_num_hands=2,  # no. of hands will be in the frame (at max).
                                     min_detection_confidence=0.5,  # Currently set to 50%  (default too)
                                     min_tracking_confidence=0.5)  # Currently set to 50%  (default too)
handsDraw = mediapipe.solutions.drawing_utils  # For drawing the landmarks detected and the connections between those..
# For FPS..
prevTime, currTime = 0, 0.1  # Intially set to 0's
############################################################################################################################
# Start of main work..
while True:
    read_status, frame = capture.read()
    frame_height, frame_width, frame_channels = frame.shape
    # if read_status == True:
    # cv2.flip(frame, 3)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe works on RGB (OpenCV-BGR), so conversion.
    # Detect the hands in the frame..
    detections = handsDetector.process(frame_RGB)
    # print(detections.multi_hand_landmarks)                   # this information is used for further processings..
    if detections.multi_hand_landmarks:  # Only if got some landmarks..
        for hand_landmarks in detections.multi_hand_landmarks:  # Work on each single hand detected..
            handsDraw.draw_landmarks(frame, hand_landmarks, mediaPipeHands.HAND_CONNECTIONS)
            for id, landmarks in enumerate(hand_landmarks.landmark):  # For parsing the each landmark.. !! We get co-ordinates in the float which are by ratios of height and width of image. So to get back actual values, do multiply..
                x_pt, y_pt = int(landmarks.x * frame_width), int(landmarks.y * frame_height)
                print(id, (x_pt, y_pt))
                # check..
                cv2.circle(frame, (x_pt, y_pt), 10, (255, 0, 255), cv2.FILLED)

    # FPS calculations...
    currTime = time.time()  # Get the current time...
    # print(prevTime, currTime)
    FPS = 1 / (currTime - prevTime)  # Calculate the frequency..
    prevTime = currTime  # For the next iteration, current will be the previous right..!! So doing the same here...
    # Write on the image..
    cv2.putText(frame, "FPS: " + str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Hand Detector", frame)
    if cv2.waitKey(1) == 27:
        break

# Release the instances hold...
cv2.destroyAllWindows()
capture.release()
print("Successfully released the resources. Job Done. Thanks for utilizing our service... Have a good day..!!")
