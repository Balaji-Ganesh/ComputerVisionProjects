import cv2  # For dealing with the cameras and basic image related operations..
import mediapipe  # For dealing with the hand-detection and hand-tracking
import time  # For FPS calculation..


class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        This is the constructor for the HandDetector class. It prepares the MediaPipe Detector by some default values.
        :param static_image_mode:  # False:(Faster) Detects and Tracks based on confidence levels, when falls < confidence levels - starts detecting. True:(Bit slower) Only detection
        :param max_num_hands: # no. of hands will be in the frame (at max).
        :param min_detection_confidence: # Currently set to 50%  (default too)
        :param min_tracking_confidence: # Currently set to 50%  (default too)
        """
        # Create the data members of the class..
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Setup the detector..
        self.mediaPipeHands = mediapipe.solutions.hands  # To get the hand detection. Its the formality..
        self.handsDetector = self.mediaPipeHands.Hands(self.static_image_mode,  self.max_num_hands,
                                                       self.min_detection_confidence, self.min_tracking_confidence)
        self.handsDraw = mediapipe.solutions.drawing_utils  # For drawing the landmarks detected and the connections between those..

    def detect_hands(self, image, is_draw=False):
        """
        This method detects the hands in the passed image and draws the list of all the 21 points (if is_draw=True).
        :param image: Image on which the detection processing has to be done.
        :param is_draw: A boolean parameter. True: Draws the landmarks, False:(Default) not
        :return: An image with the drawn landmark points (if is_draw=True, else same image as passed.)
        """
        frame_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # mediapipe works on RGB (OpenCV-BGR), so conversion.
        # Detect the hands in the frame..
        self.detections = self.handsDetector.process(frame_RGB)

        if self.detections.multi_hand_landmarks:                         # Only if got some landmarks..
            for hand_landmarks in self.detections.multi_hand_landmarks:  # Work on each single hand detected..
                if is_draw:
                    self.handsDraw.draw_landmarks(image, hand_landmarks, self.mediaPipeHands.HAND_CONNECTIONS)  # Draw the landmarks on the image passed...

        return image

    def detected_landmarks(self, image, hand_idx=0, is_draw=False, color=(255, 0, 0)):
        """
        Returns the co-ordinates of all the 21 detected hand landmarks (order will be as per the MediaPipe)
        :param image: Image on which the draw is to be performed. Also used to get the image dimensions.
        :param hand_idx: Index of the hand when multiple hands. (Default=0, i.e., Single hand). Pass '-1' to detect all the hands or specific hand-counts.
        :param is_draw: Boolean parameter. True: draws all the landmarks
        :param color: Color with which the landmarks has to be drawn. Default=Blue
        :return: List of tuples of all the 21 points. Tuple format: (id, x_pt, y_pt).
        NOTE
        ----
            Sometimes may get the list as empty when no hand was detected, so make sure to handle this condition to avoid errors.
        """
        total_landmarks = []                                # An empty list to store all the landmarks
        if self.detections.multi_hand_landmarks:            # Only if the landmarks are detected..
            image_height, image_width, frame_channels = image.shape
            if hand_idx != -1:
                required_hand = self.detections.multi_hand_landmarks[hand_idx]
            else:
                required_hand = self.detections.multi_hand_landmarks
            for hand_landmarks in required_hand:
                for id, landmarks in enumerate(hand_landmarks.landmark):  # For parsing the each landmark.. !! We get co-ordinates in the float which are by ratios of height and width of image. So to get back actual values, do multiply..
                    x_pt, y_pt = int(landmarks.x * image_width), int(landmarks.y * image_height)
                    total_landmarks.append((id, (x_pt, y_pt)))
                    if is_draw:
                        cv2.circle(image, (x_pt, y_pt), 10, color, cv2.FILLED)

        detected_hands_count = 0                # To store how many hands were detected..
        if self.detections.multi_hand_landmarks is not None:
            detected_hands_count = len(self.detections.multi_hand_landmarks)
        return detected_hands_count, total_landmarks


def main():
    # For camera..
    capture = cv2.VideoCapture(0)  # For reading the frames from camera..
    # Object instantiations of the HandDetector class..
    hand_detector = HandDetector()
    # For FPS..
    prev_time, curr_time = 0, 0.1  # Intially set to 0's
    detection_status = ""       # To show the status on the window..
    while True:
        read_status, frame = capture.read()
        if read_status:
            frame = cv2.flip(frame, 1)
            image = hand_detector.detect_hands(frame, is_draw=True)
            num_hands, landmarks = hand_detector.detected_landmarks(frame, hand_idx=-1, is_draw=False)
            if num_hands != 0:
                # print(landmarks[4])
                detection_status = "Detecting "+str(num_hands)+" hand(s)..."
            else:
                detection_status = "Cannot Detect.."


            # FPS calculations...
            curr_time = time.time()  # Get the current time...
            # print(prevTime, currTime)
            FPS = 1 / (curr_time - prev_time)  # Calculate the frequency..
            prev_time = curr_time  # For the next iteration, current will be the previous right..!! So doing the same here...

            # Write on the image..
            cv2.putText(frame, "FPS: ", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, str(int(FPS)), (90, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Status: ", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, detection_status, (135, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Detector", frame)
        else:
            print("[ERROR] Couldn't able to read the image. Please check the camera connection. Exiting from program.")
            break
        if cv2.waitKey(1) == 27:
            break

    # Release the instances hold...
    cv2.destroyAllWindows()
    capture.release()
    print("Successfully released the resources. Job Done. Thanks for utilizing our service... Have a good day..!!")


if __name__ == "__main__":
    main()

