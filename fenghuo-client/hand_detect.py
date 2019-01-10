from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import time
import numpy as np

detection_graph, sess = detector_utils.load_inference_graph()
cap = cv2.VideoCapture(0)
time.sleep(10)
i = 0
while True:
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, image_np = cap.read()
    if not ret:
        continue
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
    max = np.max(scores)
    if max > 0.1:
        print("%d hands detected!" % i)
        i = i + 1
    cv2.imshow("frame", image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
