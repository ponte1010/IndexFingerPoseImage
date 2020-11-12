from __future__ import division
import cv2
import os
import numpy as np

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def save_frame_camera_key(device_num, dir_path, basename, ext='jpg', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    print("カメラを起動しました")

    n = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), frame)
            img = cv2.imread('{}_{}.{}'.format(base_path, n, ext))
            frame_flip_holizontal = cv2.flip(img, 1)
            frameCopy = np.copy(frame_flip_holizontal)
            frameWidth = frame_flip_holizontal.shape[1]
            frameHeight = frame_flip_holizontal.shape[0]
            aspect_ratio = frameWidth/frameHeight

            threshold = 0.1

            h, w, c = frame_flip_holizontal.shape

            #t = time.time()
            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio*inHeight)*8)//8)
            inpBlob = cv2.dnn.blobFromImage(frame_flip_holizontal, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            #print("time taken by network : {:.3f}".format(time.time() - t))

            # Empty list to store the detected keypoints
            points = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold :
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                else :
                    points.append(None)

                if i == 8 :
                    x_index, y_index = point
                    print(int(x_index-(x_index-w/2)*2), y_index)

                cv2.imwrite('Output-Keypoints_{}.{}'.format(n, ext), frameCopy)

            n += 1
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)

save_frame_camera_key(0, 'Taken', 'camera_capture')
