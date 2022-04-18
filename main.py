import numpy as np
import cv2
from utils import *

selection_in_progress = False
boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)


def on_mouse(event, x, y, flags, params):

    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []
        # print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:
        # print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)

def get_center_points(pts):
    avg_x_coord = np.average(pts[:, 0]).astype(np.float32)
    avg_y_coord = np.average(pts[:, 1]).astype(np.float32)
    return np.array([avg_x_coord, avg_y_coord], dtype=np.float32)


cam = cv2.VideoCapture(0)
cv2.namedWindow("tracking")
# setup the mouse callback
cv2.setMouseCallback("tracking", on_mouse, 0)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
selected = False
count_frame = 0

bhattacharyya_dist_cam_list = []
bhattacharyya_dist_kf_list = []
frame_list = []

kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                      np.float32) * 0.03

while True:
    ret, frame = cam.read()
    frame_h, frame_w, _ = frame.shape

    if not ret:
        print("failed to grab frame")
        break

    if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (
            boxes[0][0] < boxes[1][0]):
        template = frame[boxes[0][1]:boxes[1][1],
               boxes[0][0]:boxes[1][0]].copy()
        cv2.imshow("template_image", template)
        height, width, channel = template.shape
        if height > 0 and width > 0:
            selected = True
            hsv_crop = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            mask = None
            # mask = cv2.inRange(hsv_crop, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roiHist = cv2.calcHist([hsv_crop], [0], mask, [180], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            # cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (
                boxes[0][0],
                boxes[0][1],
                boxes[1][0] -
                boxes[0][0],
                boxes[1][1] -
                boxes[0][1])
        boxes = []

    if selection_in_progress:
        top_left = (boxes[0][0], boxes[0][1])
        bottom_right = (
            current_mouse_position[0],
            current_mouse_position[1])
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    if selected:
        # convert the current frame to the HSV color space and perform mean shift
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv_img], [0], roiHist, [0, 180], 1)
        # apply cam shift to the back projection, convert the points to a bounding box, and then draw them
        (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
        count_frame += 1
        bbox_x_coord, bbox_y_coord, bbox_width, bbox_height = roiBox
        pts = np.int0(cv2.boxPoints(r))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        # tracked_img = frame[roiBox[0]:roiBox[0] + roiBox[2], roiBox[1]: roiBox[1] + roiBox[3]]
        tracked_img = frame[roiBox[1]:roiBox[1] + roiBox[3], roiBox[0]:roiBox[0] + roiBox[2]]
        
        edges = get_edge_features(tracked_img)

        # cv2.imshow("tracked_img",tracked_img)
        # cv2.waitKey(0)
        hsv_tracked_img = cv2.cvtColor(tracked_img, cv2.COLOR_BGR2HSV)

        # mask = cv2.inRange(hsv_tracked_img, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        
        #to avoid false values due to low light, low light values are discarded using cv2.inRange() 
        mask = cv2.inRange(hsv_tracked_img, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        tracked_roiHist_cam = cv2.calcHist([hsv_tracked_img], [0], mask, [180], [0, 180])
        tracked_roiHist_cam = cv2.normalize(tracked_roiHist_cam, tracked_roiHist_cam, 0, 255, cv2.NORM_MINMAX)
        # cv2.normalize(tracked_roiHist, tracked_roiHist, 0, 255, cv2.NORM_MINMAX)
        bhattacharyya_dist_cam = cv2.compareHist(roiHist, tracked_roiHist_cam, method=cv2.HISTCMP_BHATTACHARYYA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(bhattacharyya_dist_cam), (50, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        kf.correct(get_center_points(pts))
        prediction = kf.predict()
        if count_frame > 30:
            corr_x_coord, corr_y_coord, corr_width, corr_height = int(prediction[0] - (0.5*bbox_width)), \
                                                                  int(prediction[1] - (0.5*bbox_height)), \
                                                                  int(prediction[0] + (0.5*bbox_width)), \
                                                                  int(prediction[1] + (0.5*bbox_height))
            cv2.rectangle(frame, (corr_x_coord, corr_y_coord), (corr_width, corr_height), (0, 0, 255), 2)


            # cv2.imshow("kf_corr_img", kf_corr_img)
            # cv2.waitKey(0)
            if corr_x_coord in range(0, frame_w) and corr_y_coord in range(0, frame_h) and corr_width in range(0, frame_w) and corr_height in range(0, frame_h):
                kf_corr_img = frame[corr_y_coord: corr_height, corr_x_coord: corr_width]
                hsv_corr_img = cv2.cvtColor(kf_corr_img, cv2.COLOR_BGR2HSV)
                # mask = cv2.inRange(hsv_corr_img, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                mask = None
                tracked_roiHist_kf = cv2.calcHist([hsv_corr_img], [0], mask, [180], [0, 180])
                tracked_roiHist_kf = cv2.normalize(tracked_roiHist_kf, tracked_roiHist_kf, 0, 255, cv2.NORM_MINMAX)
                bhattacharyya_dist_kf = cv2.compareHist(roiHist, tracked_roiHist_kf, method=cv2.HISTCMP_BHATTACHARYYA)
            else:
                bhattacharyya_dist_kf = 0
            cv2.putText(frame, str(bhattacharyya_dist_kf), (50, 90), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        else:
            bhattacharyya_dist_kf = 0
            corr_x_coord, corr_y_coord, corr_width, corr_height = 0, 0, 0, 0
        # print(prediction)

        bhattacharyya_dist_cam_list.append(bhattacharyya_dist_cam)
        bhattacharyya_dist_kf_list.append(bhattacharyya_dist_kf)
        frame_list.append(count_frame)
        
        #if full occulsion (bhattacha c. and edges decrease significant, use prediction from kalman)
        if bhattacharyya_dist_cam >= bhattacharyya_dist_kf:
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        elif bhattacharyya_dist_cam < bhattacharyya_dist_kf:
            cv2.rectangle(frame, (corr_x_coord, corr_y_coord), (corr_width, corr_height), (0, 255, 255), 2)


    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plot_measures(frame_list,bhattacharyya_dist_cam_list, bhattacharyya_dist_kf_list)
save_list_txt(bhattacharyya_dist_cam_list, "bhattacharyya_dist_cam_list")
save_list_txt(bhattacharyya_dist_kf_list, "bhattacharyya_dist_kf")
save_list_txt(frame_list, "frame_list")


       
cam.release()
cv2.destroyAllWindows()