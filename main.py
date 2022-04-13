import numpy as np
import cv2

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


cam = cv2.VideoCapture(0)
cv2.namedWindow("tracking")
# setup the mouse callback
cv2.setMouseCallback("tracking", on_mouse, 0)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
selected = False

while True:
    ret, frame = cam.read()
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
            roiHist = cv2.calcHist([hsv_crop], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

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
        # convert the current frame to the HSV color space
        # and perform mean shift
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv_img], [0], roiHist, [0, 180], 1)
        # apply cam shift to the back projection, convert the
        # points to a bounding box, and then draw them
        (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
        pts = np.int0(cv2.boxPoints(r))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()