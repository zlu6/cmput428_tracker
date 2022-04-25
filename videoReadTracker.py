import numpy as np
import cv2
from utils import *
import kalman as kl
import sys

selection_in_progress = False

current_mouse_position = np.ones(2, dtype=np.int32)




if __name__ == '__main__' :
    cap = cv2.VideoCapture('dataSet/dribbleBasketball00.mp4')
    
    cap.set(cv2.CAP_PROP_FPS, 120)
    cv2.namedWindow("tracking")
    
    # Exit if video not opened.
    if not cap.isOpened():
        print ("Could not open video")
        sys.exit()
    
    
    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
    
    # Uncomment the line below to select a different bounding box
    #(x,y,w,h)
    cv2_roi = cv2.selectROI(frame, False)
    (x,y,w,h) = cv2_roi
    print(cv2_roi)
    
    
    
    

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    selected = False
    count_frame = 0
    pred_occluded = None

    # recording list 
    bhattacharyya_dist_cam_list = []
    bhattacharyya_dist_kf_list = []
    frame_list = []
    edge_pixel_count_list =[]
    slope_list = []


    KF = kl.KalmanFilter()

    # we started from assume no occlusion
    occlusion_flag = False
    occlusion_start_frame = None


    while(True):
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            print ("video end")
            break
        frame_h, frame_w, _ = frame.shape

        if (len(cv2_roi) > 1) :
            template = frame[y:y+h,x:x+w].copy()
            #cv2.imshow("template_image", template)
            
            template_img_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            edge_weight_template = getGradientMagnitude(template_img_gray)
            if h > 0 and w > 0:
                selected = True
                hsv_crop = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
                #hsv_crop = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                #mask = None
                mask = cv2.inRange(hsv_crop, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                
                roiHist = cv2.calcHist([hsv_crop], [0], mask, [180], [0, 180])
                # roiHist = cv2.calcHist([hsv_crop], [0], mask, [255], [0, 255])
                roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                # cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = cv2_roi

        if selected:
            # convert the current frame to the HSV color space and perform mean shift
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            backProj = cv2.calcBackProject([hsv_img], [0], roiHist, [0, 180], 1)
            #backProj = cv2.calcBackProject([hsv_img], [0], roiHist, [0, 255], 1)
            
            # apply cam shift to the back projection, convert the points to a bounding box, and then draw them

            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            count_frame += 1
            
            pts = np.int0(cv2.boxPoints(r))
            
            # print(pts)
            # print("--------------------------------------")
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            tracked_img = frame[roiBox[1]:roiBox[1] + roiBox[3], roiBox[0]:roiBox[0] + roiBox[2]]

            


            
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


            if not occlusion_flag:

                #update the measurement Matrix in kalman filter
                # kf.correct(get_center_points(pts))
                # prediction = kf.predict()
                # last_nonOcclusionR = get_center_points(pts)
                # bbox_x_coord, bbox_y_coord, bbox_width, bbox_height = roiBox
                prediction = KF.predict()
                KF.correct(get_center_points(pts))

                last_nonOcclusionR = get_center_points(pts)
                bbox_x_coord, bbox_y_coord, bbox_width, bbox_height = roiBox

            else:


                pred_occluded = KF.predict()
                KF.correct(pred_occluded)
                
                
                #prediction = KF.predict()
                # KF.correct(get_center_points(pts))
                # print(pred_occluded)
                # print("**************************************")
                
                # cv2.circle(frame, (int(pred_occluded[0]), int(pred_occluded[1])), 5, (255, 0, 0), 3)
                # cv2.imshow("tracked circle", frame)
                # cv2.waitKey(500)

            # prediction = kf.predict()

            # print("kf pred x: ", prediction[0] - (0.5*bbox_width), "kf pred y: ", prediction[1]- (0.5*bbox_height))
            #if occlusion occurs , we use the old estimate to get new KF prediction 
            if (count_frame>5) :
                
                if not occlusion_flag:
                    corr_x_coord, corr_y_coord, corr_width, corr_height = int(prediction[0] - (0.5*bbox_width)), \
                                                                            int(prediction[1] - (0.5*bbox_height)), \
                                                                            int(prediction[0] + (0.5*bbox_width)), \
                                                                            int(prediction[1] + (0.5*bbox_height))

                    # if not occlusion_flag:
                    cv2.rectangle(frame, (corr_x_coord, corr_y_coord), (corr_width, corr_height), (0, 0, 255), 2)
                else:

                    corr_x_coord = int(pred_occluded[0] - 0.5 * w)
                    corr_y_coord = int(pred_occluded[1]- 0.5 * h)
                    corr_width = int(corr_x_coord + 0.5 * w)
                    corr_height = int(corr_y_coord + 0.5 * h)
                    # print(corr_x_coord)
                    # print(corr_y_coord)
                    # print(corr_width)
                    # print(corr_height)
                    # print("------------------------------------------------")
                    cv2.rectangle(frame, (corr_x_coord, corr_y_coord), (corr_width, corr_height), (255, 255, 255), 2)

                    
                    
                    
                    # print(corr_x_coord)
                    # print(corr_y_coord)
                    # print(corr_width)
                    # print(corr_height)
                    # print("------------------------------------------------")

                    # cv2.circle(frame, (corr_x_coord, corr_y_coord), 2, (0, 0, 255), -1)



                # cv2.imshow("kf_corr_img", kf_corr_img)
                # cv2.waitKey(0)
                if corr_x_coord in range(0, frame_w) and corr_y_coord in range(0, frame_h) and corr_width in range(0, frame_w) and corr_height in range(0, frame_h):
                    kf_corr_img = frame[corr_y_coord: corr_height, corr_x_coord: corr_width]
                    hsv_corr_img = cv2.cvtColor(kf_corr_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_corr_img, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                    # mask = None
                    tracked_roiHist_kf = cv2.calcHist([hsv_corr_img], [0], mask, [180], [0, 180])
                    tracked_roiHist_kf = cv2.normalize(tracked_roiHist_kf, tracked_roiHist_kf, 0, 255, cv2.NORM_MINMAX)
                    bhattacharyya_dist_kf = cv2.compareHist(roiHist, tracked_roiHist_kf, method=cv2.HISTCMP_BHATTACHARYYA)

                cv2.putText(frame, str(bhattacharyya_dist_kf), (50, 90), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            else:
                bhattacharyya_dist_kf = 0
                corr_x_coord, corr_y_coord, corr_width, corr_height = 0, 0, 0, 0
            # print(prediction)

            bhattacharyya_dist_cam_list.append(bhattacharyya_dist_cam)
            bhattacharyya_dist_kf_list.append(bhattacharyya_dist_kf)
            frame_list.append(count_frame)

            rangeVal = 5
            if count_frame > 5 :
            
                slope = (bhattacharyya_dist_cam - bhattacharyya_dist_cam_list[count_frame - rangeVal] )/ rangeVal
            else:
                slope = 0
            slope_list.append(slope)

            if bhattacharyya_dist_cam >= bhattacharyya_dist_kf:
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                tracked_img_gray = cv2.cvtColor(tracked_img, cv2.COLOR_BGR2GRAY)
                edge_weight = getGradientMagnitude(tracked_img_gray)
            elif bhattacharyya_dist_cam < bhattacharyya_dist_kf:
                cv2.rectangle(frame, (corr_x_coord, corr_y_coord), (corr_width, corr_height), (0, 255, 255), 2)
                kalman_img_gray = cv2.cvtColor(kf_corr_img, cv2.COLOR_BGR2GRAY)
                edge_weight = getGradientMagnitude(kalman_img_gray)

            # if edge_weight < edge_weight_template / 2:
            #     # print("occlusion occurs")
            #     cv2.putText(frame, "occlusion occurs", (50, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # if edges and Bhattacharyya value changes a lot, means occlusion occurs
            if (edge_weight < edge_weight_template / 1) and (slope < -0.02):
                print("occlusion occurs at frame ", count_frame, "and occlusion start at ", occlusion_start_frame)
                cv2.putText(frame, "occlusion occurs", (50, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                
                # if not occlusion yet
                if not occlusion_flag :
                    print(" occlusion started")
                    occlusion_flag = True
                    occlusion_start_frame = count_frame
                #if this a consquence occlusion
                else :  
                    # if occlusion lasting longer than 4/30 second, 4 frames;
                    # we need to change to non-occlusion case, in order to make sure KF tracker wont give wrong result
                    if (count_frame  - occlusion_start_frame) > 10:
                        print("occlusion too long" )
                        occlusion_flag = False
                        occlusion_start_frame = None
                    else :
                        occlusion_flag = True
                
                # bbox_x_coord, bbox_y_coord, bbox_width, bbox_height = roiBox
                pred_occluded = np.array([(corr_x_coord+corr_width) // 2, (corr_y_coord + corr_height) // 2], dtype=np.float32)
            else:
                occlusion_flag = False
                occlusion_start_frame = None


                
        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plot_measures(frame_list,bhattacharyya_dist_cam_list, bhattacharyya_dist_kf_list)
    plot_list_count(frame_list,slope_list, "Frame", "Slope")
    save_list_txt(bhattacharyya_dist_cam_list, "bhattacharyya_dist_cam_list")
    save_list_txt(bhattacharyya_dist_kf_list, "bhattacharyya_dist_kf")
    save_list_txt(frame_list, "frame_list")
    save_list_txt(edge_pixel_count_list, "edge_pixel_count")


    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()










