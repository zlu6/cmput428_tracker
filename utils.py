# https://learnopencv.com/edge-detection-using-opencv/
import cv2



def get_edge_features(img):
    """_summary_
        get edge histograms
        we use 4 steps Canny Edge Detection algorithm to get histograms
        1.Noise Reduction
        2.Calculating Intensity Gradient of the Image
        3.Suppression of False Edges
        4.Hysteresis Thresholding

        Notice that the original paper said: only edges with magnitudes above q threshhold were consider in the edge feature
    Args:
        img (_type_): selected frame
    """
    
    # Read the original image
    # img = cv2.imread('test.jpg') 
    # Display original image
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 



   
    
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=100) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass