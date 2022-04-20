# https://learnopencv.com/edge-detection-using-opencv/
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_list_txt(values,name):
    with open(name+".txt", "w") as output:
        output.write(str(values))


def plot_measures(frames,y,z):
    """_summary_

    Args:
        frames (_type_): frames count
        y (_type_): _description_
        z (_type_): _description_
    """
    # Plotting both the curves simultaneously
    plt.plot(frames, y, color='r', label='cam')
    plt.plot(frames, z, color='g', label='kf')
    
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("frames")
    plt.ylabel("bhattacharyya dist")
    plt.title("cam and kf bhattacharyya dist")
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    plt.savefig('cam_and_kf_bhattacharyya_dist.png')
    # To load the display window
    plt.show()

def plot_list_count(x, y,Xname,Yname):

    # Plotting  the curve
    plt.plot(x, y, color='r', label=Xname+Yname+' hist')
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    plt.title(Xname+"And"+Yname)
    plt.savefig(Xname+"And"+Yname+'.png')

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
    
    # Display original image
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)
    

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=100) # Canny Edge Detection
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(contours)
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    #print(edges)
    
    return edges, contours





if __name__ == '__main__':
    pass