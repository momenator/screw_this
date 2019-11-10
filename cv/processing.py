from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import time

presentationMode = True

# debugging
def captureImage():
    webcam = cv2.VideoCapture(1)
    check, frame = webcam.read()
    cv2.imshow("Capturing", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(filename='./images/latest_image.jpg', img=frame)


def show_n_wait(title_name, image_input):
	cv2.imshow(title_name, image_input)
	cv2.waitKey(0)

def present(name, imageA):
    if(presentationMode==True):
        cv2.imwrite(filename='./images/present/' + name + '.jpg', img=imageA)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def createMask(original_image, outer_rect):
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    print(outer_rect)
    cv2.rectangle(mask, (outer_rect[0]), (outer_rect[3]), 255, -1)
    print(cv2.mean(original_image, mask))


def resize_w_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def remove_shadows(img):
    # dilate image
    # dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8)) 
    # bg_img = cv2.medianBlur(dilated_img, 21)
    # diff_img = 255 - cv2.absdiff(img, bg_img)
    # norm_img = diff_img.copy() # Needed for 3.x compatibility
    # norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # _, thr_img = cv2.threshold(norm_img, 255, 0, cv2.THRESH_TRUNC)
    # norm_img = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    shadow_threshold = 30
    return cv2.threshold(img, shadow_threshold, 255, cv2.THRESH_TOZERO)
    # return norm_img


# img should be RGB
def get_avg_pixel_val(img, pts):
    pts = np.where(pts<0, 0, pts).astype("int")
    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask.astype(np.bool)

    out = np.zeros_like(img)
    out[mask] = img[mask]
    return np.sum(out) / np.count_nonzero(out)


def classify_img(grey_val):
    if (grey_val < 95):
        return 0
    elif (grey_val < 155):
        return 1
    return -1


def check_box_exist(box, boxes):
    box_exists = False
    for cur_box in boxes:
        if np.abs(np.sum(cur_box - box)) < 30:
            box_exists = True
            break
    return box_exists


def get_measurements(image_path, real_width, is_display=False):
    image = cv2.imread(image_path)
    present('1', image)

    image = resize_w_aspect_ratio(image, 800)

    # Resize image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    present('2', gray)

    # remove shadows
    # image = remove_shadows(gray)
    # show_n_wait("shadows", gray)

    # reduce noise by blurring!
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    present('3', gray)
    # show_n_wait("gaussian", gray)

    # use canny edge detection to threshold the image
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    # METHOD 1: canny edge detection uses adaptive threshold by default
    edges = cv2.Canny(gray, 50, 100)
    present('4', edges)
    edges = cv2.dilate(edges, None, iterations=1)
    present('5', edges)
    edges = cv2.erode(edges, None, iterations=1)
    present('6', edges)

    # find contours in the edge map
    cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    measurements = []
    boxes = []

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 250:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)

        # check if cv2 contains overlap
        # what if they do?
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # print("box", createMask(orig,box))

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)

        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        #     (255, 0, 255), 2)
        # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        #     (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / real_width

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        is_anomaly = False
        if dimA * dimB > 1500 or dimA * dimB < 200:
            is_anomaly = True
            continue
        
        # check box difs
        box_exists = check_box_exist(box, boxes)
        if box_exists:
            continue
    
        boxes.append(box.astype("int"))
        
        # draw the object sizes on the image
        cv2.putText(orig, "{:.3f}mm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 2)
        cv2.putText(orig, "{:.3f}mm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 2)

        present('7', orig)

        if is_display:
            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey(0)
        
        avg = get_avg_pixel_val(gray, np.array([tl, tr, br, bl]))
        img_type = classify_img(avg)
        measurements.append((img_type, round(dimA,2), round(dimB,2)))

    if is_display:
        cv2.drawContours(orig, boxes, -1, (0, 255, 0), 2)    
        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)

    # return all the measurements here!
    return measurements
