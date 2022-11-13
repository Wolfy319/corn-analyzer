import matplotlib as plt
import cv2
import numpy as np

# Count seeds
    # Threshold endosperm

# def threshold(image, lower_thresh, upper_thresh):
#   b, g, r = cv2.split(image)
#   r_low, g_low, b_low = lower_thresh
#   r_high, g_high, b_high  = upper_thresh

#   # Define thresholds
#   mask =  (r > r_low) &  (r < r_high) & (g > g_low) & (g < g_high) & (b > b_low) & (b < b_high)
          
#   # Apply mask
#   image = image * mask.reshape(mask.shape[0], mask.shape[1], 1)
#   _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
#   return result

def isolate_shapes(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

    closed_im = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilate = cv2.morphologyEx(closed_im, cv2.MORPH_DILATE, kernel)
    median = cv2.medianBlur(dilate, 11)
    # median = cv2.medianBlur(median, 7)
    open = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    closed_im2 = cv2.morphologyEx(open, cv2.MORPH_CLOSE, big_kernel, iterations=3)
    cv2.imshow("close", closed_im)
    cv2.imshow("dil", dilate)
    cv2.imshow("median", median)
    cv2.imshow("close2", closed_im2)
    cv2.imshow("open", open)

    return closed_im2

def detect_shapes(image):
    # Find shape contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find approximate center of each contour
    centers = []
    final_contours = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] < 100:
            continue
        final_contours.append(contour)
        centerX = int(moments["m10"]/moments["m00"])
        centerY = int(moments["m01"]/moments["m00"])
        centers.append([centerX, centerY])
    return len(centers), centers, final_contours

def remove_shapes(image, shape_im):
    shape_im = cv2.morphologyEx(shape_im, cv2.MORPH_DILATE, np.ones((10,10)))
    mask = np.where(shape_im > 0)

    for i in range(len(mask[0])):
        image[mask[0],mask[1]] = 0
    return image

# Count endosperm, coleoptile, and roots
def get_endosperm_or_coleoptile(image, lower_thresh, upper_thresh):
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Extract desired objects
    thresh_im = cv2.inRange(image, lower_thresh, upper_thresh)
    # Reduce noise
    # median_im = cv2.medianBlur(thresh_im, 5)
    # Isolate regions
    shape_im = isolate_shapes(thresh_im)
    
    num_shapes, shape_centers, contours = detect_shapes(shape_im)

    cv2.imshow("thresh", thresh_im)
    # cv2.imshow("median filtered", median_im)
    cv2.imshow("shapes", shape_im)
    # Count shapes
    return num_shapes, shape_centers, shape_im, contours


# Find root
    # Grab touching endosperm
def get_roots(image, endosperm_shapes, endosperm_contours):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    opened = cv2.morphologyEx(endosperm_shapes, cv2.MORPH_OPEN, np.ones((10,10)))
    cv2.imshow("root open", opened)
    endosperm_top = np.min(np.where(opened > 0)[0]) + 10
    endosperm_bottom = np.max(np.where(opened > 0)[0])

    no_endosperm = remove_shapes(image, endosperm_shapes)
    # cv2.imshow("removed", no_endosperm)

    # soil_border = np.max(np.where(endosperm_shapes > 0)[0])
    # print(soil_border)
    # pruned_im = cv2.line(image, (0, soil_border), (image.shape[0],soil_border), color=1,thickness=3)

    pruned_im = image[endosperm_bottom:,:]
    av_kernel = (1/3)**2 * np.ones((3,3))
    blur = cv2.filter2D(pruned_im, -1, av_kernel)
    edges = cv2.Canny(blur, 0,255,None,3)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5)))
    _, thresh_im = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
    
    # edges = cv2.Canny(median, 0,255)
    cv2.imshow("e", thresh_im)

    filtered = cv2.medianBlur(thresh_im, 5)
    # cv2.imshow("e", edges)
    cv2.imshow("e2", filtered)
    cv2.imshow("e3", closed_edges)


# Group endosperm with it's corresponding coleoptile and roots
def group_objects(seed_centers, sprout_centers, root_tips):
    groupings = []
    min_distance = 25
    for seed in seed_centers:
        seedX = seed[0]
        seed_data = []
        grouped_roots = []
        seed_data.append(seed)
        for sprout in sprout_centers:
            sproutX = sprout[0]
            if np.abs(seedX - sproutX) <= min_distance:
                seed_data.append(sprout)
        for root in root_tips: 
            rootX = root[0]
            if np.abs(seedX - rootX) <= min_distance:
                grouped_roots.append(root)
        seed_data.append(grouped_roots)
        groupings.append(seed_data)

def germination_rate(seed_data):
    num_seeds = len(seed_data)
    num_successful = 0
    for seed in seed_data:
        has_coleoptile = seed[0]
        num_roots = seed[1]
        if has_coleoptile and num_roots > 0:
            num_successful += 1
    germination_rate = (num_successful / num_seeds) * 100
    print("%.2f of seeds germinated", germination_rate)
    return
    
# Calculate germination rate
# Count roots


original = cv2.imread("maize.jpeg")
h,w,_ = original.shape
roi = original[100:h-50, 50:w-50, :]
cv2.imshow("maize", original)
cv2.imshow("roi", roi)

num_endo, endo_centers, endo_shapes, endo_contours = get_endosperm_or_coleoptile(roi, (185,125,0), (255,255,100))
# num_col, col_centers, _, col_contours = get_endosperm_or_coleoptile(roi, (125,155,10), (188,205,120))
print("Number of endosperm: ", num_endo)
# print("Number of coleoptile: ", num_coleoptile)
get_roots(roi, endo_shapes, endo_contours)
cv2.waitKey()