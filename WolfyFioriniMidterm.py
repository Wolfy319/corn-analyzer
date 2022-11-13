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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    dil = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    closed_im = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("close", closed_im)
    # opened_im = cv2.morphologyEx(closed_im, cv2.MORPH_OPEN, kernel)

    # return opened_im
    # return closed_im
    return closed_im

def detect_shapes(image):
    # Find shape contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find approximate center of each contour
    centers = []
    for contour in contours:
        moments = cv2.moments(contour)
        centerX = int(moments["m10"]/moments["m00"])
        centerY = int(moments["m01"]/moments["m00"])
        centers.append([centerX, centerY])
    return len(contours), centers

def remove_shapes(image, shape_im):
    shape_im = cv2.morphologyEx(shape_im, cv2.MORPH_DILATE, np.ones((5,5)), iterations=2)
    mask = np.where(shape_im > 0)

    for i in range(len(mask[0])):
        image[mask[0],mask[1]] = 0

# Count endosperm, coleoptile, and roots
def get_endosperm_or_coleoptile(image, lower_thresh, upper_thresh):
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Extract desired objects
    thresh_im = cv2.inRange(image, lower_thresh, upper_thresh)
    # Reduce noise
    median_im = cv2.medianBlur(thresh_im, 5)
    # Isolate regions
    shape_im = isolate_shapes(median_im)
    
    num_shapes, shape_centers = detect_shapes(shape_im)

    cv2.imshow("thresh", thresh_im)
    cv2.imshow("median filtered", median_im)
    cv2.imshow("shapes", shape_im)
    # Count shapes
    return num_shapes, shape_centers, shape_im


# Find root
    # Grab touching endosperm
def get_roots(image, endosperm_shapes):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rem = remove_shapes(image, endosperm_shapes)
    endosperm_border = np.min(np.where(endosperm_shapes > 0)[0])
    # soil_border = np.max(np.where(endosperm_shapes > 0)[0])
    # print(soil_border)
    # pruned_im = cv2.line(image, (0, soil_border), (image.shape[0],soil_border), color=1,thickness=3)
    pruned_im = image[endosperm_border:,:]
    cv2.imshow("removed", rem)
    # edges = cv2.Canny(pruned_im, 255,255,None,3)
    _, thresh = cv2.threshold(pruned_im, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow("e", thresh)
    # filtered = cv2.medianBlur(edges, 3)
    # cv2.imshow("e", edges)
    # cv2.imshow("e2", filtered)


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

num_endosperm, endosperm_centers, endosperm_shapes = get_endosperm_or_coleoptile(roi, (185,125,0), (255,255,100))
num_coleoptile, coleoptile_centers, coleoptile_shapes = get_endosperm_or_coleoptile(roi, (125,155,10), (188,205,120))
# print("Number of endosperm: ", num_endosperm)
# print("Number of coleoptile: ", num_coleoptile)
get_roots(roi, endosperm_shapes)
cv2.waitKey()