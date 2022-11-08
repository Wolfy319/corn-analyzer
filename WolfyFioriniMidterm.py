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
    # Find shape contours
    contours, _ = cv2.findContours(shape_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find approximate center of each contour
    centers = []
    for contour in contours:
        moments = cv2.moments(contour)
        centerX = int(moments["m10"]/moments["m00"])
        centerY = int(moments["m01"]/moments["m00"])
        centers.append([centerX, centerY])
    print(centers)
    cv2.imshow("thresh", thresh_im)
    cv2.imshow("median filtered", median_im)
    cv2.imshow("shapes", shape_im)
    # Count shapes
    return len(contours), centers

# Find roots
    # Grab touching endosperm
def get_roots(image, endosperm_thresh, coleoptile_thresh):
    edges = cv2.Canny(image, 255,255,None,3)
    cv2.imshow("e", edges)
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
cv2.imshow("maize", original)

# num_endosperm, endosperm_centers = get_endosperm_or_coleoptile(original, (185,125,0), (255,255,100))
# num_coleoptile, coleoptile_centers = count_shapes(original, (125,155,10), (188,205,120))
# print("Number of endosperm: ", num_endosperm)
# print("Number of coleoptile: ", num_coleoptile)
get_roots(original, [(185,125,0), (255,255,100)],[(125,155,10), (188,205,120)])
cv2.waitKey()