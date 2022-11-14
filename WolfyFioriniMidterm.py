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
    centers = []
    for contour in contours:
        # Find center of each contour
        moments = cv2.moments(contour)
        if moments["m00"] < 100:
            continue
        centerX = int(moments["m10"]/moments["m00"])
        centerY = int(moments["m01"]/moments["m00"])

        # Find contour's highest point, used for coleoptile
        max_height = np.min(contour[:][1])
        centers.append([centerX, centerY, max_height])

    return len(centers), centers

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
    # Isolate regions
    shape_im = isolate_shapes(thresh_im)
    # Get shape data
    num_shapes, shape_centers = detect_shapes(shape_im)
    
    return num_shapes, shape_centers, shape_im

# Find root
    # Grab touching endosperm
def get_roots(image, endosperm_shapes):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    opened = cv2.morphologyEx(endosperm_shapes, cv2.MORPH_OPEN, np.ones((10,10)))
    # cv2.imshow("root open", opened)
    endosperm_bottom = np.max(np.where(opened > 0)[0])


    pruned_im = image[endosperm_bottom:,:]
    av_kernel = (1/3)**2 * np.ones((3,3))
    blur = cv2.filter2D(pruned_im, -1, av_kernel)
    edges = cv2.Canny(blur, 0,255,None,3)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5)))[5:,:]
    edges2 = cv2.Canny(closed_edges, 0,255,None,3)
    points = np.where(edges2[0,:] > 0)
    points = np.sort(points)
    sum = 0
    root_coords = []
    for point in points[0]:
        sum+=1
        if sum % 2 == 0:
            root_y = endosperm_bottom
            root_x = point
            root_coords.append([root_x,root_y])

    return root_coords

# Group endosperm with it's corresponding coleoptile and roots
def group_objects(seed_centers, sprout_centers, root_tips):
    groupings = []
    min_distance = 50
    for seed in seed_centers:
        seedX = seed[0]
        seed_data = []
        grouped_roots = []
        sprout_data = []
        seed_data.append(seed)
        for sprout in sprout_centers:
            sproutX = sprout[0]
            if np.abs(seedX - sproutX) <= min_distance:
                sprout_data.append(sprout)
        for root in root_tips: 
            rootX = root[0]
            if np.abs(seedX - rootX) <= min_distance:
                grouped_roots.append(root)
        seed_data.append(sprout_data)
        seed_data.append(grouped_roots)
        groupings.append(seed_data)
    return groupings

def germination_rate(seed_data):
    num_seeds = len(seed_data)
    num_successful = 0
    for seed in seed_data:
        sprout_data = seed[1]
        root_data = seed[2]
        if len(sprout_data) and len(root_data) > 0:
            num_successful += 1
    germination_rate = (num_successful / num_seeds) * 100
    print("{percent}% of seeds germinated".format(percent = germination_rate))
    return
    
def draw_data(image, seed_data):
    for i, seed in enumerate(seed_data):
        endo_X, endo_Y = seed[0][0], seed[0][1]
        roots = seed[2]
        print(roots)
        cv2.rectangle(image, (endo_X - 25, endo_Y - 25), (endo_X + 25, endo_Y + 25), (0,255,0), thickness = 3)
        cv2.line(image, (seed[1][0][0],seed[1][0][2]), (seed[1][0][0], endo_Y - 25), (0,0,255), thickness = 3)        
        for root in roots:
            cv2.circle(image, root, 3, (255,255,255), 3)

    cv2.imshow("seed", image)


original = cv2.imread("maize.jpeg")
h,w,_ = original.shape
roi = original[100:h-50, 50:w-50, :]
cv2.imshow("maize", original)
cv2.imshow("roi", roi)

num_endo, endo_centers, endo_shapes = get_endosperm_or_coleoptile(roi, (185,125,0), (255,255,100))
num_col, col_centers, col_shapes = get_endosperm_or_coleoptile(roi, (125,155,10), (188,205,120))
print("Number of endosperm: ", num_endo)
# print("Number of coleoptile: ", num_coleoptile)
roots = get_roots(roi, endo_shapes)
seed_data = group_objects(endo_centers, col_centers, roots)
cv2.imshow("roots", roi)
germination_rate(seed_data)
draw_data(roi, seed_data)
cv2.waitKey()