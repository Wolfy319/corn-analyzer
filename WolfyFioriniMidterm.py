import matplotlib as plt
import cv2
import numpy as np

def isolate_shapes(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

    # Perform morph ops to isolate shape structures
    closed_im = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilate = cv2.morphologyEx(closed_im, cv2.MORPH_DILATE, kernel)
    median = cv2.medianBlur(dilate, 11)
    open = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    closed_im2 = cv2.morphologyEx(open, cv2.MORPH_CLOSE, big_kernel, iterations=3)

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

# Find tips of roots that go under the soil
def get_roots(image, endosperm_shapes):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find bottom of seeds, this will be soil height
    opened = cv2.morphologyEx(endosperm_shapes, cv2.MORPH_OPEN, np.ones((10,10)))
    endosperm_bottom = np.max(np.where(opened > 0)[0])
    # Remove everything above soil
    pruned_im = image[endosperm_bottom:,:]
    # Blur image to remove noise
    av_kernel = (1/3)**2 * np.ones((3,3))
    blur = cv2.filter2D(pruned_im, -1, av_kernel)
    # Find edges of roots
    edges = cv2.Canny(blur, 0,255,None,3)
    # For some reason roots are detected better with this operation
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5)))[5:,:]
    edges2 = cv2.Canny(closed_edges, 0,255,None,3)
    # Find tips of roots
    points = np.where(edges2[0,:] > 0)
    points = np.sort(points)
    sum = 0
    root_coords = []
    for point in points[0]:
        sum+=1
        # Add root for every 2 edges
        if sum % 2 == 0:
            root_y = endosperm_bottom
            root_x = point
            root_coords.append([root_x,root_y])

    return root_coords

# Group endosperm with it's corresponding coleoptile and roots
def group_objects(seed_centers, sprout_centers, root_tips):
    groupings = []
    # Set the minimum distance away from 
    #   seed to be considered part of seed
    min_distance = 50
    for seed in seed_centers:
        seedX = seed[0]
        seed_data = []
        grouped_roots = []
        sprout_data = []
        # Add endosperm location
        seed_data.append(seed)
        # Group coleoptile with endosperm if close by
        for sprout in sprout_centers:
            sproutX = sprout[0]
            if np.abs(seedX - sproutX) <= min_distance:
                sprout_data.append(sprout)
        # Group root tip with seed if close by
        for root in root_tips: 
            rootX = root[0]
            if np.abs(seedX - rootX) <= min_distance:
                grouped_roots.append(root)
        seed_data.append(sprout_data)
        seed_data.append(grouped_roots)
        groupings.append(seed_data)
    return groupings

# Find percent of seeds that have germinated
def germination_rate(seed_data):
    num_seeds = len(seed_data)
    num_germinated = 0
    for seed in seed_data:
        sprout_data = seed[1]
        root_data = seed[2]

        # Seed is germinated if it has a coleoptile 
        # and at least one root in the soil
        if len(sprout_data) and len(root_data) > 0:
            num_germinated += 1
    germination_rate = (num_germinated / num_seeds) * 100
    
    return "Germination rate: {percent}%".format(percent = germination_rate)
    
# Mark endosperm location, coleoptile location, 
# root location, and germination rate on image
def draw_data(image, seed_data, germ_rate, x_offset, y_offset):
    for i, seed in enumerate(seed_data):
        endo_X, endo_Y = seed[0][0], seed[0][1]

        roots = seed[2]

        # Draw with offset to reverse ROI cropping
        cv2.rectangle(image, (endo_X - 25 + x_offset, endo_Y - 25 + y_offset), (endo_X + 25 + x_offset, endo_Y + 25 + y_offset), (0,255,0), thickness = 3)
        cv2.line(image, (seed[1][0][0] + x_offset,seed[1][0][2] + y_offset), (seed[1][0][0] + x_offset, endo_Y - 25 + y_offset), (0,0,255), thickness = 3)        
        for root in roots:
            root_x = root[0]
            root_y = root[1] + 50
            cv2.circle(image, (root_x + x_offset,root_y + x_offset), 3, (255,255,255), 3)
        
        # Write text data
        cv2.putText(image, "{number} roots".format(number = len(roots)), (endo_X-50 + x_offset, endo_Y + 50 + y_offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,255))
        cv2.putText(image, "coleoptile", (endo_X-50 + x_offset, endo_Y + 60 + y_offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255))
        cv2.putText(image, "length:", (endo_X-50 + x_offset, endo_Y + 70 + y_offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255))
        cv2.putText(image, "{number} px".format(number = (endo_Y - seed[1][0][2])), (endo_X-50 + x_offset, endo_Y + 80 + y_offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255))

    cv2.putText(image, germ_rate, (0, image.shape[1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))
    cv2.imshow("seed", image)


original = cv2.imread("maize.jpeg")
h,w,_ = original.shape
# Crop out container edges
roi = original[100:h-50, 50:w-50, :]

# Find endosperm data
num_endo, endo_centers, endo_shapes = get_endosperm_or_coleoptile(roi, (185,125,0), (255,255,100))
# Find coleoptile data
num_col, col_centers, col_shapes = get_endosperm_or_coleoptile(roi, (125,155,10), (188,205,120))
# Find root data
roots = get_roots(roi, endo_shapes)
# Group endosperm with their roots/coleoptiles
seed_data = group_objects(endo_centers, col_centers, roots)
# Calculate germination rate
germ_string = germination_rate(seed_data)
# Show data
draw_data(original, seed_data, germ_string, 50, 100)
cv2.waitKey()