import matplotlib as plt
import cv2

# Count seeds
    # Threshold endosperm

def threshold(image, lower_thresh, upper_thresh):
  b, g, r = cv2.split(image)
  r_low, g_low, b_low = lower_thresh
  r_high, g_high, b_high  = upper_thresh

  # Define thresholds
  mask =  (r > r_low) &  (r < r_high) & (g > g_low) & (g < g_high) & (b > b_low) & (b < b_high)
          
  # Apply mask
  image = image * mask.reshape(mask.shape[0], mask.shape[1], 1)
  _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
  return result

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
def count_shapes(image, lower_thresh, upper_thresh):
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

    cv2.imshow("thresh", thresh_im)
    cv2.imshow("median filtered", median_im)
    cv2.imshow("shapes", shape_im)

    # Count shapes
    return len(contours)


# Calculate germination rate
# Count roots
# Count Coleoptiles


original = cv2.imread("maize.jpeg")
cv2.imshow("maize", original)

num_endosperm = count_shapes(original, (185,125,0), (255,255,100))
num_coleoptile = count_shapes(original, (125,155,10), (188,205,120))
print("Number of endosperm: ", num_endosperm)
print("Number of coleoptile: ", num_coleoptile)

cv2.waitKey()