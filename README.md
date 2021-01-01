# YOLO_social_distance_detector
 I am well aware that this project is not perfect so these are a few ideas how this application be improved :  Using a faster model in order to perform real-time social distancing analysis. Use a model more robust to occlusions. 


# Steps

**Step 1: Find the number of people in the frame/Image.**


**Step 2: Create Bounding Box over the people identified using YOLO.**


**Step 3: A width threshold is set for object among which the distance is measured i.e. the width of the people. I am setting width as 27inch or 0.70 meter. Try other values if required.**


**Step 4: Mapping the pixels to metric (meter or inches).**


**Step 5: Find the distance between, the center point of one person to another person in meters.**

