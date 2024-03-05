#%%
import cv2

#%%
def extract_transfrom(img1, img2):
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Optional
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # Optional
    height, width = img2.shape
    
    # Create ORB detector
    orb_detector = cv2.ORB_create(5000)
    
    # Find keypoints and descriptors
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    
    # Match features between the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors
    matches = matcher.match(d1, d2)
    
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))
    
    def apply_transform(img):