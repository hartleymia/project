import cv2
import numpy as np
from PIL import Image

# Read card and symbol images
card_files = ["cards/card_1.jpg", "cards/card_2.jpg"]
card_images = [cv2.imread(filename) for filename in card_files]

symbol_files = ["symbols/apple.png", "symbols/candle.png", "symbols/clock.png", "symbols/dinosaur.png", 
                "symbols/dog.png", "symbols/exclamation mark.png", "symbols/fire.png", "symbols/lightning.png", 
                "symbols/bottle.png", "symbols/musical note.png", "symbols/question mark.png", "symbols/sun.png", 
                "symbols/target.png", "symbols/water.png"]
symbol_images = [cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in symbol_files]

# Preprocess card images
card_preprocessed = []
for card_image in card_images:
    # Resize to a uniform size
    card_image = cv2.resize(card_image, (400, 400))
    # Convert to grayscale
    card_image_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    card_image_gray = cv2.GaussianBlur(card_image_gray, (5, 5), 0)
    card_preprocessed.append(card_image_gray)

# Perform template matching with ORB
orb = cv2.ORB_create()
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

for i in range(len(card_preprocessed)):
    for j in range(len(symbol_images)):
        # Detect keypoints and compute descriptors for symbol image
        kp1, des1 = orb.detectAndCompute(symbol_images[j], None)
        # Detect keypoints and compute descriptors for preprocessed card image
        kp2, des2 = orb.detectAndCompute(card_preprocessed[i], None)
        # Match descriptors
        matches = matcher.match(des1, des2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw matches
        img_matches = cv2.drawMatches(symbol_images[j], kp1, card_preprocessed[i], kp2, matches[:10], None)
        # Show image
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)

cv2.destroyAllWindows()
