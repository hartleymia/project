import cv2
import numpy as np
from PIL import Image

# Load card and symbol images
card_files = ["cards/card_1.jpg", "cards/card_2.jpg"]
card_images = [cv2.imread(filename) for filename in card_files]

symbol_files = ["symbols/clock.png", "symbols/candle.png", "symbols/apple.png", "symbols/dinosaur.png", 
                "symbols/dog.png", "symbols/exclamation mark.png", "symbols/fire.png", "symbols/lightning.png", 
                "symbols/bottle.png", "symbols/musical note.png", "symbols/question mark.png", "symbols/sun.png", 
                "symbols/target.png", "symbols/water.png"]
symbol_images = [cv2.imread(filename) for filename in symbol_files]

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

# Convert symbol images to grayscale and find keypoints and descriptors
symbol_kps = []
symbol_descs = []
orb = cv2.ORB_create()
for symbol_image in symbol_images:
    symbol_gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(symbol_gray, None)
    symbol_kps.append(kp)
    symbol_descs.append(desc)

# Convert card images to grayscale and find keypoints and descriptors
card_kps = []
card_descs = []
for card_image in card_preprocessed:
    kp, desc = orb.detectAndCompute(card_image, None)
    card_kps.append(kp)
    card_descs.append(desc)

# Find matches between symbols and cards
matches = []
for i, card_desc in enumerate(card_descs):
    for j, symbol_desc in enumerate(symbol_descs):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        match = bf.match(card_desc, symbol_desc)
        match = sorted(match, key=lambda x:x.distance)
        matches.append((i, j, match))

# Find symbol that appears in both cards
symbol_count = [0] * len(symbol_files)
for match in matches:
    if match[0] == 0:   # Only consider matches with card 1 
        symbol_count[match[1]] += 1
    elif match[0] == 1: # Only consider matches with card 2 
        symbol_count[match[1]] += 1

matching_symbol_idx = symbol_count.index(2)  # Find the symbol that appears in both cards
matching_symbol_file = symbol_files[matching_symbol_idx]
matching_symbol_image = cv2.imread(matching_symbol_file)

print("Matching symbol found:", matching_symbol_file)
cv2.imshow("Matching symbol", matching_symbol_image)

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
for i, card_desc in enumerate(card_descs):
    for j, symbol_desc in enumerate(symbol_descs):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        match = bf.match(card_desc, symbol_desc)
        match = sorted(match, key=lambda x:x.distance)
        matches.append((i, j, match))

# Find symbol that appears in both cards
symbol_count = [0] * len(symbol_files)
for match in matches:
    if match[0] == 0:  # Only consider matches with card 1 
        symbol_count[match[1]] += 1
    elif match[0] == 1:  # Only consider matches with card 2 
        symbol_count[match[1]] += 1

matching_symbol_idx = symbol_count.index(2)  # Find the symbol that appears in both cards
matching_symbol_file = symbol_files[matching_symbol_idx]
matching_symbol_image = Image.open(matching_symbol_file)

print("Matching symbol found:", matching_symbol_file)
matching_symbol_image.show()
