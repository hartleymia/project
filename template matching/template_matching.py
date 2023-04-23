import cv2
import numpy as np
from PIL import Image

# Load card and symbol images
card_files = ["cards/card_1.jpg", "cards/card_2.jpg"]
card_images = [Image.open(filename) for filename in card_files]

symbol_files = ["symbols/clock.png", "symbols/candle.png", "symbols/apple.png", "symbols/dinosaur.png", 
                "symbols/dog.png", "symbols/exclamation mark.png", "symbols/fire.png", "symbols/lightning.png", 
                "symbols/bottle.png", "symbols/musical note.png", "symbols/question mark.png", "symbols/sun.png", 
                "symbols/target.png", "symbols/water.png"]
symbol_images = [Image.open(filename) for filename in symbol_files]

# Convert symbol images to grayscale and find keypoints and descriptors
symbol_kps = []
symbol_descs = []
orb = cv2.ORB_create()
for symbol_image in symbol_images:
    symbol_gray = cv2.cvtColor(np.array(symbol_image), cv2.COLOR_RGB2GRAY)
    kp, desc = orb.detectAndCompute(symbol_gray, None)
    symbol_kps.append(kp)
    symbol_descs.append(desc)

# Convert card images to grayscale and find keypoints and descriptors
card_kps = []
card_descs = []
for card_image in card_images:
    card_gray = cv2.cvtColor(np.array(card_image), cv2.COLOR_RGB2GRAY)
    kp, desc = orb.detectAndCompute(card_gray, None)
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
    if match[0] == 0 and match[1] != 12:  # Only consider matches with card 1 and not the water symbol
        symbol_count[match[1]] += 1
    elif match[0] == 1 and match[1] != 12:  # Only consider matches with card 2 and not the water symbol
        symbol_count[match[1]] += 1

matching_symbol_idx = symbol_count.index(2)  # Find the symbol that appears in both cards
matching_symbol_file = symbol_files[matching_symbol_idx]
matching_symbol_image = Image.open(matching_symbol_file)

print("Matching symbol found:", matching_symbol_file)
matching_symbol_image.show()



