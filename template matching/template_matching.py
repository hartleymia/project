# This code is the artefact submitted alongside the dissertation titled:
# "Using Template Matching to Find the Matching Symbol Between a Pair of Dobble Cards"
# By Mia Hartley 25179600 <25179600@students.lincoln.ac.uk> 

import cv2

#function to load the images
def load_images(file_list):
    # Check if file_list is a string and convert it to a list if needed
    if isinstance(file_list, str):
        file_list = [file_list]
    images = []
    for f in file_list:
        image = cv2.imread(f)
        #error handling for if the file is not valid or cant be opened
        if image is None:
            print(f"Could not read image file '{f}'")
        else:
            images.append(image)
    return images

# function to preprocess the card images
def preprocess_card_images(card_images):
    preprocessed_images = []
    for card_image in card_images:
        # Resize to a uniform size
        card_image = cv2.resize(card_image, (400, 400))
        # Convert to grayscale
        card_image_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to remove noise
        card_image_gray = cv2.GaussianBlur(card_image_gray, (5, 5), 0)
        preprocessed_images.append(card_image_gray)
    return preprocessed_images

# function to convert symbol images to grayscale and find keypoints and descriptors
def detect_symbols(symbol_images):
    symbol_kps = []
    symbol_descs = []
    for symbol_image in symbol_images:
        symbol_gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(symbol_gray, None)
        symbol_kps.append(kp)
        symbol_descs.append(desc)
    return symbol_kps, symbol_descs

# function to convert card images to grayscale and find keypoints and descriptors
def detect_cards(card_preprocessed):
    card_kps = []
    card_descs = []
    for card_image in card_preprocessed:
        kp, desc = orb.detectAndCompute(card_image, None)
        card_kps.append(kp)
        card_descs.append(desc)
    return card_kps, card_descs

# function to find matches between symbols and cards
def find_matches(card_descs, symbol_descs):
    matches = []
    for i, card_desc in enumerate(card_descs):
        for j, symbol_desc in enumerate(symbol_descs):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            match = bf.match(card_desc, symbol_desc)
            match = sorted(match, key=lambda x:x.distance)
            matches.append((i, j, match))
    return matches

# function to find symbol that appears in both cards
def find_matching_symbol(matches, symbol_files):
    symbol_count = [0] * len(symbol_files)
    for match in matches:
        if match[0] == 0:
            symbol_count[match[1]] += 1
        elif match[0] == 1:
            symbol_count[match[1]] += 1
    matching_symbol_idx = symbol_count.index(2)
    matching_symbol_file = symbol_files[matching_symbol_idx]
    matching_symbol_image = cv2.imread(matching_symbol_file)
    # print the name of the symbol that appears in both cards
    print("Matching symbol found:", matching_symbol_file)
    # show the image of the symbol that appears in both cards
    cv2.imshow("Matching symbol", matching_symbol_image) 
    return matching_symbol_file, matching_symbol_image

# function to perform template matching with ORB 
def perform_template_matching(card_image, symbol_image):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    kp1, des1 = orb.detectAndCompute(symbol_image, None)
    kp2, des2 = orb.detectAndCompute(card_image, None)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(symbol_image, kp1, card_image, kp2, matches[:10], None)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    return img_matches

# load card images by calling the load_images function
card_files = ["cards/card_1.jpg", "cards/card_2.jpg"]
card_images = load_images(card_files)

# load symbol images by calling the load_images function
symbol_files = ["symbols/clock.png", "symbols/dog.png", "symbols/water.png", "symbols/sun.png", 
                "symbols/candle.png", "symbols/target.png", "symbols/fire.png", "symbols/lightning.png", 
                "symbols/bottle.png", "symbols/musical note.png", "symbols/question mark.png", "symbols/dinosaur.png", 
                "symbols/exclamation mark.png", "symbols/apple.png"]
symbol_images = load_images(symbol_files)

# call on the preprocess_card_images function
card_preprocessed = preprocess_card_images(card_images)

# define ORB detector
orb = cv2.ORB_create()

# call the detect_symbols function
symbol_kps, symbol_descs = detect_symbols(symbol_images)

# call on the detect_cards function
card_kps, card_descs = detect_cards(card_preprocessed)

# call on the find_matches function
matches = find_matches(card_descs, symbol_descs)

# call on the find_matching_symbol_function
matching_symbol_file, matching_symbol_image = find_matching_symbol(matches, symbol_files)

# call the perform_template_matching function
for i in range(len(card_preprocessed)):
    for j in range(len(symbol_images)):
        img_matches = perform_template_matching(card_preprocessed[i], symbol_images[j])
