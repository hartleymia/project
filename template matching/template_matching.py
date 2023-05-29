# This code is the artefact submitted alongside the dissertation titled:
# "Using Template Matching to Find the Matching Symbol Between a Pair of Dobble Cards"
# By Mia Hartley 25179600 <25179600@students.lincoln.ac.uk> 

import cv2
import time

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
    preprocessed_card_images = []
    for card_image in card_images:
        #resize to a uniform size
        card_image = cv2.resize(card_image, (600, 600))
        #convert to grescale
        card_image_grey = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        #apply Gaussian blur to remove noise
        card_image_grey = cv2.GaussianBlur(card_image_grey, (5, 5), 0)
        #use canny to detect image edges
        card_image_edges = cv2.Canny(card_image_grey, 100, 200)
        preprocessed_card_images.append(card_image_edges)
    return preprocessed_card_images

# function to preprocesses the symbol images
def preprocess_symbol_images(symbol_images):
    preprocess_symbol_images = []
    for symbol_image in symbol_images:
        #resize to uniform size
        symbol_image = cv2.resize(symbol_image, (400,400))
        #convert to greyscale
        symbol_image_grey = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
        #apply gaussian blur to remove noise
        symbol_image_grey = cv2.GaussianBlur(symbol_image_grey, (5,5), 0)
        #use canny to detect image edges
        symbol_image_edges = cv2.Canny(symbol_image_grey, 100, 200)
        preprocess_symbol_images.append(symbol_image_edges)
    return preprocess_symbol_images

# function to find symbol keypoints and descriptors
def detect_symbols(symbol_preprocessed):
    symbol_kps = []
    symbol_descs = []
    for symbol_image in symbol_preprocessed:
        kp, desc = orb.detectAndCompute(symbol_image, None)
        symbol_kps.append(kp)
        symbol_descs.append(desc)
    return symbol_kps, symbol_descs

# function to find card keypoints and descriptors
def detect_cards(card_preprocessed):
    card_kps = []
    card_descs = []
    for card_image in card_preprocessed:
        kp, desc = orb.detectAndCompute(card_image, None)
        card_kps.append(kp)
        card_descs.append(desc)
    return card_descs

# function to find matches between symbols and cards
def find_matches(card_descs, symbol_descs):
    matches = []
    for i, card_desc in enumerate(card_descs):
        for j, symbol_desc in enumerate(symbol_descs):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            match = bf.match(card_desc, symbol_desc)
            match = sorted(match, key=lambda x:x.distance)
            if match[0].distance < 8:
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
    #print the name of the symbol that appears in both cards
    print("Matching symbol found:", matching_symbol_file)
    #show the image of the symbol that appears in both cards
    cv2.imshow("Matching symbol", matching_symbol_image) 
    return matching_symbol_idx

# function to perform feature based template matching with ORB 
def perform_template_matching(card_image, symbol_image):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    kp1, des1 = orb.detectAndCompute(symbol_image, None)
    kp2, des2 = orb.detectAndCompute(card_image, None)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(symbol_image, kp1, card_image, kp2, matches[:20], None)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    return img_matches

def match_cards(card_files):
    #start time - in seconds
    start_time = time.perf_counter()
    #load card images by calling the load_images function
    card_images = load_images(card_files)
    #call on the preprocess_card_images function
    card_preprocessed = preprocess_card_images(card_images)
    #call on the detect_cards function
    card_descs = detect_cards(card_preprocessed)
    #call on the find_matches function
    matches = find_matches(card_descs, symbol_descs)
    #call on the find_matching_symbol_function
    matching_symbol_idx = find_matching_symbol(matches, symbol_files)
    #end time - in seconds
    end_time = time.perf_counter()
    #print time it takes to match symbol in seconds
    total_time = end_time - start_time
    print(f"time to find matching symbol: {total_time:0.4f} seconds")
    perform_template_matching(card_preprocessed[0], symbol_preprocessed[matching_symbol_idx])

# define ORB detector
orb = cv2.ORB_create()

# load symbol images by calling the load_images function
symbol_files = ["symbols/candle.png", "symbols/dog.png", "symbols/water.png", "symbols/sun.png", 
                "symbols/clock.png", "symbols/target.png", "symbols/fire.png", "symbols/lightning.png", 
                "symbols/bottle.png", "symbols/musical note.png", "symbols/question mark.png", "symbols/dinosaur.png", 
                "symbols/exclamation mark.png", "symbols/apple.png", "symbols/hand.png"]
symbol_images = load_images(symbol_files)

# call on the preprocess_symbol_images function
symbol_preprocessed = preprocess_symbol_images(symbol_images)

# call the detect_symbols function
symbol_kps, symbol_descs = detect_symbols(symbol_preprocessed)

#call the match cards function
match_cards(["cards/card_1.jpg", "cards/card_2.jpg"])
