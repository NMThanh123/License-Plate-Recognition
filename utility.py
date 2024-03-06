import cv2 as cv
import numpy as np
from scipy.ndimage import interpolation as inter
import os
import onnxruntime as ort
from thefuzz import fuzz

# initialize model
def initialize_model():
    from paddleocr import PaddleOCR
    file_name = os.access('model/det/det.onnx', os.F_OK)
    if file_name:
        det = ort.InferenceSession('model/det/det.onnx')
        rec = PaddleOCR(det_model_dir='model/ocr/det', rec_model_dir='model/ocr/rec', rec_char_dict_path='model/ocr/en_dict.txt', show_log=False)   
    else:
        return 'Error to initialize model'

    return det, rec

# unwrap license plate for better recognition
def unwrap_image(image, delta=1, limit=15):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    img_rotate = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, \
            borderMode=cv.BORDER_REPLICATE)

    return img_rotate

# auto adjust brightness and contrast of image
def adjust_image(img, clip_hist_percent=25):

    def convertScale(img, alpha, beta):
        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 1.5

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    image = convertScale(img, alpha=alpha, beta=beta)
    return image

# process correct format string for license plate
def format_string(text):
    replaces1 = {'$':'5', '&':'8', '!':'1', "'": '', ':':'', ']':'','[':'', '(': '', ')':'', 'L':'4', 'T':'1'}
    replaces = {'8':'B', '6':'G', '7':'Z', '0':'D', '4':'L', '5':'S'}
    for c in text:
        if c in replaces1:
            text = text.replace(c, replaces1[c])
    if len(text) <= 6:
        return 'Not have ability to recognize'
    if text[2] in [' ', '.', ':', '  ']:
        text = text.replace(text[2], '-')
    l = list(text.replace(' ',''))
    if l[2] in replaces:
        a=l.pop(2)
        l.insert(2, replaces[a])
    if (l[3] in replaces and l[2] == '-'):
        a=l.pop(3)
        l.insert(3, replaces[a])
    if l[4] in replaces1 and l[3].isalpha():
        a=l.pop(4)
        l.insert(4, replaces1[a])
    if l[-3] == '-':
        l.pop(-3)
        l.insert(-3, '.')
    if l[3] == '1' and l[2] == '-':
        l.pop(3)
        l.insert(3, 'T')
    if len(l) > 7 and l[2].isalpha() and l[3].isalpha():
        l.insert(2, '-')
    text =''.join(t for t in l)

    return text.upper()

# get x1, y1, x2, y2 of bounding box
def get_box(box):
    x1 = int(box[0] - box[2]/2)
    y1 = int(box[1] - box[3]/2)
    x2 = int(box[0] + box[2]/2)
    y2 = int(box[1] + box[3]/2)
    return x1, y1, x2, y2

# preprocess image before recogize
def process_image(image):
    input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32)/255.0
    input_image = np.transpose(input_image, (2, 0, 1))  
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

# check plate exists in treeview or not
def check_plate(text1, text2):
    return fuzz.partial_ratio(text1, text2)