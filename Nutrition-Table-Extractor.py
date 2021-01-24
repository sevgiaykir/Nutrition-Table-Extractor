import cv2
import re
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, grayscale, and threshold
image = cv2.imread('food-label.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh_img = thresh.copy()
thresh_img = cv2.resize(thresh_img, (600, 600))
cv2.imshow('Thresholding', thresh_img)
cv2.waitKey(0)

# configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6'
# now feeding image to tesseract
details = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=custom_config, lang='eng')
print(details.keys())

# draw the bounding box on text area
def boundingBox(details,thresh):
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > 30:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],
                details['height'][sequence_number])
            thresh = cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)

boundingBox(details,thresh)

# Remove horizontal lines
def removeHorizontal(thresh):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 2)

removeHorizontal(thresh)

# Remove vertical lines
def removeVertical(thresh):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)

removeVertical(thresh)

# Dilate to connect text and remove dots
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
dilate = cv2.dilate(thresh, kernel, iterations=2)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 500:
        cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

# Bitwise-and to reconstruct image
result = cv2.bitwise_and(image, image, mask=dilate)
result[dilate == 0] = (255, 255, 255)

# OCR
d = pytesseract.image_to_string(result, lang='eng')

print("Nutritional content: ")

try:
    cal = re.findall(r"Calories [\w]+", d)
    print(cal[0])
except IndexError:
    print("No information about Calories")
try:
    tof = re.findall(r"Total Fat [\w]+", d)
    print(tof[0])
except IndexError:
    print("No information about Total Fat")
try:
    sf = re.findall(r"Saturated Fat [\w]+", d)
    print(sf[0])
except IndexError:
    print("No information about Saturated Fat")
try:
    tf = re.findall(r"Trans Fat [\w]+", d)
    print(tf[0])
except IndexError:
    print("No information about Trans Fat")
try:
    pf = re.findall(r"Polyunsaturated Fat [\w]+", d)
    print(pf[0])
except IndexError:
    print("No information about Polyunsaturated Fat")
try:
    mf = re.findall(r"Monounsaturated Fat [\w]+", d)
    print(mf[0])
except IndexError:
    print("No information about Monounsaturated Fat")
try:
    ch = re.findall(r"Cholesterol [\w]+", d)
    print(ch[0])
except IndexError:
    print("No information about Cholesterol")
try:
    s = re.findall(r"Sodium [\w]+", d)
    print(s[0])
except IndexError:
    print("No information about Sodium")
try:
    toc = re.findall(r"Total Carbohydrate [\w]+", d)
    print(toc[0])
except IndexError:
    print("No information about Total Carbohydrate")
try:
    fi = re.findall(r"Fiber [\w]+", d)
    print(fi[0])
except IndexError:
    print("No information about Fiber")
try:
    su = re.findall(r"Sugars [\w]+", d)
    print(su[0])
except IndexError:
    print("No information about Sugars")
try:
    p = re.findall(r"Protein [\w]+", d)
    print(p[0])
except IndexError:
    print("No information about Protein")

result = cv2.resize(result, (600, 600))
thresh = cv2.resize(thresh, (600, 600))
cv2.imshow('Bounding Box on Text Area', thresh)
cv2.waitKey(0)
# Maintain output window until user presses a key
cv2.imshow('Result', result)
cv2.waitKey()
