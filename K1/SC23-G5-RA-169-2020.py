import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys

kernel = np.ones((3,3), np.uint8)
if len(sys.argv)-1 == 0:    
    path = 'pictures2/'
else:
    path = sys.argv[1]

def main():
    img_names,correct_values = load_data('bulbasaur_count.csv')
    aprox_values = []

    for idx,img_name in enumerate(img_names):
        img = cv2.imread(path+img_name)
        img,img_bin = preprocess_image(img)
        aprox_value = get_aprox_bulb(img,img_bin)
        aprox_values.append(aprox_value)
        print(f"{img_name}-{correct_values[idx]}-{aprox_value}")

    MAE = calculate_mae(correct_values,aprox_values)
    print(f"MAE={MAE}")    


def load_data(path):
    csv_file = pd.read_csv(path)
    img_names = csv_file['Naziv slike'].values.tolist()
    correct_values = csv_file['Broj bulbasaur-a'].values.tolist()
    return img_names,correct_values

def preprocess_image(img):

    height, _,_ = img.shape
    img = img[215:height, :]

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv_image)

    _,img_bin = cv2.threshold(s,0,255,cv2.THRESH_OTSU)

    img_close = cv2.morphologyEx(img_bin,cv2.MORPH_CLOSE,kernel,iterations=2)
    return img,img_close
  
def get_aprox_bulb(img,img_bin):

    dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 3)

    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0) 
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(img_bin, kernel, iterations=3)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _,markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)

    unique_colors = {x for l in markers for x in l}
    num_of_bulb = len(unique_colors) - 2
    return check_contour_aspect_ratio(markers)

def check_contour_aspect_ratio(markers):
    num_of_bulb = 0

    for color in np.unique(markers):
        if color < 2: 
            continue

        marker_mask = np.uint8(markers == color)
        contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            _, _, w, h = cv2.boundingRect(contours[0])
            aspect_ratio = float(w) / h
    
            if aspect_ratio <= 1.3:
                num_of_bulb += 1

    return num_of_bulb

def calculate_mae(correct,aprox):
    abso = [abs(c - s) for c, s in zip(correct, aprox)]
    return sum(abso)/len(correct)


if __name__ == "__main__":
    main()