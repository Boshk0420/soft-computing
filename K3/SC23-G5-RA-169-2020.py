import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow import keras
from keras.models import Sequential,load_model, save_model
from keras.layers import Dense
from keras.optimizers import SGD
import os
import sys

if len(sys.argv)-1 == 0:    
    path = 'data2/'
else:
    path = sys.argv[1]

imgPath = path+'pictures/'
csvPath = path+'res.csv'
modelPath = "ocr_model_cz.h5"


kernel = np.ones((3, 3)) 

def main():

    file,text = load_csv()
    alphabet,inputs,k_means = prepare_train_set(file,text)
   
    ann = get_ann(alphabet,inputs,save=False)

    prediction = predict_captcha(file,text,alphabet,ann,k_means)
    calculate_error(file,text,prediction)
    return 0 

def get_ann(alphabet,inputs,save=False):

    ann = None

    if os.path.exists(modelPath) and save:
        ann = load_model(modelPath)
        print(f"\nModel loaded successfully!")
    else:
        outputs = convert_output(alphabet)
        ann = create_ann(output_size=len(alphabet))
        ann = train_ann(ann,inputs,outputs,epochs=2000,save=save) 

    return ann  

def load_csv():
    csv_file = pd.read_csv(csvPath)
    captcha_file = csv_file['file'].values.tolist()
    captcha_text = csv_file['text'].values.tolist()

    return captcha_file,captcha_text

def prepare_train_set(captcha_file,captcha_text):
    chars = [char for string in captcha_text for char in string if char != ' ']
    inputs_all = []
    distances = []

    for file in captcha_file:

        img = load_image(imgPath+file)
        img_bin = to_binary(img)

        _,letter,distance = select_roi_with_distances(img,img_bin)

        input = prepare_for_ann(letter)
        inputs_all += input
        distances += distance

    if ( len(chars) != len(inputs_all) ):
        raise ValueError(f"Chars and Inputs set are of different length!! Char length: {len(chars)} ; Input length: {len(inputs_all)} ;")
    
    inputs = [] 
    alphabet = []

    for i,char in enumerate(chars):

        if(char not in alphabet):
            alphabet.append(char)
            inputs.append(inputs_all[i])
  

    distances = np.array(distances).reshape(len(distances), 1)
    k_means = KMeans(n_clusters=2,n_init='auto')      
    k_means.fit(distances)

    return alphabet,inputs,k_means
    
def predict_captcha(captcha_file,captcha_text,alphabet,ann,k_means):

    prediction = []

    for file in captcha_file:
        img = load_image(imgPath+file)
        img_bin = to_binary(img)

        _,letter,distances = select_roi_with_distances(img,img_bin)

        distances = np.array(distances).reshape(len(distances), 1)
        labels = k_means.predict(distances)
  
        input = prepare_for_ann(letter)
        results = ann.predict(np.array(input,np.float32))
        result = display_result_with_spaces(results,alphabet,labels)
        prediction.append(result)

    return prediction

def calculate_error(captcha_file,correct,predicted):

    error = 0

    for i,file in enumerate(captcha_file):
        print(f"{file}-{correct[i]}-{predicted[i]}")
        result = hamming(correct[i],predicted[i])
        error += result
    
    print(f"Hamming distance error is: {error}")

def hamming(s1,s2):
    result=0

    if len(s1)!=len(s2):    
        result += abs(len(s1)-len(s2))

        small = s1 if len(s1) < len(s2) else s2
        big = s1 if len(s1) > len(s2) else s2

        for i,char in enumerate( small ):
            if char!=big[i]:
                result+=1
    else:
        for (char1,char2) in zip(s1,s2):
            if char1!=char2:
                result+=1

    return result

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    cropped_image = img[175:265,250:830]
    return cropped_image

def to_binary(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image

def matrix_to_vector(image):
    return image.flatten()

def scale_to_range(image):
    return image/255

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs,save=False):
    X_train = np.array(X_train, np.float32) 
    y_train = np.array(y_train, np.float32)
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")

    if(save):
        save_model(ann,modelPath)
        print(f"\nModel saved to file: {modelPath}")

    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def filter_diacritics(boxes):

    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    pairs = []
    skip_next = False

    for index in range(len(sorted_boxes)-1):
        
        current = sorted_boxes[index]
        next = sorted_boxes[index+1]
        
        if(skip_next):

            if(index == len(sorted_boxes)-2):
                pairs.append(next)
            skip_next = False   
            continue
    

        x1_c,y1_c,x2_c,y2_c = current
        x1_n,y1_n,x2_n,y2_n = next

        if(is_in_between(current,next) or is_in_between(next,current)):
            x1 = min((x1_c,x1_n))
            y1 = min((y1_c,y1_n))
            x2 = max((x2_c,x2_n))
            y2 = max((y2_c,y2_n))

            pair = (x1,y1,x2,y2)
            pairs.append(pair)
            skip_next = True
        else:
            pairs.append(current)
            if(index == len(sorted_boxes)-2):
                pairs.append(next)    

    return pairs

def is_in_between(box1,box2,threshold=10):
    x11,_,x12,_ = box1
    x21,_,x22,_ = box2

    return x11 - threshold <= x21 and x12 + threshold >= x22 

def draw_bounding_boxes(boxes,img):
    for box in boxes:
        x1,y1,x2,y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def get_regions(boxes,img_bin):
    regions = []

    for box in boxes:
        x1,y1,x2,y2 = box
        region = img_bin[y1:y2+1,x1:x2+1]
        regions.append([resize_region(region),(x1,y1,x2-x1,y2-y1)])

    return regions

def select_roi_with_distances(image_orig, image_bin):

    contours, _ = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    bounding_boxes = [(x, y, x+w, y+h) for contour in contours for x, y, w, h in [cv2.boundingRect(contour)]]   
    pairs = filter_diacritics(bounding_boxes)
    

    draw_bounding_boxes(pairs,image_orig)


    regions_array = get_regions(pairs,image_bin)
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    

    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def display_result_with_spaces(outputs, alphabet, labels):
    
    result = alphabet[winner(outputs[0])]

    for idx, output in enumerate(outputs[1:, :]):
        if(labels[idx]==1):
            result += ' '
        result += alphabet[winner(output)]
    return result

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    main()

