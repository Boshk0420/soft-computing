import os
import numpy as np
import cv2 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv)-1 == 0:    
    path = 'data2/'
else:
    path = sys.argv[1]

picPath = path + "pictures/" ## 120 x 60
vidPath = path + "videos/"
csvPath = path + "counts.csv"


def main():


    pos_imgs, neg_imgs = load_all_images()
    hog,x, y = hog_desc(pos_imgs,neg_imgs)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    svm = SVM_Fit(x_train,x_test,y_train,y_test)

    process_all_videos(hog,svm)

def process_all_videos(hog,svm,showVideo=False):
    
    assummed_values = []
    video_names,correct_values = load_csv()
    
    for video_name in video_names:
        assummed_value = process_video(vidPath+video_name+".mp4",hog,svm,showVideo)
        assummed_values.append(assummed_value)

    mae = calculate_mae(correct_values,assummed_values)
    print_results(video_names,assummed_values,correct_values)
    print(f"MAE={mae}")

def print_results(video_names,assummed,correct):
    
    for idx,video in enumerate(video_names):
        print(f"{video}-{correct[idx]}-{assummed[idx]}")

def load_csv():
    csv_file = pd.read_csv(csvPath)
    video_names = csv_file['Naziv_videa'].values.tolist()
    correct_values = csv_file['Broj_kolizija'].values.tolist()

    return video_names,correct_values

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def load_all_images():

    pos_imgs = []
    neg_imgs = []

    for img_name in os.listdir(picPath):
        img_path = os.path.join(picPath, img_name)
        img = load_image(img_path)
        if 'p_' in img_name:
            pos_imgs.append(img)
        elif 'n_' in img_name: 
            neg_imgs.append(img)


    return pos_imgs,neg_imgs

def hog_desc(pos_imgs,neg_imgs):

    img_size = (120,60)
    cell_size = (8,8)
    block_size = (3,3)
    nbins = 9

    pos_features = []
    neg_features = []
    labels = []

    hog = cv2.HOGDescriptor(_winSize=(img_size[0] // cell_size[1] * cell_size[1],
                                      img_size[1] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    
    for img in pos_imgs:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog.compute(img))
        labels.append(0)


    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    return hog,x,y

def SVM_Fit(x_train,x_test,y_train,y_test):

    clf_svm = SVC(kernel='linear', probability=True,gamma="auto") 
    clf_svm.fit(x_train, y_train)

    return clf_svm

def process_video(video_path,hog,svm, showVideo=False):
     
    num_of_cars = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1,frame_num)
    frames_to_skip = 1

    
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break

        if frame_num % frames_to_skip != 0:
            continue
        frames_to_skip = 2
        
        resFrame = resize_frame(frame)
        lines = detect_red_lines(resFrame)

        found_boxes,scores = sliding_window(resFrame,hog,svm)
        boxes = non_max_suppression_fast(np.column_stack((found_boxes,scores)),0.30)

        boxes = [box[:4] for box in boxes]
        detected = detect_collision(boxes,lines)

        if(detected != 0 ):
                num_of_cars += detected
                frames_to_skip = frame_num+6

        if(showVideo):
            tuple_boxes = [tuple(box) for box in boxes]
            draw_boxes(resFrame,tuple_boxes)
            show_video(resFrame,lines)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

    

    return num_of_cars

def resize_frame(frame):

    height, width = frame.shape[:2]
    new_width = 720
    aspect_ratio = new_width / width
    new_height = int(height * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    return resized_frame

def show_video(frame,lines):
     
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
    cv2.imshow('Line Detection', frame)

def draw_boxes(frame,boxes):
    for (x_min, y_min, x_max, y_max) in boxes:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def detect_red_lines(frame, min_aspect_ratio=3):
 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _,s,_ = cv2.split(hsv)

    blurred = cv2.GaussianBlur(s, (3, 3), 0)

    edges = cv2.Canny(blurred, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=15, maxLineGap=5)

    filtered_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            aspect_ratio = abs((y2 - y1) / (x2 - x1 + 1e-5))
            if aspect_ratio > min_aspect_ratio and abs(x2 - x1) < 5:
                filtered_lines.append(line)


    return filtered_lines

def sliding_window(frame,hog,svm):
    window_size = (120, 60)
    step_size = 12

    found_boxes = []
    scores = []

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    for y in range(0, gray_frame.shape[0]-window_size[1], step_size):
        for x in range(0, gray_frame.shape[1]-window_size[0], step_size):

            window = gray_frame[y:y+window_size[1], x:x+window_size[0]]
            
            if(window.shape == (window_size[1],window_size[0])):

                features = hog.compute(window).reshape(1,-1)
                prediction = svm.predict(features)

                if prediction >= 0.9:
                    found_boxes.append((x, y, x+window_size[0], y+window_size[1]))
                    scores.append(prediction)

    return found_boxes,scores

def non_max_suppression_fast(boxes, overlapThresh):

	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
	
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		
		overlap = (w * h) / area[idxs[:last]]
	
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	
	return boxes[pick].astype("int")

def detect_collision(boxes,line):

    collisions = 0

    if not len(line) or not len(boxes):
        return 0

    line_x1, line_y1, line_x2, line_y2  = line[0][0]

    for box in boxes:
        x1, y1, x2, y2 = box

        if (
            (x1 <= line_x1 <= x2 or x1 <= line_x2 <= x2) and
            (line_y2 <= y1 <= line_y1 or line_y2 <= y2 <= line_y1)
        ):
            collisions += 1

    return collisions

def calculate_mae(correct,aprox):
    abso = [abs(c - s) for c, s in zip(correct, aprox)]
    return sum(abso)/len(correct)

if __name__ == "__main__":
    main()