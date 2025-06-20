import os
import cv2 as cv
import numpy as np
import pickle
import time
from ursina import *
from PIL import Image

app = Ursina()
window.fullscreen = True
main_menu = Entity(parent=camera.ui)

if os.path.exists('names.pkl'):
    with open('names.pkl', 'rb') as f:
        Ph = pickle.load(f)
else:
    Ph = []

if not os.path.exists('data'):
    os.makedirs('data')

haar_cascade = cv.CascadeClassifier('face.xml')
if not os.path.exists('recognizer_saved.yml'):
    recog = cv.face.LBPHFaceRecognizer_create()
    recog.save('recognizer_saved.yml')
recog = cv.face.LBPHFaceRecognizer_create()
recog.read('recognizer_saved.yml')

c = None
webcam_on = False
capturing = False
obj_det_on = False
cap_name = ""
cap_count = 0
last_cap_time = 0

object_net = None
class_names = []
if os.path.exists('coco.names'):
    with open('coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

def load_obj_model():
    global object_net
    model_path = 'yolov3.weights'
    config_path = 'yolov3.cfg'
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        object_net = cv.dnn.readNet(model_path, config_path)
        return True
    else:
        return False

def start_fc_rec():
    global c, webcam_on, obj_det_on
    main_menu.enabled = False
    obj_det_on = False
    if c is not None:
        c.release()
    c = cv.VideoCapture(0)
    webcam_on = True

def reg_face():
    main_menu.enabled = False
    in_field.text = ''
    in_field.enabled = True
    reg_popup.enabled = True

def start_obj_rec():
    global c, webcam_on, obj_det_on
    main_menu.enabled = False
    webcam_on = False
    
    if not load_obj_model():
        main_menu.enabled = True
        return
    
    obj_det_on = True
    if c is not None:
        c.release()
    c = cv.VideoCapture(0)

def exit_app():
    if c is not None:
        c.release()
    cv.destroyAllWindows()
    application.quit()

def train_fc_model():
    features = []
    labels = []
    
    for person in Ph:
        os.makedirs(f'data/{person}', exist_ok=True)
    
    for label, person in enumerate(Ph):
        per_path = os.path.join('data', person)
        if not os.path.exists(per_path):
            continue
            
        for img_nm in os.listdir(per_path):
            img_path = os.path.join(per_path, img_nm)
            feature = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    
    if len(features) > 0 and len(labels) > 0:
        global recog
        recog = cv.face.LBPHFaceRecognizer_create()
        recog.train(features, np.array(labels))
        recog.save('recognizer_saved.yml')
        print(f"Trained on {len(features)} samples for {len(set(labels))} people")

def cap_fc(name):
    global c, capturing, cap_name, cap_count, last_cap_time
    try:
        os.makedirs(f'data/{name}', exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    if c is not None:
        c.release()
    c = cv.VideoCapture(0)
    cap_name = name
    cap_count = 0
    capturing = True
    last_cap_time = time.time()

def cofirm_reg():
    global Ph
    name = in_field.text.strip()
    if name == '':
        return
        
    if name not in Ph:
        Ph.append(name)
        with open('names.pkl', 'wb') as f:
            pickle.dump(Ph, f)
            
    in_field.enabled = False
    reg_popup.enabled = False
    cap_fc(name)

def det_obj(frame):
    height, width = frame.shape[:2]
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    blob = cv.dnn.blobFromImage(
        frame, 
        scalefactor=1/255.0, 
        size=(320, 320),
        mean=(0, 0, 0), 
        swapRB=True,
        crop=False
    )
    
    object_net.setInput(blob)
    
    layer_names = object_net.getLayerNames()
    output_layers = [layer_names[i-1] for i in object_net.getUnconnectedOutLayers().flatten()]
    
    start = time.time()
    outputs = object_net.forward(output_layers)
    end = time.time()
    print(f"Time: {(end-start)*1000:.2f} ms")
    
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.7:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            
            label = str(class_names[class_ids[i]]) if (class_names and class_ids[i] < len(class_names)) else str(class_ids[i])
            
            color = (0, 255, 0) if "person" in label.lower() else (0, 0, 255)
            
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidences[i]:.2f}"
            cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return cv.cvtColor(frame, cv.COLOR_RGB2BGR)

reg_popup = Entity(parent=camera.ui, enabled=False)
Text(text="Enter Name:", parent=reg_popup, y=0.05, scale=1.2)
in_field = InputField(parent=reg_popup, y=-0.05, scale=(0.5, 0.1))
in_field.enabled = False
Button(text='Confirm', parent=reg_popup, y=-0.18, scale=(0.2, 0.07), on_click=cofirm_reg)

Button(text='Recognize Face', scale=(0.3, 0.1), parent=main_menu, y=0.1, on_click=start_fc_rec)
Button(text='Add New Face', scale=(0.3, 0.1), parent=main_menu, y=-0.05, on_click=reg_face)
Button(text='Identify Objects', scale=(0.3, 0.1), parent=main_menu, y=-0.2, on_click=start_obj_rec)
Button(text='Exit', scale=(0.3, 0.1), parent=main_menu, y=-0.35, on_click=exit_app)

def update():
    global c, webcam_on, capturing, cap_name, cap_count, last_cap_time, obj_det_on

    if (webcam_on or capturing or obj_det_on) and c is not None:
        ret, frame = c.read()
        if not ret:
            return
            
        if obj_det_on:
            frame = det_obj(frame)
            cv.imshow('Object Detection', frame)
            
            if cv.waitKey(1) & 0xFF == ord('b'):
                obj_det_on = False
                if c is not None:
                    c.release()
                cv.destroyAllWindows()
                main_menu.enabled = True
            return
                
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_rec = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        for (x, y, w, h) in face_rec:
            feature = gray[y:y+h, x:x+w]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if capturing:
                current_time = time.time()
                if current_time - last_cap_time > 1 and cap_count < 10:
                    try:
                        feature = cv.resize(feature, (200, 200))
                        cv.imwrite(f'data/{cap_name}/{cap_count}.jpg', feature)
                        cap_count += 1
                        last_cap_time = current_time
                        flash = frame.copy()
                        cv.rectangle(flash, (x, y), (x+w, y+h), (0, 255, 255), 4)
                        cv.imshow('Webcam', flash)
                        cv.waitKey(100)
                    except Exception as e:
                        print(f"Error: {e}")

                progress_text = f"Capturing: {cap_count}/10 (Move slightly)"
                cv.putText(frame, progress_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(frame, f"Current: {cap_name}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if cap_count >= 10:
                    capturing = False
                    if c is not None:
                        c.release()
                    cv.destroyAllWindows()
                    train_fc_model()
                    main_menu.enabled = True

            elif webcam_on:
                try:
                    feature = cv.resize(feature, (200, 200))
                    label, confidence = recog.predict(feature)
                    
                    if confidence < 100:
                        name = Ph[label] if label < len(Ph) else "Unknown"
                        cv.putText(frame, f"{name} ({int(confidence)})", (x, y-10), 
                                 cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv.putText(frame, "Unknown", (x, y-10), 
                                 cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error: {e}")

        cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('b'):
            if webcam_on or capturing:
                webcam_on = False
                capturing = False
                if c is not None:
                    c.release()
                cv.destroyAllWindows()
                main_menu.enabled = True

app.run()
