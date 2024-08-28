import cv2
import pyttsx3
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')
labels = model.names
engine = pyttsx3.init()

def run_inference(model, image):
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    return boxes, classes, scores

def draw_boxes_and_labels(image, boxes, classes, scores, labels, score_threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(boxes)):
        if scores[i] > score_threshold:
            xmin, ymin, xmax, ymax = boxes[i]
            left = int(xmin)
            right = int(xmax)
            top = int(ymin)
            bottom = int(ymax)
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            class_id = classes[i]
            label = f"{labels[class_id]}: {int(scores[i] * 100)}%"
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(label)
            speak_text(label)

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, classes, scores = run_inference(model, image_rgb)
    draw_boxes_and_labels(image, boxes, classes, scores, labels)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, classes, scores = run_inference(model, image_rgb)
        draw_boxes_and_labels(frame, boxes, classes, scores, labels)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, classes, scores = run_inference(model, image_rgb)
        draw_boxes_and_labels(frame, boxes, classes, scores, labels)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Select an option:")
    print("1. Process an image")
    print("2. Process a video file")
    print("3. Process webcam feed")
    
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        image_path = input("Enter the path to the image: ")
        process_image(image_path)
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        process_video(video_path)
    elif choice == '3':
        process_webcam()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
