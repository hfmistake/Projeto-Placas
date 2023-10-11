import os
from ultralytics import YOLO
from PIL import Image
import uuid as uid
import threading
import cv2
import torch


def show(video_source):
    if not torch.cuda.is_available():
        print("AVISO!!! a sua GPU não está disponível.")
    model = YOLO('models/yolov8n.pt')
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def predict(video_source, line, model, object_names, object_indices, directory):
    if not torch.cuda.is_available():
        print("AVISO!!! a sua GPU não está disponível.")
    filt = []
    results = model.track(video_source, stream=True, iou=0.5, classes=object_indices, conf=0.6)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if model.names[int(box.cls[0])] in object_names:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    if track_id in filt:
                        continue
                    else:
                        filt.append(track_id)
                    im = Image.fromarray(r.orig_img)
                    im.save(f'{directory}/{uid.uuid4()}.jpg')
                    print("captura realizada")


def main():
    linha_22 = [100, 300, 1800, 300]
    linha_23 = [200, 330, 800, 330]
    vehicles = ["car", "motorcycle", "truck", "bus"]
    vehicles_index = [2, 3, 5, 7]
    cam22 = os.environ.get("CAM_22")
    cam23 = os.environ.get("CAM_23")
    model_escolhido = YOLO('models/yolov8n.pt')
    video1 = "videos/1.mp4"
    video2 = "videos/2.mp4"

    thread_tracker = threading.Thread(target=predict,
                                      args=(video1, linha_22, model_escolhido, vehicles, vehicles_index, "results22"),
                                      daemon=True)
    thread_show = threading.Thread(target=show, args=(video1,), daemon=True)
    thread_tracker.start()
    thread_show.start()
    thread_tracker.join()
    thread_show.join()


if __name__ == '__main__':
    main()
