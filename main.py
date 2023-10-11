import os
from ultralytics import YOLO
from PIL import Image

# linha para camera 22
limits = [100, 300, 1800, 300]

vehicles = ["car", "motorcycle", "truck", "bus"]
vehicles_index = [2, 3, 5, 7]

# cam 1
# source = os.environ.get("CAM_22")
# cam 2
# source = os.environ.get("CAM_23")
# video
# source = videos/2.mp4
# linha para camera 23
# limits = [200, 330, 800, 330]

x = 1
model = YOLO('models/yolov8n.pt')

filt = []

results = model.track(source, stream=True, iou=0.5, classes=vehicles_index, conf=0.6)
for r in results:
    boxes = r.boxes
    for box in boxes:
        if model.names[int(box.cls[0])] in vehicles:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                if track_id in filt:
                    continue
                else:
                    filt.append(track_id)
                im = Image.fromarray(r.orig_img)
                im.save(f'results/results{x}.jpg')
                x += 1
                print("captura realizada")
