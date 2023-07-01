from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
import onnxruntime

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Capture video from webcam (index 0)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype('float32')

def nms(dets, thresh=0.45):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

def detection(session, img, input_width, input_height, thresh):
    pred = []

    H, W, _ = img.shape

    data = preprocess(img, [input_width, input_height])

    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            if score > thresh:
                cls_index = np.argmax(data[5:])
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))

def generate_frames():
    while True:
        success, frame = cap.read()  # Read the next frame from the webcam
        if not success:
            break

        start = time.perf_counter()
        bboxes = detection(session, frame, input_width, input_height, 0.65)
        end = time.perf_counter()
        exec_time = (end - start)

        for b in bboxes:
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    input_width, input_height = 352, 352
    session = onnxruntime.InferenceSession('FastestDet.onnx')
    names = []
    with open("static/coco.names", 'r') as f:
        for line in f.readlines():
            names.append(line.strip())

    app.run(debug=True)