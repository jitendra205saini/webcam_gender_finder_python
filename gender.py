import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7, frame_color=(0, 255, 0)):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), frame_color, int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to the image file or 0 for webcam')
    parser.add_argument('--frame_color', type=str, default='0,255,0', help='Frame rectangle color in B,G,R format')
    parser.add_argument('--text_color', type=str, default='154,14,234', help='Text color in B,G,R format')
    args = parser.parse_args()

    faceProto = "D:/Machine larning project/gender finder/gender_model/opencv_face_detector.pbtxt"
    faceModel = "D:/Machine larning project/gender finder/gender_model/opencv_face_detector_uint8.pb"
    ageProto = "D:/Machine larning project/Age finder/age_model/age_deploy.prototxt"
    ageModel = "D:/Machine larning project/Age finder/age_model/age_net.caffemodel"
    genderProto = "D:/Machine larning project/gender finder/gender_model/gender_deploy.prototxt"
    genderModel = "D:/Machine larning project/gender finder/gender_model/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-3)', '(4-7)', '(8-12)', '(13-14)', '(15-20)', '(21-24)', '(25-32)', '(33-43)', '(44-59)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    
    frame_color = tuple(map(int, args.frame_color.split(',')))
    text_color = tuple(map(int, args.text_color.split(',')))

    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame, frame_color=frame_color)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                        min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                        :min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
