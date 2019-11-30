import cv2

face_img = cv2.imread('./test1.jpg')

# cv face detection.
# caution!! downloaded xml from github  (about 8 MB) is invalid.
# face_cascade = cv2.CascadeClassifier('/home/albert/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5,5)
)

print('detect result: {}'.format(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

face_img = cv2.resize(face_img, (int(face_img.shape[1] / 2), int(face_img.shape[0] / 2)))
cv2.imshow('detect_result', face_img)
cv2.waitKey(0)