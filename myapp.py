from cmath import nan
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time



def load_image(image_file):
	img = Image.open(image_file)
	return img

labels_new = ["yawn", "no_yawn"]
IMG_SIZE = 145
def prepare(image_array, face_cas_path="haarcascade_frontalface_default.xml"):
#     image_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
#     im=Image.open(filepath)
#     image_array = np.array(im)
#     image_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+face_cas_path)
    faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3), [x,y,w,h]
    return 0, []

st.markdown(""" <style> .font {
font-size:50px ; color: red;} 
</style> """, unsafe_allow_html=True)

model = load_model("model.h5")

# st.subheader("Image")
# image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

# if image_file is not None:
#         image = Image.open(image_file)
#         img_array = np.array(image)
#         # To See details
#         file_details = {"filename":image_file.name, "filetype":image_file.type,
#                         "filesize":image_file.size}

#         # To View Uploaded Image
#         st.image(load_image(image_file),width=250)
        
        


# if st.button('Analyse'):
#         face = [prepare(img_array)]
#         if np.array(face).any():
#                 prediction = model.predict(face)
#                 st.markdown((prediction), unsafe_allow_html=True)
#                 st.markdown(np.argmax(prediction), unsafe_allow_html=True)
#                 if np.argmax(prediction) == 0:
#                         st.markdown('<strong class="font">Drowsiness Alert!!</strong>', unsafe_allow_html=True)
#                 else:
#                         st.markdown('<strong class="font">Normal</strong>', unsafe_allow_html=True)
                
#         else:
#                 st.markdown('no face', unsafe_allow_html=True)

# define a video capture object
vid = cv2.VideoCapture(0)
result = st.empty()
pre = st.empty()
if st.button('Start'):
    while(True):
        time.sleep(0.5)
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if not ret:
                continue
        
        
        image = np.array(frame)
        face = [prepare(image)[0]]
        location = prepare(image)[1]
        if np.array(face).any():
                prediction = model.predict(face)
                # st.markdown((prediction), unsafe_allow_html=True)
                # st.markdown(np.argmax(prediction), unsafe_allow_html=True)
                pre.markdown(prediction)
                if np.argmax(prediction) == 0:
                        result.markdown("drowsy")
                else:
                        result.markdown("normal")
                        
        else:
                result.markdown("noface")
        # Display the resulting frame
        if location == []:
                cv2.imshow('frame', frame)
        else:
                cv2.rectangle(frame, (location[0], location[1]), (location[0] + location[2], location[1] + location[3]), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()