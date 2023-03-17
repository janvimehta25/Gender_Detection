import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("https://media.istockphoto.com/id/1135173484/vector/human-polygonal-face.jpg?s=612x612&w=0&k=20&c=7D7SOSZJF10bEm74oSyDSWZQwdzgAT3lf2WaSO1mxc4=");
                background-attachment: fixed;
                background-size: cover
            }}
            .sidebar .sidebar-content {{
                background-color= #D6EAF8
            }}
        </style>
        """,
        unsafe_allow_html = True
    )
   


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Detect face
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    # Draw rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (250,0,0),2)
    return img,faces




def main():
    add_bg_from_url()
    st.title("Face Detection App")
    #st.text("Build with Steamlit and OPencv")

    activities = ["Detection","Gender", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        #st.subheader("Face Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])#input

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text ("Original Image")
            st.image(our_image)

        enhance_type = st.sidebar.selectbox("Enhance Type", ["Original","Gray-Scale"])
        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.image(gray)
        
        # Face Detection
        task= ["Faces"]
        feature_choice = st.sidebar.radio("Find Features",task)
        if st.button("Process"):
             
            if feature_choice == "Faces":
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)

                st.success("Found {} faces".format(len(result_faces)))
    
    
    elif choice == 'About' :
        st.subheader("About")
        st.text("Built with Streamlit and OpenCv ")
        st.text("It takes jpeg , png and jpg files as input and converts them to grayscale")
        st.text("Finally it detects only the human face and not cartoons or distored images")

    

    elif choice =='Gender Detection' :
         #image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])#input1

         #if image_file is not None:
         #   our_image = Image.open(image_file)
         #   st.text ("Original Image")
         #   st.image(our_image)
         #st.subheader("Gender Detection")
         #enhance_type = st.sidebar.selectbox("Enhance Type", ["Original","Gray-Scale"])
         #if enhance_type == 'Gray-Scale':
         #   new_img = np.array(our_image.convert('RGB'))
         #   img = cv2.cvtColor(new_img,1)
         #   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.image(gray)   
    
def get_face_box(net, frame, conf_threshold=0.7):

        opencv_dnn_frame = frame.copy()
        frame_height = opencv_dnn_frame.shape[0]
        frame_width = opencv_dnn_frame.shape[1]
        blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob_img)
        detections = net.forward()
        b_boxes_detect = []
        for i in range(detections.shape[2]):
           Probability = detections[0, 0, i, 2]
           if Probability > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
        return opencv_dnn_frame, b_boxes_detect

#heading
st.write("""
    # Gender prediction
    """)
#title
st.write("## Upload a picture that contains a face")

uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
    cap=cv2.imread('temp.jpg')

    face_txt_path="opencv_face_detector.pbtxt"
    face_model_path="opencv_face_detector_uint8.pb"

    #age_txt_path="age_deploy.prototxt"
    #age_model_path="age_net.caffemodel"

    gender_txt_path="gender_deploy.prototxt"
    gender_model_path="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    #age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
    #              'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
    gender_classes = ['Male', 'Female']

    #age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
    gender_net = cv2.dnn.readNet( gender_model_path,gender_txt_path )
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    padding = 20

    t = time.time()
    frameFace, b_boxes = get_face_box(face_net, cap)
    if not b_boxes:
        st.write("No face Detected, Checking next frame")

    for bbox in b_boxes:
        face = cap[max(0, bbox[1] -padding): min(bbox[3] + padding, cap.shape[0] -1), max(0, bbox[0] -padding): min(bbox[2] +padding, cap.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        #Probability meter show
        st.write(
            f"Gender : {gender}, Probability = {gender_pred_list[0].max() }")

        #age_net.setInput(blob)
        #age_pred_list = age_net.forward()
        #age = age_classes[age_pred_list[0].argmax()]
        #st.write(f"Age : {age}, Probability = {age_pred_list[0].max() * 100}%")

        label = "{}".format(gender)
        cv2.putText(
            frameFace,
            label,
            (bbox[0],
             bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,
             255,
             255),
            2,
            cv2.LINE_AA)
        st.image(frameFace)

        #elif choice == 'About' :

    
           #st.subheader("About")
           #st.text("Built with Streamlit and OpenCv ")
           #st.text("It takes jpeg , png and jpg files as input and converts them to grayscale")
           #st.text("Finally it detects only the human face and not cartoons or distored images")



                   
    

if __name__ == '__main__':
        main()