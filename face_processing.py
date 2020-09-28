import tensorflow as tf
from config import config
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import face_recognition as fr


def load_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def localize_face(images):
    # Down scale for faster processing
    face_boxes = []
    for image in images:
        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        boxes = fr.face_locations(small_image, number_of_times_to_upsample=3)
        if not boxes:
            boxes = fr.face_locations(small_image, model='cnn')
            if not boxes: return None
        # Get largest face
        largest_face = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        # Scale back to original
        largest_face = [i * 2 for i in largest_face]
        face_boxes.append(largest_face)
    return face_boxes


class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height or desired_face_width

    def align(self, image, face_box):
        landmarks = fr.face_landmarks(image, [face_box], 'small')
        if not landmarks:
            raise ValueError('No face found')
        landmarks = landmarks[0]
        left_eye = np.array(landmarks['left_eye']).mean(axis=0)
        right_eye = np.array(landmarks['right_eye']).mean(axis=0)
        print(left_eye, right_eye)

        d_x = right_eye[0] - left_eye[0]
        d_y = right_eye[1] - left_eye[1]
        # angle = np.degrees(np.arctan2(d_y, d_x)) - 180
        angle = np.degrees(np.arctan2(d_y, d_x))
        print(angle)

        eye_dist = np.sqrt(d_x ** 2 + d_y ** 2)
        desired_eye_dist = (1.0 - self.desired_left_eye[0] * 2) * self.desired_face_width
        scale = desired_eye_dist / eye_dist
        print(scale)

        eye_center = (right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2
        print(eye_center)

        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        t_x = self.desired_face_width * 0.5
        t_y = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (t_x - eye_center[0])
        M[1, 2] += (t_y - eye_center[1])

        output = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height), flags=cv2.INTER_CUBIC)
        return output


class FaceEncodingModel:
    model = None

    @classmethod
    def load_model(cls, path):
        return tf.keras.models.load_model(path)

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = cls.load_model(config['model_path'])
        return cls.model


def processing_inputs(face_images):
    inputs = []
    for image in face_images:
        image = image.astype(np.float32)
        image = cv2.resize(image, (112, 112))
        inputs.append(image)
    inputs = np.stack(inputs)
    inputs -= 127.5
    inputs *= 0.0078125

    return inputs


def processing_output(outputs):
    outputs = normalize(outputs)
    return outputs


