from flask import Flask, request
from flask_restful import Resource, Api
from config import config
import os
import cv2

from face_processing import FaceAligner, FaceEncodingModel, load_images, localize_face, processing_inputs, \
    processing_output

app = Flask(__name__)
api = Api(app)


class FaceEncodingService(Resource):
    def __init__(self):
        self.aligner = FaceAligner(desired_left_eye=config['desired_left_eye'],
                                   desired_face_width=config['desired_face_width'],
                                   desired_face_height=config['desired_face_height'])
        self.encoder = FaceEncodingModel.get_model()

    def post(self):
        data = request.json
        image_paths = data['images']
        if 'rootPath' in data:
            image_paths = [os.path.join(data['rootPath'], path) for path in image_paths]
        images = load_images(image_paths)
        face_boxes = localize_face(images)
        if face_boxes is None:
            return {
                'error': 'Missing face'
            }
        aligned_faces = []
        for i, (image, box) in enumerate(zip(images, face_boxes)):
            face = self.aligner.align(image, box)
            aligned_faces.append(face)

            cv2.imwrite('requests/{}.jpg'.format(os.path.basename(data['images'][i])), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        inputs = processing_inputs(aligned_faces)
        encodings = self.encoder.predict(inputs)
        encodings = processing_output(encodings)
        return {
            'rootPath': data.get('rootPath', ''),
            'results': [
                {'image': image, 'encoding': encoding} for image, encoding in zip(data['images'], encodings.tolist())
            ]
        }


api.add_resource(FaceEncodingService, '/face_encoding')

if __name__ == '__main__':
    app.run(host=config['host'], port=config['port'])