#!/usr/bin/env python

import os
import joblib
import argparse
from PIL import Image
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
from face_recognition import preprocessing
from PIL import ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for detecting and classifying faces on user-provided image. This script will process image, draw '
        'bounding boxes and labels on image and display it. It will also optionally save that image.')
    parser.add_argument('--image-path' , required=True, help='Path to image file.')
    parser.add_argument('--save-dir', help='If save dir is provided image will be saved to specified directory.')
    return parser.parse_args()


def recognise_faces(img):
    faces = joblib.load(MODEL_PATH)(img)
    print ((faces))
    if faces :
        draw_bb_on_img(faces, img)
    return faces, img


def main():
    args = parse_args()
    #print (type (args))
    preprocess = preprocessing.ExifOrientationNormalize()
    img = Image.open(args.image_path)
    filename = img.filename
    img = preprocess(img)
    img = img.convert('RGB')

    faces, img = recognise_faces(img)
    for face in faces:
        top_prediction = face.top_prediction
        bb = face.bb
        all_predictions = face.all_predictions

        #print("Top prediction:", top_prediction.label)
        print("Confidence:", top_prediction.confidence)
        #print("BoundingBox Left:", bb.left)
        #print("BoundingBox Top:", bb.top)
        #print("BoundingBox Right:", bb.right)
        #print("BoundingBox Bottom:", bb.bottom)
        #print("All predictions:")
        for prediction in all_predictions:
            print("\tLabel:", prediction.label)
            print("\tConfidence:", prediction.confidence)

        print("====================================")
    if not faces:
        print('No faces found in this image.')

    if args.save_dir:
        basename = os.path.basename(filename)
        name = basename.split('.')[0]
        ext = basename.split('.')[1]
        img.save('{}_tagged.{}'.format(name, ext))

    img.show()


if __name__ == '__main__':
    main()
