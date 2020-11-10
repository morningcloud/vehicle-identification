# export ENDPOINT=af1b89a721aec11eab17706f2a65ca98-1632700.ap-southeast-2.elb.amazonaws.com
#
# curl -X POST -H "Content-Type: application/json" \
#   -d '{ "image_url": "https://c402277.ssl.cf1.rackcdn.com/photos/11552/images/hero_full/rsz_namibia_will_burrard_lucas_wwf_us_1.jpg?1462219623" }' \
#   http://$ENDPOINT/wps/production?debug=true
# export ENDPOINT=af1b89a721aec11eab17706f2a65ca98-1632700.ap-southeast-2.elb.amazonaws.com
#
# curl -X POST -H "Content-Type: application/json" \
#   -d '{ "image_url": "https://c402277.ssl.cf1.rackcdn.com/photos/11552/images/hero_full/rsz_namibia_will_burrard_lucas_wwf_us_1.jpg?1462219623" }' \
#   http://$ENDPOINT/wps/production?debug=true

from PIL import Image
import numpy as np
import re
import requests
import tensorflow as tf
import os
import sys
from collections import defaultdict
from io import StringIO, BytesIO

import tarfile


#import object_detection_utils as obj_utils
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

from googleapiclient.discovery import build
# Import the base64 encoding library.
import base64
import re

#sys.path.append(os.path.abspath("../../highlighter_client_python_latest"))
from highlighter_client import graphql as gql

# API Key 
DOWNLOAD_DIR = os.path.abspath("model/")
DOWNLOAD_PATH = os.path.join(DOWNLOAD_DIR, 'model.tar')
CHECKPOINT_PATH = os.path.join(DOWNLOAD_DIR, "busid_frozen_inference_graph.pb")
LABELS_PATH = os.path.join(DOWNLOAD_DIR, "bus_label_map.pbtxt")

class PythonPredictor:
  def __init__(self, config):

    # minimum score confidence threshold for accepting predictions
    self.min_threshold = config['min_threshold']

    # google vision API
    self.gvision_apikey = config["gvision_apikey"]
    self.vservice = build('vision', 'v1', developerKey=self.gvision_apikey, cache_discovery=False)

    context = gql.HighlighterContext(
        apitoken=config['highlighter_apitoken'],
        endpoint_url=config['highlighter_endpoint_url'],
        aws_s3_presigned_url=config['aws_s3_presigned_url'])

    # highlighter API for model download
    with context:
      model_file = gql.export_model_files(
        experiment_id=config['experiment_id'],
        training_run_id=config['training_run_id'],
        output_directory=DOWNLOAD_DIR)

    print("Extracting ", model_file)
    with tarfile.open(model_file) as buf:
        buf.extractall(path=DOWNLOAD_DIR)

    # initiate graph to be used for inference
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
        self.od_graph_def = tf.GraphDef()
        print("------Reading Graph File------")
        with tf.gfile.GFile(CHECKPOINT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            self.od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')

    print("-----load labelmap------", LABELS_PATH)
    label_map = label_map_util.load_labelmap(LABELS_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    print("self.category_index=",self.category_index)

    print("---------------INITIALIZING COMPLETED!---------------")

  def load_image_into_numpy_array(self, image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

  def run_inference_for_single_image(self, image):
      with self.detection_graph.as_default():
          with tf.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {
                  output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                  tensor_name = key + ':0'
                  if tensor_name in all_tensor_names:
                      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                          tensor_name)
              if 'detection_masks' in tensor_dict:
                  # The following processing is only for single image
                  detection_boxes = tf.squeeze(
                      tensor_dict['detection_boxes'], [0])
                  detection_masks = tf.squeeze(
                      tensor_dict['detection_masks'], [0])
                  # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                  real_num_detection = tf.cast(
                      tensor_dict['num_detections'][0], tf.int32)
                  detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                            real_num_detection, -1])
                  detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                            real_num_detection, -1, -1])
                  detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                      detection_masks, detection_boxes, image.shape[0], image.shape[1])
                  detection_masks_reframed = tf.cast(
                      tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                  # Follow the convention by adding back the batch dimension
                  tensor_dict['detection_masks'] = tf.expand_dims(
                      detection_masks_reframed, 0)
              image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              # Run inference
              output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})
              #print('output_dict',output_dict)
              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(
                  output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                  output_dict['detection_masks'] = output_dict['detection_masks'][0]
              
      return output_dict
  
  # Get normalized coordinates for bounding boxes in image for cropping
  def get_bounding_box_coordinates(self, image,
                                ymin, xmin, ymax, xmax,
                                use_normalized_coordinates=True):
    
    im_width, im_height = image.size
    if use_normalized_coordinates:
      (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                    int(ymin * im_height), int(ymax * im_height))
    else:
      (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    return (left, top, right, bottom)  # tuple order matters for cropping
  
  # Get base64 encoding of image data
  def encode_image(self, image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

  # Run Vision API
  def gvision_detect_text(self, img):
    request = self.vservice.images().annotate(body={
            'requests': [{
                    'image': {
                        'content': self.encode_image(img)
                    },
                    'features': [{
                        'type': 'TEXT_DETECTION',
                        'maxResults': 3
                    }]
                }],
            })
    responses = request.execute(num_retries=3)
    try:
      return responses['responses'][0]['textAnnotations'][0]['description']
    except:
      print('Error occured while fetching results from API')
      return ''

  # Process text read to fetch only the standard plate number without surrounding text
  def process_plateno_text(self, text):
    # TODO: Need enhacement, for now only identify two sets of alphanumeric with space
    m = re.search('( ?[a-zA-Z0-9]){1,9}[ -.]( ?[a-zA-Z0-9]){1,9}', text)
    if m:
        return m.group()
    return ''

  def process_busrun_text(self, text):
    # Filter only numbers
    m = re.search('( ?[0-9]){1,9}', text)
    if m:
        return m.group()
    return ''

  def predict(self, payload):
    print("---------------INITIALIZING PREDICTION---------------")
    predictions = defaultdict(list)
    
    #get image content from URL
    image_url = requests.get(payload["image_url"]).content
    image = Image.open(BytesIO(image_url))
    image_np = self.load_image_into_numpy_array(image)

    # object detection
    output_dict = self.run_inference_for_single_image(image_np)
    
    print("---------------OBJECT DETECTION INFERENCE COMPLETE---------------")
    # Read detection Results:
    # Predicted boxes for crop coordinates 
    # box coordinates (ymin, xmin, ymax, xmax) are relative to the image
    boxes = output_dict['detection_boxes']
    # Scores needed to filtered accepted objects based on threshold
    scores = output_dict['detection_scores']

    for i in range(boxes.shape[0]):
        # Filter based on threshold
        if scores is None or scores[i] > self.min_threshold:
            # boxes[i] is the box which will be drawn
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            class_name = self.category_index[output_dict['detection_classes'][i]]['name']

            bounding_box_coordinates = self.get_bounding_box_coordinates(image, ymin, xmin, ymax, xmax)

            # Crop image based on bounding box
            cropped_img = image.crop(bounding_box_coordinates)
            
            #read text from cropped image
            readings = self.gvision_detect_text(cropped_img)

            #fine tune reading with appropriate regex
            if class_name == 'PlateNo':
              readings = self.process_plateno_text(readings)
            elif class_name == 'BusRun':
              readings = self.process_busrun_text(readings)

            if readings:
              predictions['predictions'].append({'class':class_name,
                                'score':str(scores[i]),
                                'value':readings})
            else:
              print('nothing returned', readings)

    print("---------------PREDICTION COMPLETE---------------", predictions)
    return predictions #[self.format_prediction(prediction) for prediction in predictions]

  # not used at the moment
  def format_prediction(self, prediction):
    class_name, confidence = prediction.most_confident()
    x1, y1, x2, y2 = prediction.get_bounding_rectangle()
    return {
      'class': class_name,
      'confidence': confidence,
      'polygon': [ [x1, y1], [x2, y1], [x2, y2], [x1, y2] ]
    }
    