import busid_inference as bi
import os


pd = bi.PythonPredictor(
    {'highlighter_endpoint_url': 'https://demo.highlighter.ai/graphql',
    'aws_s3_presigned_url': 'https://demo.highlighter.ai/presign',
    #'model': 's3://s3-ghawady-001/busid_model/busid_frozen_inference_graph.pb', 
    #'labelmap': 's3://s3-ghawady-001/busid_model/label_map.pbtxt',
    'gvision_apikey': '[TO-BE-ADDED]',
    'highlighter_apitoken': '[TO-BE-ADDED]',
    'min_threshold': 0.5,
    'experiment_id': 92,
    'training_run_id': 54})

print('Returned Object: ', pd.predict({"image_url":"https://i.imgur.com/j5NxlMs.jpg"}))
print('Returned Object: ', pd.predict({"image_url":"https://imgur.com/OOpUPzF.jpg"}))
print('Returned Object: ', pd.predict({"image_url":"https://imgur.com/6BvQqTa.jpg"}))
