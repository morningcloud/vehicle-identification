# Bus Identification Project

# Background 
The project's objective is to automate the process of collecting the replacement bus's identification details (i.e. Bus Run, Plate No) from image captures via mobile devices.

# Process Flow
Below flow diagram show the overall steps involved in the identification process

![Alt text](doc/BusIDFlow.png?raw=true "Bus Identification Process Flow")

Two models are utilised as part of the flow:

1. Object Detection

We used custom images annotated in Pascal VOC format and experimented with two models:

- TF Object Detection model: Available in [TF bus identification.ipynb](https://gitlab.com/silverpond/research/application/bus-identification/-/blob/master/TF%20bus%20identification.ipynb)
- mmdetection v2.2.1: Available in [mmdetection bus identification](https://gitlab.com/silverpond/research/application/bus-identification/-/blob/master/mmdetection%20bus%20identification.ipynb)

Note: The TF Object Detection model is converted to TFlite in order to support being run on mobile

2. Text Recognition

This is done via out of the box google vision 'TEXT_DETECTION' api.
The output of the Google Vision API is processed with regex depending on the class type (i.e. Plate No or Bus ID). This adds as another layer to filter only valid sequences per expected object type.

# Documents
- [Handover Summary Doc](https://docs.google.com/document/d/1Hv95ZnxsNoY3RQTt3Ju-paVXvyPT9_uQAkjMPl5wBN0/edit)
- [R&D Tax Concession LXRP](https://drive.google.com/file/d/1Wl9x5uYjHwK8F9rpUMGezQj6QpF9Qco-/view?usp=sharing)

# Steps to replicate End to end process

## Components involved
1. Model Training (available in this repo)
2. [Highlighter Web](https://highlighter.ai/dashboard) (To create Training Run for checkpoint upload)
3. [Highlighter Client Python](https://gitlab.com/silverpond/products/highlighter/highlighter_client_python) (CLI interface to import model to HL)
4. [Highlighter Cortex Cluster](https://gitlab.com/silverpond/infrastructure/highlighter-cortex-cluster/-/tree/busid) (For model deployment for interence)

## Model Training
We used colab environment for the object detection model training, the notebooks for each contains detailed documentation and explanation of prerequisits and steps required:
- [TF bus identification.ipynb](https://gitlab.com/silverpond/research/application/bus-identification/-/blob/master/TF%20bus%20identification.ipynb)
- [mmdetection bus identification.ipynb](https://gitlab.com/silverpond/research/application/bus-identification/-/blob/master/mmdetection%20bus%20identification.ipynb)

## Highlighter Web
Following steps required for a new project to upload images and annotations (The upload part can be done either via web or highlighter_client_python)
1. Create New Object Classes (Under Resources => Object Classes)
2. Create New Data Source (Under Resources => Data Sources)
3. Create New Image Queue (Under Resources => Image Queues and add ImageFilter to the Data Source created)
4. Create New Project (Under Researches => Project and select the created Object Classes)

Following steps required to upload the model checkpoint and associated config (The upload cannot be done via web. Use model import script from highlighter_client_python):
1. Create New Research Plan (Under Resources => Research Plans)
2. Create New Experiment under the created research plan
3. Create New Model entry (This highlighter requires admin rights)

Take notes of all the relevant ids created as they are required to be passed to highlighter_client_python

## Highlighter Client Python
Following are the scripts used from Highlighter Client Python. Please refer to the latest [repo](https://gitlab.com/silverpond/products/highlighter/highlighter_client_python) for installation instructions and latest updates.
1. Dateset Import to Highlighter
Eventhough the model is not trained using highlighter, it was handy to import the dataset to highlighter to be able to export them in another format, this was used while attempting to train the object detection model using mmpond

```
highlighter dataset import ../bus-identification/Data/AllImages/ --format=pascalvoc --data-source-id=1451 --user-id=457 --project-id=477
```

2. Model Import
Through this script the model files and log files (if any) will be uploaded to S3 pucket and referenced in highlighter under the specified research plan and experiment (The ids that were created in [Highlighter Web](#highlighter-web))

```
highlighter model import-training-run 
--repo-url=https://gitlab.com/silverpond/research/application/bus-identification.git 
--repo-commit-hash=db1865d4b0312ada31d9cc89e99d9aff8ba16463
--project-id=477 \
--research-plan-id=37 \
--experiment-id=92 \
--model-id=30 \
--run-name=TFSSD-run \
--model-files=../bus-identification/ExperimentFiles/TFSSD/Model/busid_frozen_inference_graph.pb \
--model-files=../bus-identification/ExperimentFiles/TFSSD/Model/bus_label_map.pbtxt \
--log-files=../bus-identification/ExperimentFiles/TFSSD/events.out.tfevents.1588844993.ee40f3c056ac
```

```
highlighter model import-training-run 
--repo-url=https://gitlab.com/silverpond/research/application/bus-identification.git \
--repo-commit-hash=49dcf86b133f2e39b74f0504699b860469b1d112 \
--project-id=477 \
--research-plan-id=37 \
--experiment-id=153 \
--model-id=30 \
--run-name=mmdet-run \
--model-files=../bus-identification/ExperimentFiles/mmdet/faster_rcnn_r50_fpn_1x_voc0712.py \
--model-files=../bus-identification/ExperimentFiles/mmdet/busid_latest.pth \
--log-files=../bus-identification/ExperimentFiles/mmdet/events.out.tfevents.1594725086.a166ce7d8c52.1125.0
```

3. Model Export
This script is used to download the model files from highlighter. However this is not required as the export is done as part of the cortex application deployment
```
highlighter export_model_files --training-run-id=55
```

## Highlighter Cortex Cluster

Cortex is used to deploy and serve the bus identification model implements the steps mentioned in the [process flow diagram]](#process-flow)

Following is a summary of steps to do the detailed installation steps are found [here](https://gitlab.com/silverpond/infrastructure/highlighter-cortex-cluster/-/blob/busid/busid/ReadMe.md)
1. Get latest source from [here](https://gitlab.com/silverpond/infrastructure/highlighter-cortex-cluster/-/tree/busid)
2. Update following config in busid/cortex.yaml
    - highlighter_endpoint_url
    - aws_s3_presigned_url
    - highlighter_apitoken 
    - [gvision_apikey](https://cloud.google.com/vision/docs/setup)
    - training_run_id 


3. Run Following
```
cd highlighter-cortex-cluster
cortex deploy busid/cortex.yaml
cortex logs busid
cortex get busid
```
4. Once the deployment is complete, can invoke the inference model via http and passing an image URL as follows
```
curl http://localhost:8888 -X POST -H "Content-Type: application/json" -d '{"image_url":"https://imgur.com/6BvQqTa.jpg"}'
```
Expected Response
```
{"predictions": [{"class": "BusRun", "score": 0.9983952045440674, "value": "30"}, {"class": "PlateNo", "score": 0.9890475273132324, "value": "10V 4VX"}]}
```
