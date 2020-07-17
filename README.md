# Bus Identification Project

# Background 
The project's objective is to automate the process of collecting the replacement bus's identification details (i.e. Bus Run, Plate No) from image captures via mobile devices.

# Process Flow
Below flow diagram show the overall step involved in the identification process

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


## Highlighter Web
Following steps required for a new project to upload images and annotations (The upload part can be done either via web or highlighter_client_python)
1. Create New Object Classes (Under Resources => Object Classes)
2. Create New Data Source (Under Resources => Data Sources)
3. Create New Image Queue (Under Resources => Image Queues and add ImageFilter to the Data Source created)
4. Create New Project (Under Researches => Project and select the created Object Classes)

Following steps required to upload the model checkpoint and associated config (The upload cannot be done via web. Use model import script from highlighter_client_python):
1. Create New Research Plan (Under Resources => Research Plans)
2. Create New Experiment under the created research plan

Take notes of all the relevant ids created as they are required to be passed to highlighter_client_python

## Highlighter Python Client


## Highlighter Cortex Cluster



