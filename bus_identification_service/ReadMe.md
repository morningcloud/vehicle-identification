# Bus Identification ML application

An ML application for detecting identification details of bus images (Bus Run, Plate Number).

HTTP POST requests containing an _image_url_ referencing an image on AWS S3.
The ML application responds with a list of _predictions_ formatted as JSON.

The ML application has been deployed to AWS Kubernetes (EKS) using the
[Cortex ML deployment tool](https://www.cortex.dev).

# BUSID ML Model

**The application uses two models:
1- A trained SSD MobileNet v2 Quantised model. This is using a pretraining model from TF Object Detection API
2- An out of the box TEXT_DETECTION model via Google Vision API
**

The Cortex application deployment file (cortex.yaml) specifies the URL that
refers to the ML Model archive file via _predictor: model:_.

# Development

Cortex provides an adapter between HTTP POST requests and a couple of Python
functions.  The Cortex application deployment file (cortex.yaml) specifies
the Python script name via _predictor: path: busid_inference.py.

The Python script _busid_inference.py_ requires two functions to be defined ...

- init() which sets up the ML Model
- predict() which is invoked on each HTTP POST



## Development steps

- Download and install the [Cortex ML deployment tool](https://www.cortex.dev)
- Git clone this repository
- Update the Cortex cluster configuration file (cluster.yaml)
- Update the Cortex application deployment file (cortex.yaml)
- Acquire Silverpond GitLab _read_repository_ access tokens for the requirements file
- Update the Cortex application requirements file (requirements.txt)
- Implement _busid_inference.py_

```
bash -c "$(curl -sS https://raw.githubusercontent.com/cortexlabs/cortex/0.13/get-cli.sh)"
git clone <URL for bus-identification-service repository>
cd bus_identification_service
vi cluster.yaml
vi cortex.yaml
vi requirements.txt
vi busid_inference.py
```

# Deployment

## Deployment preparation

- Download and install the [Cortex ML deployment tool](https://www.cortex.dev)
- GIT clone the [service repository]
- Set up ~/.aws/credentials using the "Silverbrane" AWS account


Note: Deployment steps assume that _bus_identification_service_
is the current working directory.

## Starting the AWS Kubernetes (EKS) cluster

```
cortex cluster up --config=cluster.yaml    # takes around 15 minutes
```

```
cortex cluster info --config=cluster.yaml
```

## Deploying the busid ML application

```
cortex deploy
cortex get --watch  # monitor progress
cortex get busid
```

## Redeploying the busid ML application

```
cortex deploy --force  # recommended option
cortex get --watch     # monitor progress
```

## Taking down the busid ML application

```
cortex delete busid
```

## Stopping the AWS Kubernetes (EKS) cluster

```
cortex cluster down --config=cluster.yaml
```

# Monitoring

The deployment status can be continuously monitored using the following
command, which lists the _api name_ (production), _status_, _request count_
and _time since last update_.

```
cortex get --watch
```

# Diagnosis

Application logs can be acquired as follows, which is useful if monitoring
indicates a _status error_ or the application isn't functioning as expected.

```
cortex logs busid
```
