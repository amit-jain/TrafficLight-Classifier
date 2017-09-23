## TrafficLight-Classifier

The repository contains the classifier code for Udacity's capstone project for the team in the repo
[https://github.com/hulleywood/CarND-Capstone].

### Data
The dataset used for the project is available at [capstone-tl.zip](https://drive.google.com/open?id=0B9SivYviVPsONVByTHZsbTBjR28).
The dataset is a mixture of the simulator and the real-world site images. The images are divided into 4 categories:
* 0 - Unknown
* 1 - Red
* 2 - Orange/Yellow
* 3 - Green

There is a `simu` folder in each of the folders which contain the simulator images

### Model
The model used in based on the Squeezenet model pre-trained on the imagenet dataset (https://github.com/rcmalli/keras-squeezenet). The top few layers have been replaced by new layers and since the dataset we had was small and different 2 models for site and simulator images was trained end-to end.
The model and training is implemented in the `SqueezenetTrafficLightModel.py`. A softmax output was used to have a binary classifier which would classify images with 0 -UNKNOWN and 1 - RED traffic light state since, we were working with a smaller dataset. Some future improvements can include 
* Using other datasets specially the Bosch small traffic light dataset
* Creating a unified classifier to work on both the simulator and real world images

#### Preparation
There are some pre-processing steps on the images before the training
* Resize the images to (224, 224, 3) for simulator and (320, 320, 3) for the site images.
* Histogram equalization (for site images)
* Gamma correction (for site images)

#### Training
The model can be trained by running the following command:

`python SqueezenetTrafficLightModel.py -i <inputfolder> -o <output> -s <shape> " +
              "-t <dataset type(simulator, site)> -e <epochs> -d <dropout> -l <learning rate> " +
              "-v <validation ratio> -p <pre-process>`

For example the final model trained on site images was which also highlights the parameters used:

``python SqueezenetTrafficLightModel.py -i images -o squeezenet -s "(320, 320, 3)" -t site -e 50 -d .35 -l .0001 -v 0.30 -p true``

#### Optimization for inference
The model obtained was optimized by first converting the keras model to a tf model and the optimizing the model. The following steps were executed:
* `KerasTFConverter`
* `~/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=classifier_site_raw.pb \
--out_graph=classifier_site.pb \
--inputs='input_1' \
--outputs='output_node0' \
--transforms='strip_unused_nodes(type=float, shape="1,224,224,3") remove_nodes(op=Identity, op=CheckNumerics) round_weights(num_steps=256) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'
`
Note that this requires building tensorflow for the tools used.

### Inference

Class `Inference.py` can be used for to infer images to test.