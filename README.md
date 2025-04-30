# Scenario-Understanding-of-Traffic-Scenes-Through-Large-Visual-Language-Models

Evaluation code for the corresponding paper 🔬 [arXiv Paper](https://arxiv.org/abs/2501.17131)

This is the complete code, including evaluation and code for models not used in inference


## image_extract.py

### install

```
pip install -r requirements.txt
```

extracts images from mcap file located on the minio server.
todo: make the filepaths easier adaptable for other users.

## anonymizer

anonymization of license plates and persons 
for installation see the corresponding readme file

```
python dashcamcleaner/cli.py --input_path <path_to_input_image_directory> --output_path <path_to_output_image_directory>
```


## base
predict the tags for images.

### install
tested for python version 3.10.12
```
pip install -r requirements.txt
```
### prediction
```
python main.py <arg1> <arg2> ...

arguments:
    --config: path to configuration file
    --task: whether to execute prediction or evaluation
    --model: which model to use, only required for prediction
    --prompt: the prompt to use when predicting the image content, only required for prediction
    --input_path: parent directory of the images, searches images recursively, only required for prediction
    --output_path: path to save prediction at, only required for prediction
    --label_path: the path of the saved ground truth labels, only required for evaluation
    --prediction_path: the path + name of the prediction to evaluate (without fileending), only required for evaluation
```
