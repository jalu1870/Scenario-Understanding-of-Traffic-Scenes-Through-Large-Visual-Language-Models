# Scenario-Understanding-of-Traffic-Scenes-Through-Large-Visual-Language-Models

Evaluation code for the corresponding [paper](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/papers/Rivera_Scenario_Understanding_of_Traffic_Scenes_Through_Large_Visual_Language_Models_WACVW_2025_paper.pdf) 🔬

## image_extract

extract images from mcap files from an s3fs server using depricated_image_extract.py or the EDGAR data sdk via image_extract_sdk.py

### install

```
pip install -r requirements.txt
```

todo: make the filepaths easier adaptable for other users.

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

## 📖 How to Cite

If you use this code or data in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{rivera2025scenariounderstandingtrafficscenes,
      title={Scenario Understanding of Traffic Scenes Through Large Visual Language Models}, 
      author={Esteban Rivera and Jannik Lübberstedt and Nico Uhlemann and Markus Lienkamp},
      year={2025},
      eprint={2501.17131},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.17131}, 
}
