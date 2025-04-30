import configparser
from gpt4 import GPT4sender
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import argparse
from tqdm import tqdm
import LLaVA
from LLaVA.LLaVA import LLaVA, internlm, deepseek, cog, blip
import json
import ast
import util as util
from clip import CLIP
import numpy as np
import torch
# from Yi.VL.YI import YI
# import time
from similarity import Classifier
import pickle
from natsort import natsorted

from sklearn.metrics import f1_score
print(os.getcwd())
print(torch.cuda.is_available())
sys.path.append('./LLaVA/')

models = {
    "gpt4": GPT4sender,#
    "llava-1.6_7m":LLaVA,#
    "llava-1.6_7v": LLaVA,#
    "llava-1.5": LLaVA,#
    "llava-1.6_13":LLaVA,#
    "llava-1.6_34":LLaVA,#
    "clip": CLIP,
    # "yi-6":YI,
    # "yi-34":YI,
    "composer-vl":internlm,#
    "composer-hd":internlm,#
    "composer-hd-336":internlm,#
    "deepseek":deepseek,
    "cogvlm":cog,#
    "cogagent":cog,#
    "cogagent-vqa":cog,#
    "blip-caption-large":blip,
    "blip-itm-coco":blip,
    "blip-capfilt-large":blip,
    "blip2-opt":blip,#
    "blip2-t5-xxl":blip,#
    "instructblip-13b-v":blip#
}
image_extensions = ["jpg","jpeg","png"]

#debug
track_name = "newpp.txt"

def neural_similarity(outputs, options, model):
    """
    Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

    Args:
        outputs (string): model output.
        options (list,string): possible tags.

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """
    result = model.run(outputs,options)
    category, option = result.split(" : ")
    return option

def predict(config,model,save_result=True):
    """
    Calculates predictions for all the images in the directory specified in config["General"]["input_path"] with the model config["General"]["model"].
    Predictions are saved to config["General"]["output_path"].
    Saves a txt file containing the config of the prediction run and a .pkl fille containing the scores and string output of the model

    Args:
        config (dict): configuration.
        save_result (bool): if the predictions should be saveed.

    Returns:
        None

    """

    #if input path is a directory, read all files. If it is a file only use this image
    input_dir = config["General"]["input_path"]
    print(input_dir)
    input_dir = "/app/data/tmp/"
    if(os.path.isdir(input_dir)):
        image_paths_ = util.get_all_files_in_directory(input_dir)
    else:
        print("i was here")
        image_paths_ = [input_dir]
        image_paths_ = util.get_all_files_in_directory(input_dir)
    
    #check if file extension is correct
    image_paths = [image_path for image_path in image_paths_ if image_path.split(".")[-1] in image_extensions]
    print("image_paths:")
    print(image_paths_)
    if(len(image_paths_) > len(image_paths)):
        print("Warning: only {} as image types supported".format(', '.join(image_extensions)))


    prompt = config["General"]["prompt"]


    #if save_result saves predictions to a json file, containing metainformation under "config", predictions for each image with the image as key under "prediction"
    result_dict_json = {"config":{},"prediction":{}}
    if(save_result):
        output_dir = config["General"]["output_path"]
        output_path = output_dir + "/predictions_0.json"
        if(output_dir == "/-1"):
            print("--output_path needs to be specified to save predictions")
            sys.exit()

        #find prediction index not occupied so far, create directories
        if(os.path.exists(output_dir)):
            output_ind = 0
            while(os.path.exists(output_path) or output_ind < 1100):
                output_path = "{}/predictions_{}.json".format(output_dir,output_ind)
                output_ind += 1
        print("predictions will be saved to {}".format(output_path))
        os.makedirs("." + output_path.replace(output_path.split(".")[-1],""), exist_ok=True)

        #copy config to dictionary
        for section in config.sections():
            for option in config.options(section):
                value = config[section][option]
                result_dict_json["config"][option] = value
            
        #save json
        with open(output_path, 'w') as json_file:
            json.dump(result_dict_json, json_file, indent=4)

    results = {}
    print(len(image_paths))
    for image_path in tqdm(image_paths):

        prompt_ = prompt.split(" ; ")[0] #seperate input prompt from categoryname

        #clip expects list of options as input, convert this list in stringform to list
        if(config["General"]["model"] == "clip"):
            prompt_ = ast.literal_eval(prompt_)
            
        
        result = model.run(image_path,prompt_) 

        # if save_result saves model output text to a json file if scores are not saved, else predictions are saved in a dictionary
        # keys: 
        #   -"scores": scores for each predicted token
        #   -"tokens": total number of tokens
        #   -"text": text of the model output
        if(save_result):
            if("llava" in config["General"]["model"] or "yi" in config["General"]["model"]):
                results[image_path] = {"scores": torch.nn.functional.softmax(result[0]["scores"][0],dim=-1).to(torch.float16).cpu().numpy(), "tokens": result[0]["sequences"].to(torch.float16).cpu().numpy(), "text": result[1]}
            elif(config["General"]["model"] == "clip"):
                results[image_path] = {"prompt": prompt, "scores": result[0].detach().cpu().numpy()}
            else:
                result_dict_json["prediction"][image_path] = result
                with open(output_path, 'w') as json_file:
                    json.dump(result_dict_json, json_file, indent=4)
                # with open(output_path,"a") as f:
                #     f.write("{} = {}\n".format(image_path,result))

    #saves scores to pkl files for models that return scores
    if(save_result and "llava" in config["General"]["model"] or config["General"]["model"] == "clip" or "yi" in config["General"]["model"]):
        
        results_path = output_path.replace(".txt",".pkl").replace(".json",".pkl")
        import pickle
        with open(results_path, 'wb') as file:
            # Use pickle.dump() to store the data in the file
            pickle.dump(results, file)
            
def create_config(config_file):
    
    """
    creates the base config file at the location config_file.

    Args:
        config_file (string): path where t save the config file.

    Returns:
        None.

    """
    config = configparser.ConfigParser()

    # Add sections and options to the configuration file
    config["General"] = {"model": "llava", "input_path": "/-1", "output_path": "/-1", "prompt": "dummy", "label_path": "labels.txt", "prediction_path":"/-1"}


    # Write the configuration to the file
    with open(config_file, 'w') as configfile:
        config.write(configfile)

def read_config(args):
    """
    reads the arguments specified in args and overwrites with those specified the current config, makes sure to use the command line arguments over those in the config file.

    Args:
        args (string): command line arguments.

    Returns:
        config: modified config.

    """
    config = configparser.ConfigParser()
    config.read(args.config)

    #overwrite config returned from file with console arguments
    for key, value in vars(args).items():
        config["General"][key] = value if value is not None else config["General"][key]
    return config

def parse_command_line_args():
    """
    parses the arguments specified in the command line.
        options: 
            --config: path to configuration file
            --task: whether to execute prediction or evaluation
            --model: which model to use, only required for prediction
            --prompt: the prompt to use when predicting the image content, only required for prediction
            --input_path: parent directory of the images, searches images recursively, only required for prediction
            --output_path: path to save prediction at, only required for prediction
            --label_path: the path of the saved ground truth labels, only required for evaluation
            --prediction_path: the path + name of the prediction to evaluate (without fileending), only required for evaluation

    Returns:
        args: the parsed command line arguments.

    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='Specify the path to the configuration file', default='conf/config.ini')
    parser.add_argument('--task',"-t", help='define the task to be executed', required=False, choices=["predict","evaluate"],metavar="task")
    parser.add_argument('--model',"-m", help='model to use',metavar="model")
    parser.add_argument('--prompt',"-p", help='prompt to use in prediction',metavar="prompt")
    parser.add_argument('--input_path',"-i", help='path of a input image or a directory containing input images',metavar="input_path")
    parser.add_argument('--output_path',"-o", help='path to save predictions to',metavar="output_path")
    parser.add_argument('--label_path',"-l", help='path to load labels from',metavar="label_path")
    parser.add_argument('--prediction_path',"-pred", help='path to load predictions from',metavar="prediction_path")

    args = parser.parse_args()
    return args

def correct_tag(pred_text,category):
    """
    uses manual rules to correct the model prediction to fit the possible tags of the given category

    Args:
        pred_text (string): model output.
        category (string): category name.

    Returns:
        string: the modified model output to match the possible tags.

    """
    
    if(category == "number_of_lanes"):
        if(pred_text == "zero"):
            pred_text = "0"
        elif(pred_text == "one"):
            pred_text = "1"
        elif(pred_text == "two"):
            pred_text = "2"
        elif(pred_text == "three"):
            pred_text = "3"
        elif(pred_text == "four"):
            pred_text = "4"
        elif(pred_text == "five"):
            pred_text = "5"
        elif(pred_text == "six"):
            pred_text = "6"
    elif(category == 'number_of_vulnerable_road_users'):
        if("no" in pred_text or pred_text == "0"):
            pred_text = "none"
        elif(pred_text == "1" or pred_text == "2" or "one" in pred_text or "two" in pred_text):
            pred_text = "few"
    elif(category == 'land_use'):
        if(pred_text == "urban"):
            pred_text = "urban area"
        elif(pred_text == "suburban"):
            pred_text = "suburban area"
        elif(pred_text == "industrial"):
            pred_text = "industrial area"
        elif(pred_text == "rural"):
            pred_text = "rural area"
    elif(category == 'road_condition'):
        if(pred_text == "dry"):
            pred_text = "dry road"
        elif(pred_text == "wet"):
            pred_text = "wet road"
        elif(pred_text == "snowy"):
            pred_text = "snowy road"
        elif(pred_text == "muddy"):
            pred_text = "muddy road"
    elif(category == 'street_configuration'):
        if(pred_text == "1" or pred_text == "one" or "one-directional" in pred_text or "1-directional" in pred_text or 
           "one-way street" in pred_text or pred_text == "1 way" or pred_text == "1-way" or pred_text == "one-way"  or pred_text == "one way" or pred_text == "unidirectional"):
            pred_text = "one way street"
        elif(pred_text == "2" or "two-directional" in pred_text or "2-directional" in pred_text or "bi-directional" in pred_text or 
             "two-way street" in pred_text or pred_text == "2 way" or pred_text == "2-way" or pred_text == "two-way"  or pred_text == "two way"):
            pred_text = "two way street"
    elif(category == 'urban_environment'):
        if(pred_text == "gas station"):
            pred_text = "gas stations"
        elif("highway" in pred_text):
            pred_text = "highway"
        elif("highway" in pred_text):
            pred_text = "highway"
        elif(pred_text == "residential area"):
            pred_text = "residential"# area"
    elif(category == 'lanemarks'):
        if("normal" in pred_text):
            pred_text = "normal lanemarks"
        elif("no " in pred_text):
            pred_text = "no lanemarks"
        elif("crosswalk" in pred_text):
            pred_text = "crosswalk"
        elif("bus" in pred_text):
            pred_text = "bus lane"
        elif("no" in pred_text):
            pred_text = "no lanemarks"
        elif("yes" == pred_text):
            pred_text = "normal lanemarks"
    elif(category == 'traffic_scene'):
        if("free-flowing" in pred_text or "free flowing" in pred_text):
            pred_text = "free-flowing traffic"
        elif("congested" in pred_text):
            pred_text = "congested traffic"
        elif("accident" in pred_text):
            pred_text = "traffic accident"
        elif("construction zone" in pred_text):
            pred_text = "construction zone"
    elif(category == 'vehicle_maneuver'):
        if("stopping" in pred_text or pred_text == "stop"):
            pred_text = "stopping"
        elif("forward" in pred_text):
            pred_text = "moving forward"
        elif("lane chang" in pred_text):
            pred_text = "lane changing"
        elif(pred_text == "turn"):
            pred_text = "turning"
            pred_text = "lane changing"
        elif(pred_text == "park"):
            pred_text = "parking"
    elif("yes" in pred_text):
        pred_text = "yes"
    elif("no" in pred_text and category != 'lanemarks' and category != 'number_of_vulnerable_road_users' and category != 'weather'):
        pred_text = "no"

    return pred_text

def convert_gpt4_result_to_label(result_path,category_dict,zero_shot_similarity):
    """
    given the result file of a previous run with gpt4 as the model to predict the image tags, converts the prediction to a format that is easily comparable to the ground truth.

    Args:
        result_path (string): path to model output.
        category_dict (dictionary): possible tags.
        zero_shot_similarity (bool): whether to use neural similarity between model output and the possible tags. Maps model ouput to the tag with the highest embedding similarity. 
                    else: uses manual rules for the mapping

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.
    """

    #load prediction file
    data = {}
    with open(result_path + ".json", 'r') as file:
        data = json.load(file)

    #define zero shot classifier for predictions not matchable with manual rules
    if(zero_shot_similarity == "True"):
        model = Classifier()
        


    #list of strings possibly returned by gpt-4, which are not considered as prediction
    cont_list = ["",",",", ","[","]"]

    choices = {}
    category_keys = list(category_dict.keys())

    #iterate over the prediction per image
    pred_dict = {}
    for preds_per_file in data["prediction"]:
        if(preds_per_file == "\n"):
            continue

        filename = preds_per_file.split("/")[-1]

        #get prediction per image
        content = data["prediction"][preds_per_file]["choices"][0]["message"]["content"].lower()
        
        #one category prediction per line, containing potentially key(categoryname): value(predicted tagname) pairs
        pairs = content.split("\n")

        pred_dict[filename] = {}
        
        count = 0 
        
        #warning if less predictions than categories defined in labels
        if(len(pairs) < len(category_dict)):
            print("not all results given {}".format(filename))

        for pair in pairs:
            if(pair in cont_list):
                continue
            
            #split potential key, value pairs
            if(":" in pair):
                pair = pair.replace(":","?")
            if("?" in pair):
                category, pred_text = pair.split("?")
            else:
                pred_text = pair
                
            category_name = category_keys[count]
            
            #corrects returned prediction to match defined tags with manual rules
            pred_text = correct_tag(pred_text,category_name)
            pred_text = pred_text.strip().replace("'","")

            # if prediction matches a defined tag, set prediction for the category
            # else check if a defined tag is contained inside the prediction
            #     if not use zero shot classifier to assign tag with the highest similarity
            if(pred_text in category_dict[category_name]):
                tag_index = category_dict[category_name].index(pred_text)
                pred_dict[filename][category_name] = tag_index
            else:
                tag_inc = False
                for category_tag in category_dict[category_name]:
                    if(category_tag in pred_text):
                        pred_dict[filename][category_name] = category_dict[category_name].index(category_tag)
                        tag_inc = True
                        break
                if(tag_inc == False):
                    if(zero_shot_similarity == "True"):
                        choices = category_dict[category_name]
                        options = ["{} : {}".format(category_name,choice) for choice in choices]
                        result = neural_similarity(pred_text,options,model)
                        pred_dict[filename][category_name] = choices.index(result)
                        str_ = "{}, {}, {}\n".format(category_name,pred_text,result)
                        with open(track_name,"a") as f:
                            f.write(str_)
                        # print(category_name,pred_text,result)
                    else:
                        pred_dict[filename][category_name] = -1
            count += 1    
    return category_keys, pred_dict

def convert_llava_result_to_label(result_path,category_dict,threshold,zero_shot_similarity):
    """
    given the result file of a previous run with llava as the model to predict the image tags, converts the prediction to a format that is easily comparable to the ground truth.

    Args:
        result_path (string): model output.
        category_dict (dictionary): possible tags.
        zero_shot_similarity (bool): whether to use neural similarity between model output and the possible tags. Maps model ouput to the tag with the highest embedding similarity. 
                    else: uses manual rules for the mapping

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.

    """
    import pickle

    json_path = result_path + ".json"
    pkl_path = result_path + ".pkl"


    # return the config defined in the predictionfile
    config_dict = util.get_config_from_prediction(json_path)

    #category name of the prediction is appended after the prompt, extract that
    
    #correct _ - errors i made
    category = config_dict["prompt"].split(";")[1].strip()
    if(category == 'traffic_sign_for_ego_vehicle'):
        category = 'traffic_sign_for_ego-vehicle'
    if(category == 'traffic_light_for_ego_vehicle'):
        category = 'traffic_light_for_ego-vehicle'


    # Open file containting the predicted text and corresponding scores
    data = []
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
        
    #define zero shot classifier for predictions not matchable with manual rules
    if(zero_shot_similarity == "True"):
        model = None

    # possible tags for the category given     
    choices = category_dict[category]

    #iterate over the prediction per image
    pred_dict = {}
    for preds_per_file in data:

        #correct filename to not contain the path to the file
        filename = preds_per_file.split("/")[-1]
        pred_dict[filename] = {}

        #get prediction text for the given image
        pred_text = data[preds_per_file]["text"]
        if isinstance(pred_text, str):
            pred_text = pred_text.lower()
        else:
            pred_text = pred_text[0].lower()

        #if predicted text is "", no need to process further
        if(pred_text == ""):
            pred_dict[filename][category] = -1
            continue

        # get prediction score for the predicted text, get item with largest score. 
        # This corresponds to "yes" or "no" if the prediction text is "yes" or "no"
        sm = data[preds_per_file]["scores"][0]
        score = np.sort(sm)[-1].item()

        #corrects returned prediction to match defined tags with manual rules
        pred_text = correct_tag(pred_text.strip(),category)
            
        # if a manual threshold is defined above which to consider a binary prediction as positive, use this threshold
        # else check if prediction matches a defined tag, set prediction for the category
        #    else check if a defined tag is contained inside the prediction
        #       if not use zero shot classifier to assign tag with the highest similarity
        if(threshold != "None" and (pred_text == "yes" or pred_text == "no")):
            if(score < float(threshold)):
                if(pred_text == "yes"):
                    pred_text = "no"
                else:
                    pred_text = "yes"
        elif(pred_text in choices):
            pred_dict[filename][category] = choices.index(pred_text)
        else:
            in_ = False
            for choice in choices:
                if(choice in pred_text):
                    pred_dict[filename][category] = choices.index(choice)
                    in_ = True
                    break
            if(in_ == True):
                continue
            if(zero_shot_similarity == "True"):
                options = ["{} : {}".format(category,choice) for choice in choices]
                if(model == None):
                    model = Classifier()
                result = neural_similarity(pred_text,options,model)
                str_ = "{}, {}, {}\n".format(category,pred_text,result)
                
                #debug
                with open(track_name,"a") as f:
                    f.write(str_)
                pred_dict[filename][category] = choices.index(result)
            else:
                pred_dict[filename][category] = -1
        

    return [category], pred_dict

def convert_clip_result_to_label(result_path,category_dict):
    """
    given the result file of a previous run with clip as the model to predict the image tags, converts the prediction to a format that is easily comparable to the ground truth.

    Args:
        result_path (string): model output.
        category_dict (dictionary): possible tags.

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.

    """

    txt_path = result_path + ".txt"
    pkl_path = result_path + ".pkl"

    result_txt = ""
    with open(txt_path, 'r') as file:
        # Read the content of the file
        result_txt = file.read()

    split = result_txt.split("+" * 20 + "PREDICTIONS" + "+" * 20)

    categorie_split, _ = split


    config =  categorie_split.split("\n")
    config_dict = {}
    for conf in config[2:]:
        if(conf == ""):
            continue
        key, value = conf.split(": ")
        key = key.strip()
        value = value.strip()
        config_dict[key] = value 

    category = config_dict["prompt"].split(" ; ")[1].strip()
    choices = category_dict[category]
    data = []

    # Open the file in binary mode
    with open(pkl_path, 'rb') as file:
        # Load the pickled data
        data = pickle.load(file)

    pred_dict = {}
    for preds_per_file in data:
        filename = preds_per_file.split("/")[-1]
        pred_scores = data[preds_per_file]["scores"]
        tag = np.argmax(pred_scores)
        
        pred_dict[filename] = {category: tag}
        

    return [category], pred_dict

def convert_composer_result_to_label(result_path,category_dict,zero_shot_similarity):
    """
    given the result file of a previous run with llava as the model to predict the image tags, converts the prediction to a format that is easily comparable to the ground truth.

    Args:
        result_path (string): model output.
        category_dict (dictionary): possible tags.
        zero_shot_similarity (bool): whether to use neural similarity between model output and the possible tags. Maps model ouput to the tag with the highest embedding similarity. 
                    else: uses manual rules for the mapping

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.

    """
    json_path = result_path + ".json"

    # return the config defined in the predictionfile
    config_dict = util.get_config_from_prediction(json_path)

    #category name of the prediction is appended after the prompt, extract that
    category = config_dict["prompt"].split(";")[1].strip()

    
    #correct _ - errors i made
    if(category == 'traffic_sign_for_ego_vehicle'):
        category = 'traffic_sign_for_ego-vehicle'
    if(category == 'traffic_light_for_ego_vehicle'):
        category = 'traffic_light_for_ego-vehicle'
    
    # Open file containting the predicted text, get predictions
    data = []
    with open(json_path, 'r') as file:
        # Read the content of the file
        data = json.load(file)
    data = data["prediction"]
    
    #define zero shot classifier for predictions not matchable with manual rules
    if(zero_shot_similarity == "True"):
        model = None
        
    # possible tags for the category given     
    choices = category_dict[category]
    
    #iterate over the prediction per image
    pred_dict = {}
    for preds_per_file in data:

        #correct filename to not contain the path to the file
        filename = preds_per_file.split("/")[-1]
        pred_dict[filename] = {}
        
        #get prediction text for the given image
        pred_text = data[preds_per_file]
        if isinstance(pred_text, str):
            pred_text = pred_text.lower()
        else:
            pred_text = pred_text[0].lower()

        #if predicted text is "", no need to process further
        if(pred_text == ""):
            pred_dict[filename][category] = -1
            continue
        
        #corrects returned prediction to match defined tags with manual rules
        pred_text = correct_tag(pred_text,category)

        # if prediction matches a defined tag, set prediction for the category
        #    else check if a defined tag is contained inside the prediction
        #       if not use zero shot classifier to assign tag with the highest similarity
        if(pred_text in choices):
            pred_dict[filename][category] = choices.index(pred_text)
        else:
            in_ = False
            for choice in choices:
                if(choice in pred_text):
                    pred_dict[filename][category] = choices.index(choice)
                    in_ = True
                    break
            if(in_ == True):
                continue
            if(zero_shot_similarity == "True"):
                if(model == None):
                    model = Classifier()
                options = ["{} : {}".format(category,choice) for choice in choices]
                result = neural_similarity(pred_text,options,model)
                pred_dict[filename][category] = choices.index(result)
                str_ = "{}, {}, {}\n".format(category,pred_text,result)

                #debug
                with open(track_name,"a") as f:
                    f.write(str_)
            else:
                pred_dict[filename][category] = -1
        

    return [category], pred_dict

def get_predictions(pred_path,category_dict,threshold,zero_shot_similarity,model):
    """
    helper function to call the appropriate prediction to label conversion.

    Args:
        pred_path (string): path + name of the model output.
        category_dict (dictionary): key: possible categories, value: possible tags.
        threshold (float): if model prediction score is larger than the threshold, considers a positive model prediction as one, else false

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.s

    """
   
    
    if(model == "gpt4"):
        return convert_gpt4_result_to_label(pred_path,category_dict,zero_shot_similarity)
    if("llava" in model or "yi" in model):
        return convert_llava_result_to_label(pred_path,category_dict,threshold,zero_shot_similarity)
    if(model == "clip"):
        return convert_clip_result_to_label(pred_path,category_dict)
    else:
        return convert_composer_result_to_label(pred_path,category_dict,zero_shot_similarity)

def get_labels(label_path,dataset="txt"):
    """
    loads the labels into a dictionary for simple comparison to predictions.

    Args:
        label_path (string): path + name to the label file.
        dataset (string): specifies which dataset format is used. currently supported: bdd100k and txt

    Returns:
        dictionary: key: name of the images, value: dictionary: key: category name, value: index of the tag.s

    """

    others = []
    category_dict = {}
    if("bdd100k" in label_path):

        #categories to use with the bdd100k dataset
        category_dict = {
            'weather' : ['rainy','snowy','clear','overcast','partly cloudy','foggy','undefined'],
            'time_of_day' : ['twilight','daytime','nighttime','undefined'],
            'urban_environment' : ['tunnel', 'residential area', 'parking lot', 'city street', 'gas stations', 'highway', 'undefined'],
            'number_of_vulnerable_road_users' : ['none','1-4','5-9','>9'],
            'number_of_large_motor_vehicles' : ['none','1-4','5-9','>9'],
            'traffic_sign_for_ego-vehicle' : ['yes', 'no'],
            'traffic_light_for_ego-vehicle' : ['yes', 'no'],
            'train' : ['yes', 'no'],
            'person' : ['yes', 'no'],
        }

        #load labelfile
        with open(label_path, 'r') as file:
            content = json.load(file)

        #iterate over labels foreach image
        labels_dict = {}
        for i, file in enumerate(content):

            #read image metainformation
            name = file["name"]
            attributes = file["attributes"]
            weather = attributes["weather"]
            time_of_day = attributes["timeofday"].replace("night","nighttime").replace("dawn/dusk","twilight")
            urban_enviroment = attributes["scene"]

            other = file["labels"]
            number_of_vulnerable_road_users = 0
            number_of_large_motor_vehicles = 0
            traffic_sign_for_ego_vehicle = 1
            traffic_light_for_ego_vehicle = 1
            train = 1
            vru_labels = ["pedestrian","rider"]
            vehicle_labels = ["car","truck","bus"]

            # iterate over objects in the label file and check for object type, if object type should be used as labelinformation, 
            # increase its count or set its boolean value
            for other_ in other:
                other__ = other_["category"]
                if(other__ == "traffic light"):
                    traffic_light_for_ego_vehicle = 0
                if(other__ == "traffic sign"):
                    traffic_sign_for_ego_vehicle = 0
                if(other__ == "train"):
                    train = 0
                if(other__ in vru_labels):
                    number_of_vulnerable_road_users += 1
                if(other__ in vehicle_labels):
                    number_of_large_motor_vehicles += 1
                if(other__ not in others):
                    others.append(other__)

            #assign tags per category based on the number of objects specified in the labelfile
            if(number_of_large_motor_vehicles > 10):
                number_of_large_motor_vehicles = 3
            elif(number_of_large_motor_vehicles > 4):
                number_of_large_motor_vehicles = 2
            elif(number_of_large_motor_vehicles > 0):
                number_of_large_motor_vehicles = 1

            if(number_of_vulnerable_road_users > 10):
                number_of_vulnerable_road_users = 3
            elif(number_of_vulnerable_road_users > 4):
                number_of_vulnerable_road_users = 2
            elif(number_of_vulnerable_road_users > 0):
                number_of_vulnerable_road_users = 1

            if(urban_enviroment == "residential"):
                urban_enviroment = "residential area"
            
            #create the label dictionary containing the indices corresponding to the tag
            labels_dict[name] = {"weather": category_dict["weather"].index(weather), "time_of_day": category_dict["time_of_day"].index(time_of_day),\
                                 "urban_environment": category_dict["urban_environment"].index(urban_enviroment), \
                                 "traffic_sign_for_ego-vehicle": traffic_sign_for_ego_vehicle, \
                                 "traffic_light_for_ego-vehicle": traffic_light_for_ego_vehicle, \
                                 "train": train, \
                                 "number_of_large_motor_vehicles": number_of_large_motor_vehicles,\
                                 "number_of_vulnerable_road_users": number_of_vulnerable_road_users,\
                                 "person": 0 if number_of_vulnerable_road_users > 0 else 1}
        

    elif("txt" in label_path):
        content = ""


        #read text file
        with open(label_path, 'r') as file:
            content = file.read()
            
        #split config and labelinformation
        split = content.replace("\\","").split("+" * 20 + "LABELS" + "+" * 20)
        categorie_split, label_split = split
        categorie_split = categorie_split.replace("++++++++++++++++++++LABELS++++++++++++++++++++","")

        #read categories for which the labelfile contains information and the corresponding tags into dictionary
        categories = categorie_split.split("\n")    
        category_dict = {}
        for line in categories[1:]:
            if(line == "" or "category_dict" in line):
                continue
            split = line.split(":")
            
            category, value = split
            category_dict[category.replace(" ","").replace("'","")] = ast.literal_eval(value)

        # create label dictionary, containing the image name as key, 
        # the value of each image is a dictionary with the category name as key and the prediction as value, i.e.
        #   - image_name: category_name: label_index
        # achieved by iterating over the lines of the textfile containing the labels and then over the label per category
        category_keys = list(category_dict.keys())
        labels_per_files = label_split.split("\n")        
        labels_dict = {}
        for label_per_file in labels_per_files:
            if(len(label_per_file) < 5):
                continue

            #get filename of the image and the labels per category
            filename, value = label_per_file.split("=")

            filename = filename.strip()
            value = value.strip()
            labels_dict[filename] = {}
            
            # iterate over the indices of the label for each of the categories, save in label_dict
            value = ast.literal_eval(value)
            for i in range(len(value)):
                
                labels_dict[filename][category_keys[i].replace("'","")] = value[i]  
        
    return category_dict, labels_dict

def compare_prediction_to_labels(predictions,labels,category_dict,prediction_categories):
    """
    compares the predictions specified in predictions to the labels specified in labels.

    Args:
        predictions (dictionary): key:image name, value: dictionary(key: category name, value: tag index).
        labels (dictionary): key:image name, value: dictionary(key: category name, value: tag index).
        category_dict (dictionary): key: category names, value: tags per category.
        prediction_categories (list,string): predicted categories.

    Returns:
        dictionary: key: category name, value: accuracy
        dictionary: key: category name, value: total number of predictions per category
        dictionary: key: category name, value: number of no prediction per category
        dictionary: key: category name, value: confusion matrix per category
    """

    #check if predicted categories are contained in labelfile
    label_categories = category_dict.keys()
    for pred_category in prediction_categories:
        if(pred_category not in label_categories):
            print("category {} not in label file".format(pred_category))
            sys.exit()

    #define dictionaries for perfomance per category
    accuracy_dict = {prediction_category: 0 for prediction_category in prediction_categories}
    total_dict = {prediction_category: 0 for prediction_category in prediction_categories}
    no_pred_dict = {prediction_category: 0 for prediction_category in prediction_categories}

    #dicts for wrong predictions
    mismatch_dict = {prediction_category: [] for prediction_category in prediction_categories}
        
    #dict to count correct predicions per tag
    confusion_dict = {prediction_category: np.zeros((10,10)) for prediction_category in prediction_categories}

    #iterate over per image predictions
    for image_path in predictions.keys():
            #iterate over per category predictions
        for category in prediction_categories:
            total_dict[category] += 1
                
            #correct name if old name was used in prediction file
            displyed_image_name = image_path.split("rosbag2")[-1]
                
            #check if a prediction for the image and category exists
            if(category not in predictions[image_path].keys()):
                mismatch_dict[category].append((image_path,"{} not tracked".format(displyed_image_name)))
                continue
                
            #get prediction an label for the image and category
            find = predictions[image_path][category]
            sind = labels[image_path][category]
            confusion_dict[category][find,sind] += 1
            
            if(predictions[image_path][category] == -1):
                no_pred_dict[category] += 1


            #check if matching
            if(predictions[image_path][category] == labels[image_path][category]):
                accuracy_dict[category] += 1
            else:
                mismatch_dict[category].append((image_path,displyed_image_name + " " + category_dict[category][predictions[image_path][category]]))
                
    return accuracy_dict, total_dict, no_pred_dict, mismatch_dict, confusion_dict

def get_optimal_threshold(result_path,category_dict,zero_shot_similarity, labels, tag_frequency):
    """
    given a prediction file from a model that returns scores (not gpt4) iterates over thresholds in the range of 0.01 to 0.99 with step size 0.01. 
    Prints accuracy for each of the thresholds

    Args:
        result_path (string): path + name of the prediction file without file ending.
        category_dict (list,string): possible tags.
        zero_shot_similarity (bool): possible tags.
        labels (list,string): possible tags.
        tag_frequency (list,string): numpy, int32(#categories,10) tag counts.

    Returns:
        None
    """
    accuracies = np.zeros(100,np.int32)

    #iterate over scores from 0 to 0.99
    for i in range(0,100):

        #get labels
        prediction_categories, predictions = convert_llava_result_to_label(result_path,category_dict,i / 100.,zero_shot_similarity)
        
        accuracy_dict, total_dict, no_pred_dict, mismatch_dict, confusion_dict = \
            compare_prediction_to_labels(predictions,labels,category_dict,prediction_categories)

        
        #calculate precision, recall, f1 per category 
        precision_dict = {category:(confusion_dict[category] * np.eye(10)) / (np.sum(confusion_dict[category],axis=1) + 1e-7) for category in prediction_categories}
        recall_dict = {category:(confusion_dict[category] * np.eye(10)) / (np.sum(confusion_dict[category],axis=0) + 1e-7) for category in prediction_categories}
        f1_dict = {category: 2 * precision_dict[category] * recall_dict[category] / (precision_dict[category] + recall_dict[category] + 1e-7) for category in prediction_categories}
        
        #update score dictionaries
        for category in accuracy_dict.keys():
            f = f1_dict[category] * tag_frequency[category]
            f1_dict[category] = sum(sum(f1_dict[category] * tag_frequency[category]))
            accuracy_dict[category] = accuracy_dict[category] / total_dict[category]
        print(i, accuracy_dict,f1_dict)
    pass

def evaluate(config):
    """
    Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

    Args:
        outputs (string): model output.
        options (list,string): possible tags.

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """
    config_ = util.get_config_from_prediction(config["General"]["prediction_path"] + ".json")
    model = config_["model"]
    
    #debug
    with open(track_name,"a") as f:
        f.write("PREDICTION: "  + config["General"]["prediction_path"] + " model: " + model + "\n")

    category_dict, labels = get_labels(config["General"]["label_path"],config["General"]["dataset"])
    prediction_categories, predictions = get_predictions(config["General"]["prediction_path"],category_dict,config["General"]["threshold"],config["General"]["zero_shot_similarity"],model)
    
    #debug
    if("llava" in model and os.path.exists(config["General"]["prediction_path"] + ".pkl") == False):
        with open(track_name,"a") as f:
            f.write("no pkl file")
        return
    
    if(predictions == {}):
        with open(track_name,"a") as f:
            f.write("predictions empty")
        return
    
    
    accuracy_dict, total_dict, no_pred_dict, mismatch_dict, confusion_dict = \
        compare_prediction_to_labels(predictions,labels,category_dict,prediction_categories)

    #calculate f1 score per category
    f1_dict = {}
    for prediction_category in prediction_categories:
        prediction_keys = natsorted(predictions.keys())
        sorted_prediction_dict = {key: predictions[key] for key in prediction_keys}

        label_keys = natsorted(labels.keys())
        sorted_label_dict = {key: labels[key] for key in label_keys}

        prediction_list = util.merge_values(sorted_prediction_dict,prediction_category)
        label_list = util.merge_values(sorted_label_dict,prediction_category)
        if(len(prediction_list) == len(label_list)):
            f1_score_ = f1_score(prediction_list, label_list,average="weighted")
            f1_dict[prediction_category] = f1_score_
        else:
            with open(track_name,"a") as f:
                f.write("not the same number of labels and predictions")

            f1_dict[prediction_category] = 0

    #calculate percentages
    for category in accuracy_dict.keys():
        accuracy_dict[category] = accuracy_dict[category] / total_dict[category]
        no_pred_dict[category] = no_pred_dict[category] / total_dict[category]
    
    #create path to save the evaluation results
    print(config["General"]["prediction_path"])
    prediction_path = config["General"]["prediction_path"].split(".")[0].replace("base/out/","base/out/analysed_predictions/")
    if(os.path.exists(prediction_path) == False):
        os.mkdir(prediction_path)

    #if specified, visualizes the wrong predictions
    if(config["General"]["visualize_wrongs"] == "True"):
        util.visualize_wrongs(prediction_path + "/",config["General"]["input_path"] + "/",mismatch_dict,prediction_categories)
    
    print("accuracy",accuracy_dict)
    print("f1 score",f1_dict)
    print("No prediction",no_pred_dict)

    #evaluation results are stored at a different location to discriminate zero shot evaluated performance
    if(config["General"]["zero_shot_similarity"] == "True"):
        split = prediction_path.split("/")
        prediction_path_ = "/".join(split[:-1]) + "/zero_shot_similarity/"# + 
        if(os.path.exists(prediction_path_) == False):
            os.mkdir(prediction_path_)
        prediction_path_ += split[-1]
        if(os.path.exists(prediction_path_) == False):
            os.mkdir(prediction_path_)
    else:
        prediction_path_ = prediction_path

    #write performance to file
    with open(prediction_path_ + "/measures.txt","w") as f:
        for key in accuracy_dict.keys():
            f.write("{} : {}, {}, {}\n".format(key,accuracy_dict[key],f1_dict[key],no_pred_dict[key]))
    return accuracy_dict

def create_prediction_file(number_of_prompts):
    """
    Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

    Args:
        outputs (string): model output.
        options (list,string): possible tags.

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """
    
    category_dict, labels = get_labels("labels_257.txt","txt")
    base_path = "data/out/"
    out_files = os.listdir(base_path)
    print(out_files)
    max_index = int(natsorted(out_files)[-1].split("_")[-1].split(".")[-2])

    files = {}
    for i in range(max_index-number_of_prompts+1,max_index+1):
        config_ = util.get_config_from_prediction(base_path + "predictions_" + str(i) + ".json")
        model = config_["model"]
        thresh = "0.74" if i == 291 else "None"
        prediction_categories, predictions = get_predictions(base_path + "predictions_" + str(i),category_dict,thresh,"False",model)
        for file in predictions.keys():
            keys_ = list(predictions[file].keys())
            if(file not in files):
                files[file] = {key_ : category_dict[key_][predictions[file][key_]] for key_ in keys_}
            else:
                for key_ in keys_:
                    files[file][key_] = category_dict[key_][predictions[file][key_]]

    grouped_files = util.group(files)
    # grouped_files = util.old_name_to_new(grouped_files)
    for group_file  in grouped_files:
        sorted_dict = {key: grouped_files[group_file][key] for key in natsorted(grouped_files[group_file])}
        
        with open("data/results/" + group_file + ".json", 'w') as json_file:
            json.dump(sorted_dict, json_file, indent=4)
        print("saved to: " + "data/results/" + group_file + ".json")

def predict_and_prediction_file():
    args = parse_command_line_args()

    if(os.path.exists(args.config) == False):
        create_config(args.config)
    config = read_config(args)

    model = models[config["General"]["model"]](config)

    with open("prompts.json", 'r') as file:
        prompts = json.load(file)

    prompts = prompts["best_prompts"][config["General"]["model"]]
    #if multiple prompts given by a list, iterate over them
    if isinstance(prompts, str):
        config["General"]["prompt"] = prompts
        predict(config,model)
    elif isinstance(prompts, list):
        for prompt in prompts:
            prompt = str(prompt)
            config["General"]["prompt"] = prompt
            predict(config,model)
    
    create_prediction_file(len(prompts))


def multiple():
    """
    simple prediction for multiple models and prompts.

    Args:
        outputs (string): model output.
        options (list,string): possible tags.

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """
    
    # "gpt4": GPT4sender,#
    # "llava-1.6_7m":LLaVA,#
    # "llava-1.6_7v": LLaVA,#
    # "llava-1.5": LLaVA,#
    # "llava-1.6_13":LLaVA,#
    # "llava-1.6_34":LLaVA,#
    # "clip": CLIP,
    # "yi-6":YI,
    # "yi-34":YI,
    # "composer-vl":internlm,#
    # "composer-hd":internlm,#
    # "deepseek":deepseek,
    # "cogvlm":cog,#
    # "cogagent":cog,#
    # "cogagent-vqa":cog,#


    #appendix
    # "blip-caption-large":blip,
    # "blip-itm-coco":blip,
    # "blip-capfilt-large":blip,
    # "blip2-opt":blip,#
    # "blip2-t5-xxl":blip,#
    # "instructblip-13b-v":blip#
    #bilder direkt über sdk update fist un
        
    args = parse_command_line_args()
    models__ = ["llava-1.5"]
    prompts = ["is the weather in the scene rainy, snowy, clear, overcast, partly cloudy, foggy, undefined? return only most fitting tag. ; weather"]
    
    prompts = ["{  'prompt': 'weather',  'classification': {    'labels': ['rainy','snowy','clear','overcast','partly cloudy','foggy','undefined'],    'output_format': 'label'  }} ; weather"]
    
    for model_name in models__:
        config = read_config(args)
        
        # #for faster debugging
        config["General"]["task"] = "predict"#"evaluate"#x
        config["General"]["model"] = model_name
        config["General"]["zero_shot_similarity"] = "True"
        config["General"]["visualize_wrongs"] = "True"
        config["General"]["input_path"] = "/home/ge32buc/base/rosbag_complete_new"#"/home/ge32buc/bdd100k/data/bdd100k/images/100k/val"#"/home/ge32buc/base/rosbag_validation_257_anon/"#"/home/ge32buc/base/LLaVA/rosbag_three_images_per_file"
        config["General"]["output_path"] = "/home/ge32buc/base/out"
        config["General"]["label_path"] = "/home/ge32buc/bdd100k/data/bdd100k/labels/bdd100k_labels_images_val.json"#"/home/ge32buc/base/labels_257.txt"#"/home/ge32buc/base/labels.txt"#
        base_pred_path = "/home/ge32buc/base/out/new/"
        config["General"]["output_path"] = "/home/ge32buc/base/out/new/"
        config["General"]["prediction_path"] = base_pred_path
        
        
        if(config["General"]["task"] == "predict"):
            print(config["General"]["input_path"])
            if(config["General"]["input_path"] == "/-1"):
                print("--input_path needs to be specified for prediction")
                sys.exit()
            if(prompts is None):
                prompts = config["General"]["prompt"]
            if(config["General"]["model"] not in models.keys()):
                print("accepted models are: {}".format(', '.join(models.keys())))
                sys.exit()
            model = models[config["General"]["model"]](config)
            if isinstance(prompts, str):
                config["General"]["prompt"] = prompts
                predict(config,model)
            elif isinstance(prompts, list):
                for prompt in prompts:
                    prompt = str(prompt)
                    config["General"]["prompt"] = prompt
                    print(model_name, prompt)
                    predict(config,model)
                
        elif(config["General"]["task"] == "evaluate"):
            if(config["General"]["label_path"] == "/-1" or config["General"]["prediction_path"] == "/-1"):
                print("--label_path and --prediction_path need to be specified for evaluation")
                sys.exit()

            # specify range of predictionfiles to evaluate multiple
            for i in tqdm(range(1836,1899)):
                path = base_pred_path + "predictions_{}".format(i)
                if(os.path.exists(path + ".json")):
                    config["General"]["prediction_path"] = path
                    evaluate(config)
            break
# multiple()
def main(prompts = None):
    args = parse_command_line_args()


    if(os.path.exists(args.config) == False):
        create_config(args.config)
    config = read_config(args)

    if(config["General"]["task"] == "predict"):
        print(config["General"]["input_path"])

        #check if input path is specified
        if(config["General"]["input_path"] == "/-1"):
            print("--input_path needs to be specified for prediction")
            sys.exit()

        #check if model specified is available
        if(config["General"]["model"] not in models.keys()):
            print("accepted models are: {}".format(', '.join(models.keys())))
            sys.exit()

        #define model
        model = models[config["General"]["model"]](config)

        #if multiple prompts given by a list, iterate over them
        if isinstance(prompts, str):
            config["General"]["prompt"] = prompts
            predict(config,model)
        elif isinstance(prompts, list):
            for prompt in prompts:
                prompt = str(prompt)
                config["General"]["prompt"] = prompt
                predict(config,model)
            
    elif(config["General"]["task"] == "evaluate"):
        #ccheck if label path and prediction path are specified
        if(config["General"]["label_path"] == "/-1" or config["General"]["prediction_path"] == "/-1"):
            print("--label_path and --prediction_path need to be specified for evaluation")
            sys.exit()

        if(os.path.exists(config["General"]["prediction_path"] + ".json")):
            evaluate(config)
    


if __name__ == '__main__':
    predict_and_prediction_file()