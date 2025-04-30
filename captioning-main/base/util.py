import matplotlib.pyplot as plt

import ast
import time
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
import json

from PIL import Image, ImageDraw, ImageFont

# def load_results():
#     """
#     Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

#     Args:
#         outputs (string): model output.
#         options (list,string): possible tags.

#     Returns:
#         list,float: softmax over similarity between output and rach of the tags specified in options.

#     """
#     import pickle
#     with open("/home/ge32buc/base/out/resultss_18.pkl", 'rb') as file:
#         # Use pickle.load() to retrieve the data from the file
#         loaded_data = pickle.load(file)
    
#     print(loaded_data)

def merge_values(dict,prediction_category):
    """
    recursively merges the values of dictionary.

    Args:
        dict (dictionary): dictionary containing the keys of prediction_category.
        prediction_category (string): name of the current category.

    Returns:
        list,float: values inside dict.

    """
    merged_values = []
    for value in dict.values():
        if isinstance(value, dict):
            # If the value is another dictionary, recursively merge its values
            if(prediction_category in value.keys()):
                value = merged_values.append(value[prediction_category])
            else:
                merged_values.extend(merge_values(value,prediction_category))
        else:
            # If the value is a list (or other iterable), extend the merged_values list
            merged_values.append(value)
    return merged_values

def count_tag_frequency(category_dict,labels_dict):
    """
    helper function for evaluation to count for each tag how often it occured in the given prediction or label dictionary.

    Args:
        category_dict (dictionary): key: possible category names, value: possible tag names.
        labels_dict (dictionary): key: category name, value: index of the tag prediction.

    Returns:
        numpy int (# categories,10): count for each of the tag per category in the prediction or label dictionary.

    """
    tags = category_dict
    tag_counts = np.zeros((len(tags),10),dtype=np.int32)
    for item in labels_dict:
        for i, tag in enumerate(tags):
            tag_counts[i,labels_dict[item][tag]] += 1
    return tag_counts

def get_all_files_in_directory(directory):
    """
    finds all files recursively in a given directory. used to get the input images in a directory

    Args:
        directory (string): directory to find files in recursively.

    Returns:
        list,string: absolute path of all files in the directory.

    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

# def build_suffix_array(s):
#     # Build suffix array using O(n log n) approach (e.g., using Suffix Array Induced Sorting)
#     suffixes = sorted((s[i:], i) for i in range(len(s)))
#     return [suffix[1] for suffix in suffixes]

# def build_lcp_array(s, suffix_array):
#     """
#     Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

#     Args:
#         outputs (string): model output.
#         options (list,string): possible tags.

#     Returns:
#         list,float: softmax over similarity between output and rach of the tags specified in options.

#     """
#     n = len(s)
#     rank = [0] * n
#     lcp = [0] * n

#     for i, suffix in enumerate(suffix_array):
#         rank[suffix] = i

#     h = 0
#     for i in range(n):
#         if rank[i] > 0:
#             j = suffix_array[rank[i] - 1]
#             while i + h < n and j + h < n and s[i + h] == s[j + h]:
#                 h += 1
#             lcp[rank[i]] = h
#             if h > 0:
#                 h -= 1

#     return lcp

# def find_duplicate_substrings(s):
#     """
#     Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

#     Args:
#         outputs (string): model output.
#         options (list,string): possible tags.

#     Returns:
#         list,float: softmax over similarity between output and rach of the tags specified in options.

#     """
#     n = len(s)
#     suffix_array = build_suffix_array(s)
#     lcp_array = build_lcp_array(s, suffix_array)

#     duplicates = set()
#     for i in range(1, n):
#         length = lcp_array[i]
#         if length > 0:
#             substring = s[suffix_array[i]:suffix_array[i] + length]
#             duplicates.add(substring)

#     return list(duplicates)

def group(files):
    """
    Calculates similarity between the textual embedding of the model output and each of the texts specified in options.

    Args:
        outputs (string): model output.
        options (list,string): possible tags.

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """

    grouped_files = {}
    for file in files.keys():
        if("cam_fc" not in file and "pylon_camera" not in file and "cam_fl" not in file and "cam_rl_" not in file and \
           "cam-fc" not in file and "pylon-camera" not in file and "cam-fl" not in file and "cam-rl_" not in file and "camera_basler_frontcenter_" not in file):
            print(file)
        file_rep = file.replace("cam_fc_","").replace("pylon_camera_","").replace("cam_fl_","").replace("cam_rl_","") \
                        .replace("cam-fc_","").replace("pylon-camera_","").replace("cam-fl_","").replace("cam-rl_","").replace("camera_basler_frontcenter_","")
        mcap_split = file_rep.split("_")
        mcap_split_copy = mcap_split.copy()
        indices = []
        while "rosbag2" in mcap_split_copy:
            if(len(indices) != 0):
                indices.append(mcap_split_copy.index("rosbag2") + indices[-1] + 1)
            else:
                indices.append(mcap_split_copy.index("rosbag2"))
            mcap_split_copy = mcap_split_copy[indices[-1] + 1:]
        #for split in mcap_split:
        mcap_name = "_".join(mcap_split[:-2])
        mcap_name_ = "_".join(mcap_split[indices[-1]:-1])#.split(".")[0]
        if(mcap_name not in grouped_files.keys()):
            grouped_files[mcap_name] = {mcap_name_ : files[file]}
        else:
            grouped_files[mcap_name][mcap_name_] = files[file]

    return grouped_files


def old_name_to_new(grouped_files):
    """
    converts the image name from the s3fs representation to the edgar sdk name.

    Args:
        grouped_files (dictionary): filenames.

    Returns:
        dictionary: key: old filename, value: new filename.
    """

    # Specify the path to your JSON file containing the dictionary of bagname to scenename
    file_path = '/bags.json'

    # Open the JSON file and load its contents
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    if(isinstance(grouped_files, dict)):
        keys_ = list(grouped_files.keys())
        for i, grouped_file in enumerate(keys_):
            t = False
            for dat in data:
                f_ = data[dat][0].split(".")[0]
                f__ = "_".join(f_.split("_")[:-1])
                if(f__ in grouped_file):
                    splitted_r = grouped_file.split("rosbag2")[-1]
                    grouped_files[dat + "_rosbag2" + splitted_r] = grouped_files.pop(grouped_file)
                    #grouped_files_.append(grouped_file)
                    t = True
                    break
            if(t == False):
                grouped_files["rosbag2" + "rosbag2".join(grouped_file.split("rosbag2")[1:])] = grouped_files.pop(grouped_file)
                #grouped_files[i] = "rosbag2" + "rosbag2".join(grouped_file.split("rosbag2")[1:])
                print(grouped_file)
        
        return grouped_files
    elif(isinstance(grouped_files, list)):
        keys_ = grouped_files
        grouped_files_ = []
        for i, grouped_file in enumerate(keys_[1:-1]):
            t = False
            for dat in data:
                f_ = data[dat][0].split(".")[0]
                f__ = "_".join(f_.split("_")[:-1])
                if(f__ in grouped_file):
                    #grouped_files[dat] = grouped_files.pop(grouped_file)
                    splitted_r = grouped_file.split("rosbag2")[-1]
                    grouped_files[i] = dat + "_rosbag2" + splitted_r
                    #grouped_files_.append(grouped_file)
                    t = True
                    break
            if(t == False):
                grouped_files[i] = "rosbag2" + "rosbag2".join(grouped_file.split("rosbag2")[1:])
                print(grouped_file)
            
        
        return grouped_files

def write_text_on_image(input_image_path, output_image_path, text_to_write):
    """
    writes prediction on the image to simplyify debugging.

    Args:
        input_image_path (string): image path.
        output_image_path (string): output image path.
        text_to_write (string): text to write.

    Returns:
        None.

    """
    # Open the input image
    image = Image.open(input_image_path)

    # Initialize the drawing context
    draw = ImageDraw.Draw(image)

    # Choose a font (you can specify the font file path and font size)
    font_size = 36
    font = ImageFont.load_default()  # You can specify a custom font here
    font = font.font_variant(size=font_size)

    # Calculate text size and position
    #text_width, text_height = draw.textlength(text_to_write, font=font)
    text_x = 5#(image.width - text_width) // 2  # Center the text horizontally
    text_y = 5#(image.height - text_height) // 2  # Center the text vertically

    # Draw text on the image
    draw.text((text_x, text_y), text_to_write, font=font, fill="red")

    # Save the modified image
    image.save(output_image_path)

    print(f"Text '{text_to_write}' written on image '{output_image_path}'")

def visualize_result_in_image(filename, filecontent, image_path, category_dict):
    """
    writes multiple predictions on an image to simplyify debugging.

    Args:
        filename (string): image namefilename.
        filecontent (dictionary): key: category name, value: prediction.
        image_path (string): path to input image.
        category_dict (dictionary): key: possible category names, value: possible tag names.

    Returns:
        None.

    """
    prediction_txt = ""
    for cat_ in filecontent:
        cat = list(cat_.keys())[0]
        pred = cat_[cat]
        name = category_dict[cat][pred]
        prediction_txt += cat + " " + name + "\n"
    write_text_on_image(image_path + "/" + filename, image_path + "/pred/" + filename, prediction_txt)
        
def visualize_prediction(files,category_dict):
    
    """
    writes prediction on images to simplyify debugging.

    Args:
        files (list): filenames.
        category_dict (dictionary): key: file name, value: (key: category names, value: prediction).

    Returns:
        None.

    """
    image_path = "/home/ge32buc/base/rosbag_one_image_per_file"
    for file in files:
        visualize_result_in_image(file,files[file],image_path,category_dict)

def get_category_dict_from_labels(label_path="/home/ge32buc/base/labels.txt"):
    """
    returns a dictionary containing the categorie names as key and the tags per category in a list as value.

    Args:
        label_path (string): path to labelfile.

    Returns:
        category_dict (dictionary): key: possible category names, value: possible tag names.

    """
    category_dict = {}
    with open(label_path, 'r') as file:
        content = file.read()
    categories = content.split("++++++++++++++++++++LABELS++++++++++++++++++++")[0].split("\n")
    category_dict_str = "\n".join(categories[1:])

    #split text per line, remove blanks at front and end
    lines = [line.strip() for line in category_dict_str.split('\n') if line.strip()]

    # Iterate over lines, parse and add key-value pairs to the dictionary
    for line in lines:
        key, value = map(str.strip, line.split(':', 1))
        category_dict[ast.literal_eval(key)] = ast.literal_eval(value)
    return category_dict, content

# def load_labels(base_path):
#     """
#     writes prediction on the image to simplyify debugging.

#     Args:
#         files (list): filenames.
#         prediction_dict (dictionary): key: file name, value: (key: category names, value: prediction).

#     Returns:
#         list,float: softmax over similarity between output and rach of the tags specified in options.

#     """
#     label_dict = {}
#     label_path = base_path + '/labels.txt'
#     category_dict, content = get_category_dict_from_labels(label_path)

#     labels = content.split("\n")
        
#     for line in labels:
#         if(".jpg" not in line):
#             continue
#         splitted = line.split(" = ")
#         key_ = splitted[0].replace("\t","").replace(" ","")
#         label_dict[key_] = np.array(ast.literal_eval(splitted[1].replace("\n","")))
#         if(label_dict[key_][-3] == 2 or label_dict[key_][-3] == 3):
#            label_dict[key_][-3] = 1
#     return label_dict, category_dict

def visualize_accuracy(accs):
    """
    plots a graph to show the accuracy depending on the threshold.

    Args:
        accs (list): accuracies.

    Returns:
        int: index of maxaccuracy.

    """

    # Create sample data

    x = np.arange(len(accs))
    y = accs

    # Find the index of the maximum value
    max_index = np.argmax(y)
    max_x = x[max_index]
    max_y = y[max_index]

    # Plot the graph
    plt.plot(x, y)
    plt.scatter(max_x, max_y, color='red', label='Maximum', marker='o')
    plt.annotate(f'Max: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y), xytext=(max_x + 1, max_y - 0.5),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Set labels and title
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.title('Simple Graph with Maximum Highlighted')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()
    return max_index

def get_config_from_prediction(json_path):
    """
    returns the configuration of the given prediction file.

    Args:
        json_path (string): path to the json prediction file.

    Returns:
        dictionary: key: name of the information provided in the config, value: corresponding value.

    """

    result_txt = ""
    
    with open(json_path, 'r') as file:
        data = json.load(file)

    return data["config"]

def get_all_categories(max_dict):
    """
    returns the name of all categories inside max_dict

    Args:
        max_dict (dictionary): key: image name, value(key:category name, value: score in that category).

    Returns:
        list,string: name of the categories in max_dict.

    """
    categories = []
    for key in max_dict.keys():
        for key_ in max_dict[key].keys():
            if(key_ not in categories):
                categories.append(key_)
    return categories

def get_model_dict(max_dict,categories):
    """
    transfers scores in dict to score in list per image.

    Args:
        max_dict (dictionary): key: image name, value(key:category name, value: score in that category).
        categories (list,string): name of the categories in max_dict

    Returns:
        dictionary: key: model name, value:list,float scores

    """
    model_dict = {model : [] for model in max_dict.keys()}
    #iterate over each category and each model, add the score for the category if the model has one, else 0
    for category in categories:
        for model in max_dict.keys():
            if(model == "llava-1.6_34"):
                print("huh")
            categories_ = list(max_dict[model].keys())
            if(category in categories_):
                model_dict[model].append(max_dict[model][category])
            else:
                model_dict[model].append(0)
    return model_dict

def create_bar_chart(max_dict):
    """
    creates a bar chart comparing the maximum performace of different models per category

    Args:
        max_dict (dictionary): key: image name, value(key:category name, value: score in that category).

    Returns:
        None.

    """

    # Sample data

    categories = get_all_categories(max_dict)
    model_dict = get_model_dict(max_dict,categories) 

    #title contains the mean of the accuracies
    mean = {}
    title = "mean accuracies: "
    for model in model_dict.keys():
        mean[model] = sum(model_dict[model]) / len(model_dict[model])
        title += model + ": " + str(round(mean[model], 4)) + "   "
    
    # performance without brightness
    # title += "without brightness: "
    # for model in model_dict.keys():
    #     bright_ind = categories.index("vision_impairing_brightness")
    #     scores = model_dict[model]
    #     # scores.pop(bright_ind)
    #     #scores = model_dict[model].pop(bright_ind)
    #     mean_ = sum(scores) / len(scores)
    #     title += model + ": " + str(round(mean_, 4)) + "   "
    print(mean)
    mean_str = str(mean)

    #split title in multiple lines
    mean_str = mean_str[:int(len(mean_str)/2)] + " <br> " + mean_str[int(len(mean_str)/2):]
    list_ = []
    for model in model_dict:
        for i, cat in enumerate(categories):
            list_.append([model,cat,model_dict[model][i]])

    #create dataframe with model name, category name, accuracy, for easy bar plot
    df = pd.DataFrame(list_,columns=['group','column','val'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    # print(df.loc[1,2])
    import plotly.express as px
    fig = px.bar(df, x="column", y="val",
                color='group', barmode='group',text="val", title=mean_str,
                height=1400,width=3400)#
    import plotly.io as pio
    pio.write_image(fig, 'plotly_plot.png')

def analyse_results():
    """
    collects the performance from different evaluation runs. saves an overview to an excel file for simple comparison

    Args:

    Returns:
        dictionary: keys:"model","filename","dataset","category,"prompt","accuracy","f1_score","no_pred".
            each with a list as value, containing the respective information of that evaluation

    """

    #base path of stored analysed predictions. i.e output of evaluation
    analysed_path = "/home/ge32buc/base/out/analysed_predictions/new/"
    prediction_path = analysed_path.replace("analysed_predictions/","")
    analysed_files = os.listdir(analysed_path)
    results = {"model":[],"filename":[],"dataset":[],"category":[],"prompt":[],"accuracy":[],"f1_score":[],"no_pred":[]}

    #iterate over evaluation output files
    for dir_ in analysed_files:
        if("zero_shot" in dir_):
            continue
        config_dict = get_config_from_prediction(prediction_path + dir_ + ".json")

        #colelct information about the model, dataset and the prompt
        model = config_dict["model"]
        dataset = "bdd100k" if "bdd100k" in config_dict["input_path"] else "txt"
        content = ""
        with open(analysed_path + "zero_shot_similarity/" + dir_ + "/measures.txt","r") as f:
            content = f.read()
        prompt = config_dict["prompt"]


        #iterate over categorynames, each line in the evaluation file contains results for a different category
        categories = content.split("\n")
        for category_ in categories:
            if(category_ == ""):
                continue
            splitted = category_.split(" : ")
            if(";" in prompt):
                prompt,category = prompt.split(" ; ")
            else:
                category = splitted[0]
            if("ego_vehicle" in category):
                category = category.replace("ego_vehicle","ego-vehicle")
            accuracy, f1, no_pred =  splitted[1].split(", ")

            results["model"].append(model)
            results["dataset"].append(dataset)
            results["filename"].append(dir_)
            results["prompt"].append(prompt)
            results["category"].append(category)
            results["accuracy"].append(float(accuracy))
            results["f1_score"].append(float(f1))
            results["no_pred"].append(float(no_pred))
    return results

def get_missing_prompts():
    """
    compares the prompts of different prediction runs. 
    If a model did not perform the prediction run with a prompt which was used with a different model, return that.

    Args:

    Returns:
        dictionary: key: model name, value: list,string of prompts used with different models but not this one

    """

    #get the dict of all evaluation files
    results = analyse_results()
    models = results["model"]
    prompts = results["prompt"]
    categories = results["category"]

    #restore input prompt
    prompts = [prompts[i]+ " ; " + categories[i] for i in range(len(prompts))]

    #get unique prompts
    unq_prompts = set(prompts)
    unq_models = list(set(models))

    #dictionary with key: model name, value: list, string of all the prompts this model was evaluated on
    model_prompt_dict = {model:[] for model in unq_models}
    for i in range(len(models)):
        model_prompt_dict[models[i]].append(prompts[i])

    missing_prompts = {model:[] for model in unq_models}
    for i in range(len(unq_models)):
        if(unq_models[i] == "gpt4"):
            continue
        for unq_prompt in unq_prompts:
            #only add prompt to the missing if it was not evaluated on this model previously and it is not the gpt4 prompt
            if(unq_prompt not in model_prompt_dict[unq_models[i]] and 
               "given the following categories, which tag is the most fitting out of the provided ones for each category? Return only the most fitting tag name. Do not repeat the question. " not in unq_prompt):

                missing_prompts[unq_models[i]].append(unq_prompt)
    return missing_prompts

def rename():
    """
    renaming of files.

    Args:

    Returns:

    """
    path = "/base/out/"
    files = os.listdir(path)

    for file in files:
        if(".json" not in file):
            continue
        number = int(file.replace("predictions_","").replace(".json","")) + 50
        os.rename(path + file,path+"predictions_" + str(number) + ".json")

def create_result_comparison():
    """
    saves an overview of different evaluation runs to an excel file and 
    creates a bar chart of the maximum accuracy per mmodel and categoryfor simple comparison

    Args:

    Returns:

    """
    
    #get the dict of all evaluation files
    results = analyse_results()


    df = pd.DataFrame(results)
    unique_models = df['model'].unique()

    #create dictionary with key: model name, value: (key: category name, value: maxscore per category)
    max_dict = {model : {} for model in unique_models}
    
    for model in unique_models:
        filtered_rows = df[df['model'] == model]
        filtered_dataset = filtered_rows[filtered_rows['dataset'] == "txt"]
        unique_categories = filtered_dataset['category'].unique()
        for category in unique_categories:            
            #maximum per model and category
            filtered_rows = filtered_dataset[filtered_dataset['category'] == category]
            max_ = filtered_rows["accuracy"].max()
            max_dict[model][category] = max_


    create_bar_chart(max_dict)
    # Write DataFrame to Excel file
    excel_filename = 'results_new_zero_shot_similarity.xlsx'
    df.to_excel(excel_filename, index=False)

    #automatically includes a link to the coresponding mismatch graphic
    
    # from openpyxl import Workbook
    # from openpyxl.utils.dataframe import dataframe_to_rows
    # wb = Workbook()
    # ws = wb.active

    # # Write DataFrame to worksheet
    # for r in dataframe_to_rows(df, index=False, header=True):
    #     ws.append(r)

    # # Add hyperlinks to file
    # for row in ws.iter_rows(min_row=2, min_col=len(df) + 1, max_row=len(df) + 2, max_col=len(df.columns)):
    #     for cell in row:
    #         file_path = "/home/ge32buc/base/out/analysed_predictions" + results["filename"] + "#f"./{excel_filename}"
    #         cell.hyperlink = file_path
    #         cell.style = "Hyperlink"

    # # Save Excel file
    # wb.save(excel_filename)


    return results
# create_result_comparison()

def get_missing_from_gpt4():
    """
    gets the files missing in a gpt4 prediction file. writes result to file

    Args:

    Returns:

    """

    #path to imagedirectory
    image_path = "/home/ge32buc/base/rosbag_validation_257_anon"
    images = os.listdir(image_path)

    #path to prediction file
    pred_path = "/home/ge32buc/base/out/new/predictions_80.json"
    data = ""
    with open(pred_path,"r") as f:
        data=f.read()
    with open(pred_path, 'r') as file:
        data = json.load(file)
    names = list(data["prediction"].keys())
    
    missing = []
    for image in images:
        if(image not in names):
            missing.append(image)
    str_missing = "\n".join(missing)
    with open("missing.txt","w") as f:
        f.write(str_missing)
# get_missing_from_gpt4()

def convert_old_keys_to_new():
    """
    converts old filename to new.

    Args:

    Returns:

    """
    
    #path of prediction file
    file_path = "/home/ge32buc/base/out/new/predictions_80.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    new_dict = {"config":data["config"]}
    new_dict["prediction"] = old_name_to_new(data["prediction"])
    with open(file_path, 'w') as json_file:
        json.dump(new_dict, json_file, indent=4)

def convert_to_dict():
    """
    converts the old version of a gpt4 prediction containing the returned dictionary as string to the new one.

    Args:

    Returns:

    """
    
    file_path = "/home/ge32buc/base/out/new/predictions_80.json"
    file_path_ = "/home/ge32buc/base/out/new/predictions_10.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    dt = {}
    for pred in data["prediction"]:
        dt["/home/ge32buc/base/rosbag_validation_257_anon/" + pred] = ast.literal_eval(data["prediction"][pred])
    data["prediction"] = dt

    with open(file_path_, 'w') as json_file:
        json.dump(data, json_file, indent=4)
#convert_to_dict()

# convert_old_keys_to_new()

def convert_txt_prediction_to_json():
    """
    converts all prediction txt files in base_path to json

    Args:

    Returns:

    """
    
    #path to the prediction directory
    base_path = "/home/ge32buc/base/out/new/"
    files = os.listdir(base_path)


    for file_name in files:
        result_dict_json = {"config":{},"prediction":{}}
        if(".txt" not in file_name):
                continue
        config_dict = get_config_from_prediction(base_path + file_name)
        result_dict_json["config"] = config_dict
        gpt4_pred_txt = ""
        with open(base_path + file_name, 'r') as file:
            # Read the content of the file
            gpt4_pred_txt = file.read()

        predictions = gpt4_pred_txt.split("+" * 20 + "PREDICTIONS" + "+" * 20)[1]
        predictions = predictions.split("\n")
        for prediction in predictions:
            if("=" not in prediction):
                continue
            key, value = prediction.split("=")
            result_dict_json["prediction"][key.strip()] = value.strip()
        
        with open(base_path + file_name.replace(".txt",".json"), 'w') as json_file:
            json.dump(result_dict_json, json_file, indent=4)

# convert_txt_prediction_to_json()
def visualize_wrongs(prediction_path,base_image_path,mismatches,categories):
    """
    visualizes wrong predictions per category, writes filename, prediction text on the graphic.

    Args:
        prediction_path (string): path to save visualization
        base_image_path (string): base image path
        mismatches (dictionary): key: category names. value: list,string filename
        categories (list,string): category names

    Returns:
        list,float: softmax over similarity between output and rach of the tags specified in options.

    """
    import random
    for category in categories:

        mismatches_per_category = mismatches[category]
        
        #if more than 50 wrong, select 50 randomly for display
        if(len(mismatches_per_category) > 50):
            random.shuffle(mismatches_per_category)
            mismatches_per_category = mismatches_per_category[:50]

        # Open all images and store them in a list
        images = [Image.open(base_image_path + mismatches_per_category[i][0]) for i in range(len(mismatches_per_category))]

        # Calculate the dimensions of the canvas for the composition
        # Here, we arrange images in a grid (10 images per row)
        num_images = len(images)
        images_per_row = 10
        max_rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division to determine number of rows
        canvas_width = images_per_row * images[0].width
        canvas_height = max_rows * images[0].height

        # Create a new blank image canvas for the composition
        composed_image = Image.new('RGB', (canvas_width, canvas_height))

        # Paste each image into the canvas at the appropriate position
        for i, image in enumerate(images):
            draw = ImageDraw.Draw(image)

            # Choose a font (you can specify the font file path and font size)
            font_size = 36
            font = ImageFont.load_default()  # You can specify a custom font here
            font = font.font_variant(size=font_size)

            # Calculate text size and position
            #text_width, text_height = draw.textlength(text_to_write, font=font)
            text_x = 5#(image.width - text_width) // 2  # Center the text horizontally
            text_y = 5#(image.height - text_height) // 2  # Center the text vertically

            # Draw text on the image
            draw.text((text_x, text_y), mismatches_per_category[i][1], font=font, fill="red")

            row = i // images_per_row
            col = i % images_per_row
            x_offset = col * images[0].width
            y_offset = row * images[0].height
            composed_image.paste(image, (x_offset, y_offset))


        # # Save the composed image to a file
        # import time
        # start = time.time()
        path = prediction_path + category + "_wrong__.jpg"
        #print(path)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        composed_image.save(path)