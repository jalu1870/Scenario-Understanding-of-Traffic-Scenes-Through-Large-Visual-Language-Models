from edgar_data_sdk.authentication import EdgarAuthenticator
from edgar_data_sdk.authentication.token_saver import KeyringTokenSaver
from edgar_data_sdk.connection import EdgarDataConnection
import edgar_data_sdk.cli.cli as sdk

import os
import ruamel.yaml
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
import rosbags


from tqdm import tqdm
from PIL import Image
import numpy as np
import sys


EDGAR_API_URL = "https://api.edgar.cps.cit.tum.de"

authenticator = EdgarAuthenticator(
        endpoint_url=EDGAR_API_URL, token_saver=KeyringTokenSaver(EDGAR_API_URL)
    )
connection = EdgarDataConnection(authenticator=authenticator, base_url=EDGAR_API_URL)
rides = connection.list_rides()

bags_ = []
normalized_folders = []

bags_dict = {}
for ride in rides:
    bags = connection.list_scenes(ride)
    bags_dict[ride] = bags
    for bag in bags:
        bags_.append(ride + "/" + bag)

        path = ride + "/" + bag
        path = path.split(".")[0].split("_")
        path = "_".join(path[:-1])

        #
        if(path not in normalized_folders):
            normalized_folders.append(path)

def get_images(imagedata,file):
    arr = np.empty([1200,1920,3],dtype=np.uint8)
    file = "/".join(file.split(".")[:-1])
    local_path_split = (local_path_).split("\\")
    local_path = "/".join(local_path_split) + "/"
    for split_ in range(len(local_path_split)):
        if(not os.path.exists("/".join(local_path_split[:split_ + 1]))):
            os.mkdir("/".join(local_path_split[:split_ + 1]))
    for frame_id in imagedata.keys():
        frame_path = local_path + frame_id + "/"
        if(not os.path.exists(frame_path)):
            os.mkdir(frame_path)
        framecount = 0
        for image in imagedata[frame_id]:

            y = np.array(image)[1::2]#.reshape([1200,1920])#[int(imagedata[-1].shape[0]/2):]
            u = np.array(image)[0::4].reshape(-1,1)
            v = np.array(image)[2::4].reshape(-1,1)

            u_copy = u.copy()
            v_copy = v.copy()
            u = np.append(u,u_copy,axis=1)
            v = np.append(v,v_copy,axis=1)
            arr[:,:,0] = y.reshape([1200,1920])
            arr[:,:,1] = u.reshape([1200,1920])
            arr[:,:,2] = v.reshape([1200,1920])
            im = Image.fromarray(arr,mode="YCbCr")#.reshape(3,1200,1920))
            im.save(frame_path + str(framecount) + ".jpg", "JPEG")
            framecount += 1
    
local_tmp_path = "C:\\desktop\\temp_bags\\"
def read_bag():
    imagedata = {}
    types = []
    # create reader instance and open for reading
    with AnyReader([Path(local_tmp_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
        messages = reader.messages(connections=connections)
        for connection, timestamp, rawdata in tqdm(messages):
            if("novatel_oem7_msgs/" in connection.msgtype or "gps_msgs" in connection.msgtype or "rosbag2_interfaces" in connection.msgtype):
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            # if(msg.__msgtype__ not in ["sensor_msgs/msg/NavSatFix","nav_msgs/msg/Odometry"]):
            #     print(msg.__msgtype__)
            if(msg.__msgtype__ not in types):
                types.append(msg.__msgtype__)
            if(msg.__msgtype__=='sensor_msgs/msg/Image'):
                #imagedata.append(msg.data)
                if(msg.encoding != "yuv422"):
                    continue
                frame_id = msg.header.frame_id
                if(frame_id not in imagedata):
                    imagedata[frame_id] = []
                imagedata[frame_id].append(msg.data)
    print(types)
    return imagedata


"""
def download_metadata(files):
    s3.download(files,local_tmp_path)
    with open(local_tmp_path + 'metadata.yaml') as fp:
        data = yaml.load(fp)
    if(data["rosbag2_bagfile_information"]["storage_identifier"] == ""):
        data["rosbag2_bagfile_information"]["storage_identifier"] = "mcap"
    with open(local_tmp_path + 'metadata.yaml', "w") as f:
        yaml.dump(data, f)
def correct_file_paths(file):
    with open(local_tmp_path + 'metadata.yaml') as fp:
        data = yaml.load(fp)
    data["rosbag2_bagfile_information"]["relative_file_paths"] = [file.split("/")[-1]]
    with open(local_tmp_path + 'metadata.yaml', "w") as f:
        yaml.dump(data, f)

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

count = 0
go = False


rosfiles = []
local_path__ = "E:\\rosbags\\raw-rosbags\\"

path = ""
for bag in bags_dict:
    for mcap in bags_dict[bag]:
        if(".db3" in mcap):
            continue
        local_path_ = local_path__ + bag + "\\" + mcap.replace(".mcap","").replace(".db3","")
        if(os.path.exists(local_path_)):
            count += 1
            continue

        correct_file_paths(mcap)
        connection = EdgarDataConnection(authenticator=authenticator, base_url=EDGAR_API_URL)
        # try:
        #     sensors = connection.list_scene_sensors(bag,mcap.replace(".mcap",""))
            
        #     sensor_data = connection.list_scene_sensor_data
            
        connection.save_scene(bag,mcap,local_tmp_path)
    
        imagedata = read_bag()
        get_images(imagedata,mcap)
        os.remove(local_tmp_path + mcap.split("/")[-1])
        

# def recursive_s3_ls(path):
#     content = s3.ls(path)
#     count = 0
#     for content_ in content:
#         ind = content_.find(".")
#         if(ind == -1):
#             recursive_s3_ls(content_)
#         #is file
#         file_ending = content_.split(".")[-1]
#         if(file_ending in ["db3","mcap"]):
#             rosfiles.append(rosfiles)
#             local_path_file_split = ("E:\\rosbags\\" + content_).split(".")
#             local_path_split = local_path_file_split[:-1]
#             local_path = "/".join(local_path_split) + "/"

#             end_split = local_path.split("/")[-2] + "$$"
#             nr_end_split = end_split.split("_")[-1]
#             #rep = "_".join(end_split)
#             rep = end_split.replace("_" + nr_end_split,"") + "/"
            
            
#             if(rep not in local_path):
#                 local_path = local_path.replace("\\","/")
#                 local_path = local_path.split("/")
#                 local_path.insert(-2,rep)
#                 local_path = "/".join(local_path)

#             print(local_path)
#             if(os.path.exists(local_path)):
#                 count += 1
#                 continue
#             correct_file_paths(content_)
#             s3.download(content_,local_tmp_path)
#             imagedata = read_bag()
#             get_images(imagedata,content_)
#             os.remove(local_tmp_path + content_.split("/")[-1])

#         if(file_ending == "yaml"):
#             download_metadata(content_)


# #recursive_s3_ls("raw-rosbags/")

# print("done")

"""