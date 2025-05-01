import s3fs as s3fs
 
import sys
import os
import ruamel.yaml

print(os.getcwd())
#from mcap_ros2.decoder import DecoderFactory

#from mcap.reader import make_reader
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
import rosbags
print(rosbags.__file__)

from tqdm import tqdm
from PIL import Image
import numpy as np


import sys

import numpy as np
access_key = "---------"
secret_key = "---------"
endpoint_url = "---------"
 
kwargs = {
    "key": access_key,
    "secret": secret_key,
    "client_kwargs": {"endpoint_url": endpoint_url},
}
def get_images(imagedata,file):
    arr = np.empty([1200,1920,3],dtype=np.uint8)
    file = "/".join(file.split(".")[:-1])
    local_path_split = ("E:\\" + file).split("/")
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
    # create reader instance and open for reading
    with AnyReader([Path(local_tmp_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
        messages = reader.messages(connections=connections)
        for connection, timestamp, rawdata in tqdm(messages):
            if("novatel_oem7_msgs/" in connection.msgtype or "gps_msgs" in connection.msgtype or "rosbag2_interfaces" in connection.msgtype):
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            if(msg.__msgtype__=='sensor_msgs/msg/Image'):
                #imagedata.append(msg.data)
                if(msg.encoding != "yuv422"):
                    continue
                frame_id = msg.header.frame_id
                if(frame_id not in imagedata):
                    imagedata[frame_id] = []
                imagedata[frame_id].append(msg.data)
    return imagedata

def download_metadata(files):
    s3.download(files[0],local_tmp_path)
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

s3 = s3fs.S3FileSystem(**kwargs)
ros_folders = s3.ls("raw-rosbags/")
count = 0
for ros_folder in ros_folders:
    if("Trash" in ros_folder or ".txt" in ros_folder or "munich_dataset_v1" in ros_folder or "munich_dataset_v2" in ros_folder or "munich_dataset_v3" in ros_folder):
        continue
    ros_recordings = s3.ls(ros_folder)
    
    
    if(ros_recordings[0][-1] != "p" and ros_recordings[0][-1] != "l" and ros_recordings[0][-3:] != "db3"):
        for ros_recording in ros_recordings:
            
            files = s3.ls(ros_recording)
            download_metadata(files)
            for file in files[1:]:
                print(file,count)
                file_split = "/".join(file.split(".")[:-1])
                local_path_split = ("E:\\" + file_split).split("/")
                local_path = "/".join(local_path_split) + "/"
                # local_path_file_split = ("E:\\" + file).split("/")
                # local_path_split = local_path_file_split[:-1]
                # local_path = "/".join(local_path_split) + "/"

                if(os.path.exists(local_path)):
                    count += 1
                    continue
                if(file[-3:] != "cap" and file[-3:] != "db3"):
                    print("WHAT?", file)
                    count += 1
                    continue
                correct_file_paths(file)

                s3.download(file,local_tmp_path)
                imagedata = read_bag()
                get_images(imagedata,file)
                os.remove(local_tmp_path + file.split("/")[-1])

            
                count += 1
    else:
        print("ALLO??")
        print(ros_recordings)
    # else:
    #     print(ros_recordings)
    #     count += len(ros_recordings) - 1
print("done")