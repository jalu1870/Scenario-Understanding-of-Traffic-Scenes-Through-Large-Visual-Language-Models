import base64
import requests
import time
class GPT4sender():
  
    def __init__(self,config):
        # OpenAI API Key
        self.api_key = ""

    # Function to encode the image
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run(self,image_path,prompt):

        # Getting the base64 string
        base64_image = self.encode_image(image_path)
        print("Warning you are using gpt4, this might cause costs of ~0.02$ per image, comment following line out to use")
        return

        #return "YY"
        #print("image encoded: {}".format(image_path))
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        while("error" in list(response.json().keys()) and response.json()["error"]["code"] == "rate_limit_exceeded"):
            time.sleep(1)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()

# default_prompt = "given the following categories, which tag is the most fitting out of the provided ones for each category? Return only the category names with the most fitting tag name. \
# vision_impairing_brightness = ['yes','no'] \
# weather = ['rainy','snowy','clear','overcast','partly cloudy','foggy','undefined'] \
# urban_environment = ['tunnel', 'residential', 'parking lot', 'city street', 'gas station', 'highway', 'undefined'] \
# time_of_day = ['twilight','daytime','nighttime','undefined']\
# land_use = ['urban area','rural area','suburban area','industrial area','nature']\
# road_condition = ['dry road','wet road','snowy road','icy road','muddy road'] \
# street_configuration = ['one way street','two way street'] \
# lanemarks = ['normal lanemarks','crosswalk','bus lane','no lanemarks']\
# number_of_lanes = ['0','1','2','3','4','5','6'] \
# traffic_scene = ['free-flowing traffic','congested traffic','traffic accident','construction zone']\
# road_intersection = ['yes','no'] \
# number_of_vulnerable_road_users = ['none','few','several','many'] \
# vehicle_maneuver = ['moving forward','stopping','turning','lane changing','parking']\
# traffic_sign_for_ego-vehicle = ['yes', 'no'] \
# traffic_light_for_ego-vehicle = ['yes', 'no']"

