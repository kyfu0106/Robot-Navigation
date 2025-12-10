from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math 
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='choose the target')
    
    args = parser.parse_args()
    return args

args = parse_args()
target = args.target

# RGB
target_label = {'lamp':(255,152,150),'rack':(181,207,107),'cushion':(189,158,57),'cooktop':(132,60,57),'refrigerator':(206,219,156)}

move_amount = 0.01
rotate_amount = 1
img_array = []

rrt_path = np.load(target+'_path.npy')
route = []
for i in range(rrt_path.shape[0]-1,-1,-1):
    x, y = rrt_path[i]
    x = round((x - 250)*0.0325, 4)  #0.0274
    y = round((300 - y)*0.0255, 4) #0.03266
    route.append([x, y])
route = np.array(route)
start = route[0]
end = route[-1]
'''
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi) * 180
'''
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/habitat/mesh_semantic.ply"
path = "apartment_0/habitat/info_semantic.json"

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=move_amount) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=rotate_amount) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=rotate_amount)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([start[1], 0.0, start[0]])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

       
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        semantic_img = transform_semantic(id_to_label[observations["semantic_sensor"]])
        r, c = np.where(((semantic_img[:,:,0] == target_label[target][2]) & (semantic_img[:,:,1] == target_label[target][1]) & (semantic_img[:,:,2] == target_label[target][0])))
        #cv2.imshow("semantic", semantic_img)
        color_img = transform_rgb_bgr(observations["color_sensor"])
        color_img[r,c,:] = (0,0,255)
        cv2.imshow("RGB", color_img)
        img_array.append(color_img)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        #print("camera pose: x y z rw rx ry rz")
        #print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)


#angle = angle_between((0,0,-1),(route[1][1]-start[1],0,route[1][0]-start[0]))
action_list = []
for idx in range(1, route.shape[0]):
    if idx == 1:
        v1 = (0, -1)
    else:
        v1 = (route[idx-1][1]-route[idx-2][1], route[idx-1][0]-route[idx-2][0])
    v2 = (route[idx][1]-route[idx-1][1], route[idx][0]-route[idx-1][0])
    angle = math.atan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1]) / np.pi * 180
    dist = np.linalg.norm(route[idx]-route[idx-1])
    action_list.append((angle, dist))

print(action_list)

for angle, dist in action_list:
    #angle = math.atan2(v1[1]*v2[0] - v1[0]*v2[1], v1[1]*v2[1] + v1[0]*v2[0]) / np.pi * 180
    if angle > 0:
        keystroke = "turn_right"
    else:
        keystroke = "turn_left"
        angle = -angle
    #print('angle=',angle)
    #print('dist=', dist)
    rot_times = int(angle/rotate_amount)
    for i in range(rot_times):
        cv2.waitKey(1)
        if keystroke == "turn_left":
            action = "turn_left"
            navigateAndSee(action)
            #print("action: LEFT")
        elif keystroke == "turn_right":
            action = "turn_right"
            navigateAndSee(action)
            #print("action: RIGHT")
    move_times = int(dist/move_amount)
    keystroke = 'move'
    for i in range(move_times):
        cv2.waitKey(1)
        if keystroke == 'move':
            action = "move_forward"
            navigateAndSee(action)
            #print("action: FORWARD")

print('Starting create video...')
out = cv2.VideoWriter(target+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (512,512))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print('Finish!')
'''

action = "move_forward"
navigateAndSee(action)

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue
'''