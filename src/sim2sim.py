from datetime import date
import yaml
import os
import pandas as pd
import numpy as np

def state2pose(x, y, th = 0):
    return [x,y,0, th, 0, 0]# x y z r p y

def save_object_poses(env, case = 'random'):
    path = '../sim2sim'
    path = os.path.join(path, case, 'models.yaml')
    
    try:
        env = env.robot
    except:
        raise Exception('env is not the one')
        return -1

    objects = list()   

    # Saving goal pose. Possible
    goal = {'name':'goal', \
               'type':'sdf',\
               'package':'object_spawner', \
               'pose': state2pose(env.xg, env.yg, env.thg)
        }
    objects.append(goal)

    # Saving obstacles
    for xc, yc in zip(env.env.xcs, env.env.ycs):      
        obs = {
            'name':'obstacle', \
            'type':'sdf',\
            'package':'object_spawner', \
            'pose': state2pose(xc.item(), yc.item())
            }
        objects.append(obs)    

    # Creating Yaml structure
    models = {'models': objects}

    with open(path, 'w') as file:
        documents = yaml.dump(models, file, default_flow_style=None)

def save_waypoints(xs, ys, ths, case = 'random'):
    path = '../sim2sim'
    path = os.path.join(path, case)
    
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'waypoints.csv')

    d = {'x':xs, 'y':ys, 'th':ths}
    df = pd.DataFrame(d)
    df.to_csv(path)
    
def save_episodic_results(episodes, total_steps, final_states, x0s, y0s, th0s, xfs, yfs, thfs, xgs, ygs, case = 'random'):    
    path = '../comparison/episodic'
    path = os.path.join(path, case, 'results.csv')

    d = {"episodes":episodes, "total_steps":total_steps, "final_states":final_states, "x0s":x0s, "y0s":y0s, "th0s":th0s, "xgs":xgs, "ygs":ygs, "xfs":xfs, "yfs":yfs, "thfs":thfs}

    df = pd.DataFrame(d)
    df.to_csv(path)

def save_vertices(xs, ys, ths, episode,  case = 'random'):
    cur_x = 0
    cur_y = 0
    cur_th = 0

    vert_x = []
    vert_y = []

    cond = True

    for x, y, th in zip(xs,ys, ths):
        if cond: # Va en línea recta
            if th != cur_th:
                vert_x.append(x)
                vert_y.append(y)
                cond = False

        else: # Va girando
            if np.linalg.norm([x - cur_x, y - cur_y]) > 0.01:
                cond = True

        cur_x = x
        cur_y = y
        cur_th = th 
    if cond:
        vert_x.append(cur_x)
        vert_y.append(cur_y)

    path = '../comparison/vertices'

    path = os.path.join(path, case, str(episode))

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, 'vertices.csv')
    
    d = {'x':vert_x, 'y':vert_y}
    df = pd.DataFrame(d)
    df.to_csv(path)


