from torch import nn
import torch
from src.utils import *
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channels = 2, feature_num = 64):
        super(PointNet, self).__init__()
        c1 = 16
        c2 = 32
        c3 = 64
        f1 = feature_num


        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels = c1,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv2 = nn.Conv1d(in_channels=c1, out_channels = c2,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv3 = nn.Conv1d(in_channels=c2, out_channels = c3,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(c3, f1)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)
        x = self.fc1(x)
        return x

class PPO(nn.Module):
    def __init__(self, action_dim, pointnet_input_size = 64, odom_input_size = 3):
        super(PPO, self).__init__()
        
        self.pointnet_c1 = pointnet_input_size

        self.odom_g1 = odom_input_size
        self.odom_g2 = 8
        self.odom_g3 = 8

        self.lstm_hh = 128
        self.net_f1 = 128

        self.net_f2 = 64
        self.net_f3 = 64
        self.net_f4 = 16

        self.pointnet = PointNet(feature_num = self.pointnet_c1)

        self.odom_net = nn.Sequential(
                            nn.Linear(self.odom_g1, self.odom_g2),                          nn.Sigmoid(),
                            nn.Linear(self.odom_g2, self.odom_g3),                          nn.ReLU(),
                            )

        self.net_prior = nn.Sequential(
                            nn.Linear(self.pointnet_c1 + self.odom_g3, self.net_f1),        nn.ReLU(),
                            )

        self.lstm = nn.LSTMCell(self.net_f1, self.lstm_hh)

        self.net_post = nn.Sequential(
                            nn.Linear(self.lstm_hh, self.net_f2),                           nn.ReLU(),
                            nn.Linear(self.net_f2, self.net_f3),                            nn.ReLU(),
                            )

        self.net_actor = nn.Sequential(
                            nn.Linear(self.net_f3,self.net_f4),                             nn.ReLU(),
                            nn.Linear(self.net_f4,action_dim),                              nn.Softmax(0)
                            )

        self.net_critic_int = nn.Sequential(
                            nn.Linear(self.net_f3,self.net_f4),                             nn.ReLU(),
                            nn.Linear(self.net_f4,1)
                            )

        self.net_critic_ext = nn.Sequential(
                            nn.Linear(self.net_f3,self.net_f4),                             nn.ReLU(),
                            nn.Linear(self.net_f4,1)
                            )

    def forward(self, x):
        # Divide input with old info (hx, cx)
        x, (hx, cx) = x

        # Separate input
        o_odom  = x[0:self.odom_g1]
        o_lidar = x[self.odom_g1:]
        o_lidar = o_lidar.view(1,2,-1)

        # Feed feature extractor
        obs = self.pointnet(o_lidar).squeeze()
        pose = self.odom_net(o_odom) 

        # Concatenate
        net_input = torch.cat( (obs, pose) )

        # Prior net
        z = self.net_prior(net_input)

        # LSTM
        z_e = z.view(-1,z.size(0))
        hx, cx = self.lstm(z_e, (hx, cx))

        # Post net and actor critic nets
        z = hx.squeeze()
        z = self.net_post(z)

        return self.net_actor(z), self.net_critic_int(z), self.net_critic_ext(z), (hx, cx), pose.detach()
