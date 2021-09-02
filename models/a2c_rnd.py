from torch import nn
import torch
from src.utils import *
import torch.nn.functional as F


class SimplePointNet(nn.Module):
    def __init__(self, in_channels = 2, feature_num = 64):
        super(SimplePointNet, self).__init__()
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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        sp = 64

        g1 = 8
        g2 = 8

        f1 = 128
        hh = 128
        f2 = 64
        f3 = 64

        f4 = 16
        #f5 = 64

        self.pointnet = SimplePointNet(feature_num = sp)

        self.odom_net = nn.Sequential(
                            nn.Linear(3, g1),                       nn.Sigmoid(),
                            nn.Linear(g1,g2),                        nn.ReLU(),
                            )

        self.net_prior = nn.Sequential(
                            nn.Linear(sp + g2, f1),                   nn.ReLU(),
                            )

        self.lstm = nn.LSTMCell(f1, hh)

        self.net_post = nn.Sequential(
                            nn.Linear(hh, f2),                        nn.ReLU(),
                            nn.Linear(f2, f3),                   nn.ReLU(),
                            )

        self.net_actor = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,action_dim),                nn.Softmax(0)
                            )

        self.net_critic_int = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,1)
                            )

        self.net_critic_ext = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,1)
                            )

    def forward(self, x):
        x, (hx, cx) = x
        o_odom  = x[0:3]
        o_lidar = x[3:]
        o_lidar = o_lidar.view(1,2,-1)

        obs = self.pointnet(o_lidar).squeeze()
        pose = self.odom_net(o_odom) 

        net_input = torch.cat( (obs, pose) )
        z = self.net_prior(net_input)


        z_e = z.view(-1,z.size(0))
        hx, cx = self.lstm(z_e, (hx, cx))

        z = hx.squeeze()
        z = self.net_post(z)

        return self.net_actor(z), self.net_critic_int(z), self.net_critic_ext(z), (hx, cx), pose.detach()


class RND(nn.Module):
    def __init__(self, state_dim = 16, k = 8, eta = 8):
        super(RND, self).__init__()      
        self.first = True
        self.state_dim = state_dim
        f1 = state_dim
        f2 = 64
        f3 = 32
        f4 = 16
        self.k = k
        self.eta = eta


        self.target =  nn.Sequential(
                            nn.Linear(f1, k),                   nn.Sigmoid(),
                            #nn.Linear(f2, f3),                   nn.Sigmoid(),
                            #nn.Linear(f1, f4),                    nn.Sigmoid(),
                            #nn.Linear(f3, k),                    nn.Sigmoid()
                            )  

        self.predictor = nn.Sequential(
                            nn.Linear(f1, k),                   nn.Sigmoid(),
                            #nn.Linear(f2, f3),                   nn.Sigmoid(),
                            #nn.Linear(f1, f4),                    nn.Sigmoid(),
                            #nn.Linear(f3, k),                    nn.Sigmoid()
                            )     


        self.predictor.apply(weights_init)
        self.target.apply(weights_init)

        for param in self.target.parameters():
            param.requires_grad = False


    # def init(self):
    #     SAMPLE_SIZE = 10_000
    #     predictor_opt = torch.optim.Adam(self.predictor.parameters(), lr = 1e-2)
    #     target_opt = torch.optim.Adam(self.target.parameters(), lr = 1e-2)

    #     pred_data_in = torch.rand(SAMPLE_SIZE, self.state_dim) 
    #     target_data_in = torch.rand(SAMPLE_SIZE, self.state_dim)

    #     pred_data_out = torch.zeros(SAMPLE_SIZE, self.k)
    #     target_data_out = torch.ones(SAMPLE_SIZE, self.k)

    #     pred_loss = F.binary_cross_entropy(self.predictor(pred_data_in), pred_data_out)
    #     predictor_opt.zero_grad()
    #     pred_loss.backward()
    #     predictor_opt.step()

    #     target_loss = F.binary_cross_entropy(self.target(target_data_in), target_data_out)
    #     target_opt.zero_grad()
    #     target_loss.backward()
    #     target_opt.step()

    #     for param in self.target.parameters():
    #         param.requires_grad = False

    def reset(self):
        #for param in self.target.parameters():
        #    param.requires_grad = True
        self.predictor.apply(weights_init)
        self.target.apply(weights_init)
        #self.init()

    def forward(self, x):
        to = self.target(x)
        po = self.predictor(x)

        mse = self.eta * (to - po).pow(2).sum(0) / self.k * 2

        int_reward =  mse.detach().float().unsqueeze(0)

        return to, po, int_reward

