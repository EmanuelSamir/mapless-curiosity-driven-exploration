from os import lseek
from torch import nn
from torch.autograd import Variable
import torch
import gym
import time
import numpy as np
from models.ppo import PPO
from tqdm import tqdm
from src.utils import *
from src.sim2sim import *
from src.logger_special import LoggerSpecial
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class PPOAgent:
    def __init__(self, env, obs_dim, action_dim, actions, 
                n_episodes = 1_000, 
                gamma = 0.999,
                gamma_rnd = 0.999,
                lr = 1e-3,
                lr_rnd = 1e-1,
                load_model_fn = None,
                trial = False,
                intrinsic_set = True,
                save_best = True,
                save_checkpoints = True,
                checkpoint_every = 50,
                n_opt = 10,
                lapse_rnd = 4,
                zeta = 1e-1,
                beta = 1e1
                ):

        # General parameters
        self.device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.gamma_rnd = gamma_rnd
        self.trial = trial
        self.save_best = save_best
        self.save_checkpoints = save_checkpoints
        self.checkpoint_every = checkpoint_every 
        self.intrinsic_set = intrinsic_set
        self.lr = lr
        self.lr_rnd = lr_rnd
        self.zeta = zeta
        self.beta = beta
        self.n_opt = n_opt
        self.load_model_fn = load_model_fn
        self.lapse_rnd = lapse_rnd

        self.episode = 0
        self.steps = 0

        # Discrete actions
        self.actions = actions

        # LSTM param
        self.hx = None
        self.cx = None

        # Models
        self.ppo = PPO(obs_dim, action_dim)
        self.ppo_opt = torch.optim.Adam(self.ppo.parameters(), lr = self.lr)
        self.load_models(load_model_fn)

        # RND - Curious Module
        #self.rnd = RND(state_dim=8) # 256 + 32
        #self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr = self.lr_rnd )
        self.features_ns = None


        # Loggers:
        #if not self.trial:
        #    self.logger = Logger("A2CiLSTMRNDPN", 
        #                         self.save_best,
        #                         self.save_checkpoints,
        #                         self.checkpoint_every)

        # self.logger_special = LoggerSpecial("A2CiLSTMRNDPN")


    def load_models(self, model_fn = None):
        # ActorCritic loading
        if model_fn:
            try:
                model_path = os.path.join("../checkpoints", "PPO", model_fn)
                checkpoint = torch.load(model_path)
                self.ppo.load_state_dict(checkpoint["ppo_state_dict"])
                self.ppo_opt.load_state_dict(checkpoint["ppo_state_dict"])
            except:
                raise Exception("Model filename is incorrect")


    def train(self, comment = ''):
        pbar = tqdm(total=self.n_episodes, position=0, leave=True)
        # if not self.trial:
        #     self.logger.set_description(    comment,
        #                                     self.lr,
        #                                     self.lr_rnd,
        #                                     self.zeta,
        #                                     self.beta,
        #                                     self.n_episodes,
        #                                     self.gamma,
        #                                     self.actor_critic,
        #                                     self.rnd,
        #                                     self.actions,
        #                                     self.load_model_fn)

        # self.logger_special.set_description(comment)

        try:
            for episode in range(self.n_episodes):
                
                # Reset environment 
                obs = self.env.reset()
                is_done = False
                self.episode = episode

                # LSTM
                self.lstm_hx = Variable(torch.zeros(1, self.ppo.lstm_hh))
                self.lstm_cx = Variable(torch.zeros(1, self.ppo.lstm_hh))

                # Env
                self.steps = 0

                # Reset
                while not is_done:
                    # Feed Policy network
                    probs, _, _, (lstm_hxn, lstm_cxn), _ = self.ppo((  tn(obs), (self.lstm_hx, self.lstm_cx) ))

                    # Choose sample accoding to policy
                    action_dist = torch.distributions.Categorical(probs = probs)
                    action = action_dist.sample()
                    action_ix = action.detach().data


                    # Update env
                    if self.actions:
                        next_obs, ext_reward, is_done, _ = self.env.step(self.actions[action_ix])
                    else:
                        next_obs, ext_reward, is_done, _ = self.env.step(action_ix)

                    
                    # Advantage 
                    _, nVi, nVe, _, self.features_ns = self.ppo(( tn(next_obs), (lstm_hxn, lstm_cxn) ))
                    _, Vi, Ve, _, _ = self.ppo(( tn(obs), (self.hx, self.cx) ))

                    #_, _, int_reward = self.rnd(self.features_ns)
                    int_reward = 0
                    int_reward = torch.clamp(int_reward, min = 0, max = 8)


                    advantage_ext = ext_reward + (1-is_done)*(self.gamma * nVe) - Ve 
                    advantage_int = int_reward + (1-is_done)*(self.gamma_rnd * nVi) - Vi 

                    if self.intrinsic_set:
                        advantage = advantage_ext + advantage_int 
                    else:
                        advantage = advantage_ext
                        int_reward = torch.tensor([0])

                    # Update models
                    v_loss, pi_loss, ent_loss = self.update_models(advantage, action_dist, action, probs)

                    # Record losses and reward
                    if not self.trial:
                        self.logger.update( action_ix,
                                            int_reward.detach(),
                                            ext_reward,
                                            v_loss,
                                            ent_loss,
                                            pi_loss)

                    self.logger_special.update(self.steps, self.features_ns.tolist())

                    # Env update
                    obs = next_obs
                    self.steps += 1

                    # LSTM update cell
                    self.hx = Variable(lstm_hxn.data)
                    self.cx = Variable(lstm_cxn.data)

                #self.rnd.reset()
                #if not self.trial:
                #    self.logger.consolidate(self.steps, self.episode, self.ppo, self.ppo_opt, self.rnd)
                #self.logger_special.consolidate(episode)
                pbar.update()
            
            #if not self.trial:
            #    self.logger.close()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        #finally:
            #if not self.trial:
            #    self.logger.exception_arisen(self.episode, self.actor_critic, self.actor_critic_optimizer)
            
        pbar.close()



    def update_models(self, advantage, action_dist, action, probs):
        # Actor Critic update
        value_loss = self.zeta * advantage.pow(2).mean() 
        policy_loss = - action_dist.log_prob(action) * advantage.detach()

        entropy_loss = - self.beta * (action_dist.log_prob(action) * probs).mean()
        #print('log_action = {}, probs {}'.format(action_dist.log_prob(action), probs ) )
        #print(' VL = {}, PL = {}, EL = {}'.format(value_loss, policy_loss, entropy_loss))

        ppo_loss = value_loss + policy_loss + entropy_loss

        self.ppo_opt.zero_grad()
        ppo_loss.backward()
        self.ppo_opt.step()

        # RND Update
        if self.intrinsic_set:
            x = self.features_ns.unsqueeze(0)
            xs = torch.repeat_interleave(x, self.n_opt, dim=0)

            if self.steps % self.lapse_rnd == 0:
                for x_i in xs:
                    t, p, _ = self.rnd(x_i)
                    int_loss = F.binary_cross_entropy(p, t.detach())
                    self.rnd_optimizer.zero_grad()
                    int_loss.backward()
                    self.rnd_optimizer.step()
        
        return value_loss.detach(), policy_loss.detach(), entropy_loss.detach()

    def test(self, going_back_start = False ):

        cond = True

        while cond:
            if not going_back_start:
                obs = self.env.reset()

            else:
                # Move the robot back in the simulation
                self.env.robot.xr = self.env.xr0
                self.env.robot.yr = self.env.yr0
                self.env.robot.thr = self.env.thr0

                self.env.steps = 0
                self.env.final_obs = 0

                # Transform robot pose in t wrt robot initial frame
                I_T_r = pose2mat_2d(self.env.robot.xr, self.env.robot.yr, self.env.robot.thr)
                I_T_r0 = pose2mat_2d(self.env.xr0, self.env.yr0, self.env.thr0)
                r0_T_I = inverse_mat_2d(I_T_r0)
                r0_T_r = r0_T_I.dot(I_T_r)
                xr_r0, yr_r0, thr_r0 = mat2pose_2d(r0_T_r)		

                robot_pos = np.array([xr_r0, yr_r0, thr_r0])

                r_T_I = inverse_mat_2d(I_T_r)

                # Transform goal position to robot inertia frame
                I_T_g = pose2mat_2d(self.env.robot.xg, self.env.robot.yg, self.env.robot.thg)
                r_T_g = r_T_I.dot(I_T_g)
                xg_r, yg_r, thg_r = mat2pose_2d(r_T_g)

                self.env.robot_goal = 	np.array([xg_r, yg_r, thg_r]) \
                                    if self.env.is_rotated else  \
                                    np.array([xg_r, yg_r])

                # Transform lidar to robot inertia frame
                touches = self.env.robot.scanning()
                
                xls = self.env.robot.xls[touches]
                yls = self.env.robot.yls[touches]

                if xls.size == 0:
                    xls = np.array([self.env.robot.max_range])
                    yls = np.array([0.])

                xls_r = []
                yls_r = []


                for xl, yl in zip(xls, yls):
                    I_T_l = pose2mat_2d(xl, yl, 0)
                    r_T_l = r_T_I.dot(I_T_l)
                    xl_r, yl_r, _ = mat2pose_2d(r_T_l)
                    xls_r.append(xl_r)
                    yls_r.append(yl_r)

                xls_r = np.array(xls_r)
                yls_r = np.array(yls_r)

                obs = np.concatenate( (xls_r, yls_r) )

                obs = np.concatenate( (robot_pos ,obs)) 
            
                
            #self.env.render()

            # Save for recording
            #print("Press enter")
            #_ = input()

            is_done = False
            self.hx = Variable(torch.zeros(1, self.ppo.lstm_hh))
            self.cx = Variable(torch.zeros(1, self.ppo.lstm_hh))
            self.steps = 0

            # Save for sim2sim
            #save_object_poses(self.env)
            #print(self.env.xr0, self.env.yr0, self.env.thr0)

            x_lst = []
            y_lst = []
            th_lst = []

            while not is_done:
                x_lst.append(self.env.robot.xr), y_lst.append(self.env.robot.yr), th_lst.append(self.env.robot.thr)

                # Feed Policy network
                probs, _, _, (hxn, cxn), _ = self.actor_critic((  tn(obs), (self.hx, self.cx) ))

                #print('LSTM hxn mean {} std {} y cxn mean {} std {} '.format(torch.mean(hxn), torch.std(hxn), torch.mean(cxn), torch.std(cxn) ))

                # Choose sample accoding to policy
                action_dist = torch.distributions.Categorical(probs = probs)
                action = action_dist.sample()
                action_ix = action.detach().data

                if self.actions:
                    next_obs, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
                else:
                    next_obs, ext_reward, is_done, info = self.env.step(action_ix)

                _, nVi, nVe, _, self.features_ns = self.actor_critic(( tn(next_obs), (hxn, cxn) ))
                _, Vi, Ve, _, _ = self.actor_critic(( tn(obs), (self.hx, self.cx) ))

                #_, _, int_reward = self.rnd(self.features_ns)
                
                #int_reward = torch.clamp(int_reward, min = 0, max = 8)

                int_reward = 0

                advantage_ext = ext_reward + (1-is_done)*(self.gamma * nVe) - Ve 
                advantage_int = int_reward + (1-is_done)*(self.gamma_rnd * nVi) - Vi 

                advantage = advantage_ext + advantage_int

                #print('int rew: {}, ext rew : {} '.format(int_reward, ext_reward))

                #value_loss = self.zeta * advantage.pow(2).mean() 
                #policy_loss = - action_dist.log_prob(action) * advantage.detach()

                #entropy_loss = - self.beta * (action_dist.log_prob(action) * probs).mean()
                #print('log_action = {}, probs {}'.format(action_dist.log_prob(action), probs ) )
                #print('VL = {}, PL = {}, EL = {}'.format(value_loss, policy_loss, entropy_loss))

                obs = next_obs
                self.lstm_hx = Variable(hxn.data)
                self.lstm_cx = Variable(cxn.data)

                # RND Update
                #x = self.features_ns.unsqueeze(0)
                #xs = torch.repeat_interleave(x, self.n_opt, dim=0)

                # if self.steps % self.lapse_rnd == 0:
                #     for x_i in xs:
                #         t, p, _ = self.rnd(x_i)
                #         int_loss = F.binary_cross_entropy(p, t.detach())
                #         self.rnd_optimizer.zero_grad()
                #         int_loss.backward()
                #         self.rnd_optimizer.step()

                #time.sleep(0.01)
                #self.env.render()
                self.steps += 1


            # Save for sim2sim
            #save_waypoints(x_lst, y_lst, th_lst)


            # Used for recording
            #print("Save waypoints? (y/n)")
            #z = input()
            #self.rnd.reset()
            #self.env.close()


            final_obs = self.env.final_state
            x0, y0, th0 = self.env.xr0, self.env.yr0, self.env.thr0
            xf, yf, thf = self.env.robot.xr, self.env.robot.yr, self.env.robot.thr
            xg, yg = self.env.robot.xg, self.env.robot.yg
            
            
            dist = np.linalg.norm([xg - x0, yg - y0])

            if (final_obs == 2) and dist > 7 and self.steps > 100:
                cond = False

        return x_lst, y_lst, th_lst, self.steps, final_obs, x0, y0, th0, xf, yf, thf, xg, yg
