
import torch
import torch.nn.functional as F
import numpy as np
import random
import itertools as it
#for input
trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

trunk_ori_index_k400 = [1,2,3,4,5]
left_hand_ori_index_k400 = [6,8,10]
right_hand_ori_index_k400 = [7,9,11]
left_leg_ori_index_k400 = [12, 14, 16]
right_leg_ori_index_k400 = [13, 15, 17]

trunk_k400 = [i - 1 for i in trunk_ori_index_k400]
left_hand_k400 = [i - 1 for i in left_hand_ori_index_k400]
right_hand_k400 = [i - 1 for i in right_hand_ori_index_k400]
left_leg_k400 = [i - 1 for i in left_leg_ori_index_k400]
right_leg_k400 = [i - 1 for i in right_leg_ori_index_k400]
body_parts_k400 = [trunk_k400, left_hand_k400, right_hand_k400, left_leg_k400, right_leg_k400]


part_num = len(body_parts)

import itertools
import torch
import math
from torch import nn
import numpy as np
from tqdm import tqdm


def weight_calcu(model, data, label, spa_mask_list, average_motion):
    model.eval()
    replaced_idx = np.array(spa_mask_list) # e.g., [2,4,0]
    remain_idx = list(set(range(part_num)).difference(set(spa_mask_list)))
    m = nn.Softmax(dim=1)
    r = nn.ReLU()

    start_t = 0
    end_t = 64
    step_t = 64

    average_data = torch.from_numpy(average_motion).cuda()

    weight_value = []

    with torch.no_grad():
        
        data = data.float()
        label = label.long()
        
        unchanged_data = data.clone()
        first_frame_data = average_data.unsqueeze(0) #1,C,T,V,M
        for start_temp in range(start_t,end_t,step_t):
            end_temp = start_temp+step_t
            data = unchanged_data.clone()
                
            data_with_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            data_wo_i_part = first_frame_data.clone()
            for i in replaced_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts[i],:] = data[:,:,start_temp:end_temp,body_parts[i],:]
            
            logits_with = m(model(data_with_i_part))#N,class
            logits_wo = m(model(data_wo_i_part)).repeat(data.shape[0],1)#N,class
            weight_value.append(r(torch.gather(logits_with, dim=1, index=label.unsqueeze(-1)) - torch.gather(logits_wo, dim=1, index=label.unsqueeze(-1))))
    model.train()    
    return weight_value[0].squeeze(-1)

def weight_calcu_k400(model, data, label, spa_mask_list, average_motion):
    model.eval()
    replaced_idx = np.array(spa_mask_list) # e.g., [2,4,0]
    remain_idx = list(set(range(part_num)).difference(set(spa_mask_list)))
    m = nn.Softmax(dim=1)
    r = nn.ReLU()

    start_t = 0
    end_t = 64
    step_t = 64

    average_data = average_motion

    weight_value = []

    with torch.no_grad():
        
        data = data.float()
        label = label.long()
        
        unchanged_data = data.clone()
        first_frame_data = average_data.unsqueeze(0) #1,C,T,V,M
        for start_temp in range(start_t,end_t,step_t):
            end_temp = start_temp+step_t
            data = unchanged_data.clone()
                
            data_with_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            data_wo_i_part = first_frame_data.clone()
            for i in replaced_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts_k400[i],:] = data[:,:,start_temp:end_temp,body_parts_k400[i],:]
            
            logits_with = m(model(data_with_i_part))#N,class
            logits_wo = m(model(data_wo_i_part)).repeat(data.shape[0],1)#N,class
            weight_value.append(r(torch.gather(logits_with, dim=1, index=label.unsqueeze(-1)) - torch.gather(logits_wo, dim=1, index=label.unsqueeze(-1))))
    model.train()    
    return weight_value[0].squeeze(-1)

def weight_calcu_shapley(model, data, label, spa_mask_list, average_motion):
    model.eval()
    replaced_idx = np.array(spa_mask_list) # e.g., [2,4,0]
    n = random.randint(0, 5 - len(replaced_idx)) 
    remain_idx = random.sample(list(set(range(part_num)).difference(set(spa_mask_list))), n)
    
    m = nn.Softmax(dim=1)
    r = nn.ReLU()

    start_t = 0
    end_t = 64
    step_t = 64

    average_data = torch.from_numpy(average_motion).cuda()

    weight_value = []

    with torch.no_grad():
        
        data = data.float()
        label = label.long()
        
        unchanged_data = data.clone()
        first_frame_data = average_data.unsqueeze(0) #1,C,T,V,M
        for start_temp in range(start_t,end_t,step_t):
            end_temp = start_temp+step_t
            data = unchanged_data.clone()
                
            data_with_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            data_wo_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            for i in replaced_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts[i],:] = data[:,:,start_temp:end_temp,body_parts[i],:]
            for i in remain_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts[i],:] = data[:,:,start_temp:end_temp,body_parts[i],:]
                data_wo_i_part[:,:,start_temp:end_temp,body_parts[i],:] = data[:,:,start_temp:end_temp,body_parts[i],:]
            
            logits_with = m(model(data_with_i_part))#N,class
            logits_wo = m(model(data_wo_i_part))#N,class
            weight_value.append(r(torch.gather(logits_with, dim=1, index=label.unsqueeze(-1)) - torch.gather(logits_wo, dim=1, index=label.unsqueeze(-1))))
    model.train()    
    return weight_value[0].squeeze(-1)

def weight_calcu_shapley_k400(model, data, label, spa_mask_list, average_motion):
    model.eval()
    replaced_idx = np.array(spa_mask_list) # e.g., [2,4,0]
    n = random.randint(0, 5 - len(replaced_idx)) 
    remain_idx = random.sample(list(set(range(part_num)).difference(set(spa_mask_list))), n)
    
    m = nn.Softmax(dim=1)
    r = nn.ReLU()

    start_t = 0
    end_t = 64
    step_t = 64

    average_data = average_motion

    weight_value = []

    with torch.no_grad():
        
        data = data.float()
        label = label.long()
        
        unchanged_data = data.clone()
        first_frame_data = average_data.unsqueeze(0) #1,C,T,V,M
        for start_temp in range(start_t,end_t,step_t):
            end_temp = start_temp+step_t
            data = unchanged_data.clone()
                
            data_with_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            data_wo_i_part = first_frame_data.clone().repeat(data.shape[0],1,1,1,1)
            for i in replaced_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts_k400[i],:] = data[:,:,start_temp:end_temp,body_parts_k400[i],:]
            for i in remain_idx:
                data_with_i_part[:,:,start_temp:end_temp,body_parts_k400[i],:] = data[:,:,start_temp:end_temp,body_parts_k400[i],:]
                data_wo_i_part[:,:,start_temp:end_temp,body_parts_k400[i],:] = data[:,:,start_temp:end_temp,body_parts_k400[i],:]
            
            logits_with = m(model(data_with_i_part))#N,class
            logits_wo = m(model(data_wo_i_part))#N,class
            #N, 
            weight_value.append(r(torch.gather(logits_with, dim=1, index=label.unsqueeze(-1)) - torch.gather(logits_wo, dim=1, index=label.unsqueeze(-1))))
    model.train()    
    return weight_value[0].squeeze(-1)