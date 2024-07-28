import random
import numpy as np
import torch
import torch.nn.functional as F
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

@torch.no_grad()
def ske_swap_randscale_k400(x, spa_l, spa_u, tem_l, tem_u, p=None):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        if 1:
            Cs = random.randint(spa_l, spa_u)
            # sample the parts index
            parts_idx = random.sample(body_parts_k400, Cs)
            # generate spa_idx
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()

        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        if p==None:
            p = random.random()
        if p > 0.25:
            N, C, T, V, M = xst.size()

            Ct_2 = random.randint(Ct, 25)
            tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
            rt_2 = Ct_2 * tem_downsample_ratio

            xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    xst_temp[randidx][:, :, :, spa_idx, :]
            #xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            #        x[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
            mask = torch.zeros(T // tem_downsample_ratio, V)
            mask[tem_idx:tem_idx + Ct, spa_idx] = 1
        else:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

        return randidx, xst, mask

@torch.no_grad()
def ske_swap_randscale(x, spa_l, spa_u, tem_l, tem_u, p=None):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        if 1:
            Cs = random.randint(spa_l, spa_u)
            # sample the parts index
            parts_idx = random.sample(body_parts, Cs)
            # generate spa_idx
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()

        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        if p==None:
            p = random.random()
        if p > 0.25:
            N, C, T, V, M = xst.size()

            Ct_2 = random.randint(Ct, 16)
            tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
            rt_2 = Ct_2 * tem_downsample_ratio

            xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    xst_temp[randidx][:, :, :, spa_idx, :]
            #xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            #        x[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
            mask = torch.zeros(T // tem_downsample_ratio, V)
            mask[tem_idx:tem_idx + Ct, spa_idx] = 1
        else:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

        return randidx, xst, mask

@torch.no_grad()
def ske_swap_randscale_sample_noweighted(x, spa_l, spa_u, tem_l, tem_u, part_dist, label_dist=None, adatemp=None):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        label dist: for long tailed data augmentation, labe_dist[i] is the frequency of the i-th data's (in a batch) label in the dataset
        '''
        N, C, T, V, M = x.size()
        
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        
        p = random.random()

        if p > 0.25:
            # ------ Spatial ------ #
        
            Cs = random.randint(spa_l, spa_u)
            N, C, T, V, M = xst.size()

            Ct_2 = random.randint(Ct, 16)
            tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
            rt_2 = Ct_2 * tem_downsample_ratio

            xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]
            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            lamb = []
            mapping = [
                        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
                        [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
                        [0, 1, 2], [0,1,3], [0,1,4], [0,2,3], [0,2,4],
                        [0,3,4], [1,2,3], [1,2,4], [1,3,4], [2,3,4],
                ]# body part combination

            all_part = set(range(5))
            replace2remain = {}#map the idx of relaced parts to that of remain parts 
            for replace_idx, item in enumerate(mapping):
                remain_idx = mapping.index(list(all_part.difference(set(item))))
                replace2remain[replace_idx] = remain_idx
            
            temp = 0.2 #temperature hyper-parameter
            part_prob_dist = part_dist/part_dist.sum(dim=1,keepdim=True)
            part_prob_dist = F.softmax(part_dist/temp, dim=1)

            for i in range(N):
                if label_dist != None:
                    if label_dist[i] > label_dist[randidx[i]]:
                        dist = part_prob_dist[randidx[i]]
                        diffset = False
                    else:
                        dist = part_prob_dist[i]
                        diffset = True

                spa_idx = []
                parts_idx = random.choices(list(range(20)), weights=dist, k=1)[0]
                
                p_choice = parts_idx
                if diffset:
                    p_choice = replace2remain[p_choice]
                parts_idx = mapping[p_choice]
                for part_idx in parts_idx:
                    spa_idx += body_parts[int(part_idx)]
                spa_idx.sort()
                xst[i, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                        xst_temp[randidx[i]][:, :, spa_idx, :]
                lamb.append(rt * len(spa_idx) / (T * V))
            lambd = torch.tensor(lamb).reshape((N,1)).cuda()
            
        else:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros(T // tem_downsample_ratio, V) + lamb
            lambd = torch.full((N,1),lamb).cuda()

        #print(lambd.shape)
        return randidx, xst, lambd

@torch.no_grad()
def ske_swap_randscale_sample_noweighted_k400(x, spa_l, spa_u, tem_l, tem_u, part_dist, label_dist=None, adatemp=None):
        '''
        swap a batch skeleton
        T   100 --> 50 --> 25    # 8n
        S   17 --> 17 --> 17 (5 parts)
        label dist: for long tailed data augmentation, labe_dist[i] is the frequency of the i-th data's (in a batch) label in the dataset
        '''
        N, C, T, V, M = x.size()

        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

    
        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        
        p = random.random()

        if p > 0.25:
            # ------ Spatial ------ #
        
            Cs = random.randint(spa_l, spa_u)
            N, C, T, V, M = xst.size()

            Ct_2 = random.randint(Ct, 25)
            tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
            rt_2 = Ct_2 * tem_downsample_ratio

            xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]
            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            lamb = []
            mapping = [
                        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
                        [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
                        [0, 1, 2], [0,1,3], [0,1,4], [0,2,3], [0,2,4],
                        [0,3,4], [1,2,3], [1,2,4], [1,3,4], [2,3,4],
                ]

            all_part = set(range(5))
            replace2remain = {}
            for replace_idx, item in enumerate(mapping):
                remain_idx = mapping.index(list(all_part.difference(set(item))))
                replace2remain[replace_idx] = remain_idx
            
            if adatemp == None:
                 temp = 0.2
            else:
                 temp = adatemp #N,1
            part_prob_dist = part_dist/part_dist.sum(dim=1,keepdim=True)
            part_prob_dist = F.softmax(part_dist/temp, dim=1)

            for i in range(N):
                if label_dist != None:
                    if label_dist[i] > label_dist[randidx[i]]:
                        dist = part_prob_dist[randidx[i]]
                        diffset = False
                    else:
                        dist = part_prob_dist[i]
                        diffset = True

                spa_idx = []
                #print(len(part_dist))
                parts_idx = random.choices(list(range(20)), weights=dist, k=1)[0]
                
                p_choice = parts_idx
                if diffset: 
                    p_choice = replace2remain[p_choice]
                parts_idx = mapping[p_choice]
                for part_idx in parts_idx:
                    spa_idx += body_parts_k400[int(part_idx)]
                spa_idx.sort()
                xst[i, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                        xst_temp[randidx[i]][:, :, spa_idx, :]
                lamb.append(rt * len(spa_idx) / (T * V))
            lambd = torch.tensor(lamb).reshape((N,1)).cuda()
            
        else:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros(T // tem_downsample_ratio, V) + lamb
            lambd = torch.full((N,1),lamb).cuda()

        return randidx, xst, lambd
