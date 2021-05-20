import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
class Softmax_decoder(nn.Module):

    def __init__(self,d_model,tag_num):
        super().__init__()
        self.tag_num = tag_num
        self.d_model =d_model
        self.feature2tag = nn.Linear(d_model,tag_num)
    
    def get_path(self,tag,mask):
        "tag （batch_size,seq_len)"
        reverse_mask = (mask == False)
        flatten_tag = torch.masked_select(tag,reverse_mask) # 是一维的哦
        all_path = []
        path = []
        sentence_length = torch.sum(reverse_mask,dim=-1).long()
        offset = 0
        for sent_len in sentence_length:
            path = flatten_tag[offset:offset+sent_len].detach().cpu().numpy().tolist()
            offset += sent_len
            all_path.append(path)
        return all_path


    def loss(self,feature_vec,trg,src_mask,use_gpu):
        tag_vec = self.feature2tag(feature_vec)
        tag_vec_ = tag_vec.view(-1,self.tag_num)
        trg = trg.view(-1)
        loss = F.cross_entropy(tag_vec_,trg,reduction='none',ignore_index=self.tag_num)
        tag_score = F.softmax(tag_vec,dim=-1)
        score, tag = torch.max(tag_score,dim=-1)
        # 按照mask，将 tag组织为二重列表
        all_path = self.get_path(tag, src_mask)
        return loss,all_path
    

    def forward(self,feature_vec,mask,use_gpu):
        with torch.no_grad():
            tag_vec = self.feature2tag(feature_vec)
            tag_score = F.softmax(tag_vec,dim=-1)
            score,tag = torch.max(tag_score,dim=-1)
            all_path = self.get_path(tag, mask)
            return all_path






