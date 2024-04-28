from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint
from models.vit import VisionTransformer


class BLIP_BEV_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 bev_size=50, 
                 bev_dim=256,
                 visual_width=768,                   
                 ):
        
        super().__init__()
        self.device = "cpu"
        self.tokenizer = init_tokenizer() 
        
        # BEV --------------------------------------------------------------------------------------
        self.bev_size = bev_size
        self.bev_dim = bev_dim

        # Visual Encoder ---------------------------------------------------------------------------
        self.visual_width = visual_width
        self.vit_patch_size = 5
        self.vit_depth = 3
        self.vit_num_heads = 12

        self.vis_encoder = VisionTransformer(img_size=self.bev_size,
                                             patch_size=self.vit_patch_size,
                                             in_chans=self.bev_dim,
                                             embed_dim=self.visual_width,
                                             depth=self.vit_depth,
                                             num_heads=self.vit_num_heads,
                                             use_grad_checkpointing=False,
                                             ckpt_layer=0,
                                             drop_path_rate=0)
        
        # Text Encoder -----------------------------------------------------------------------------
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = self.visual_width
        self.text_encoder = BertModel.from_pretrained(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 
        self.text_width = self.text_encoder.config.hidden_size

        # Text Decoder ----------------------------------------------------------------------------
        decoder_config = BertConfig.from_json_file(med_config) 
        decoder_config.encoder_width = self.visual_width
        self.text_decoder = BertLMHeadModel.from_pretrained(config=decoder_config)    
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))          


    def forward(self, bev, question, answer=None, n=None):
        
        bev_embeds = self.vis_encoder(bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)) 
        bev_atts = torch.ones(bev_embeds.size()[:-1],dtype=torch.long).to(self.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(self.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id

                       
        '''
        n: number of answers for each question
        weights: weight for each answer
        '''                     
        answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(self.device) 
        answer.input_ids[:,0] = self.tokenizer.bos_token_id
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

        question_output = self.text_encoder(question.input_ids, 
                                            attention_mask = question.attention_mask, 
                                            encoder_hidden_states = bev_embeds,
                                            encoder_attention_mask = bev_atts,                             
                                            return_dict = True)    

        question_states = []                
        question_atts = []  
        for b, n in enumerate(n):
            question_states += [question_output.last_hidden_state[b]]*n
            question_atts += [question.attention_mask[b]]*n                
        question_states = torch.stack(question_states,0)    
        question_atts = torch.stack(question_atts,0)     

        answer_output = self.text_decoder(answer.input_ids, 
                                            attention_mask = answer.attention_mask, 
                                            encoder_hidden_states = question_states,
                                            encoder_attention_mask = question_atts,                  
                                            labels = answer_targets,
                                            return_dict = True,   
                                            reduction = 'none',
                                            )      
        
        loss = answer_output.loss
        return loss
 
    def generate(
        self,
        bev,
        question,
        max_length=50,
        min_length=10,
        top_p=0.9,
    ):
        
        bev_embeds = self.vis_encoder(bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)) 
        bev_atts = torch.ones(bev_embeds.size()[:-1],dtype=torch.long).to(self.device)
        bs = bev_embeds.size(0)

        question_output = self.text_encoder(question.input_ids, 
                                            attention_mask = question.attention_mask, 
                                            encoder_hidden_states = bev_embeds,
                                            encoder_attention_mask = bev_atts,                                    
                                            return_dict = True) 
            
        question_states = question_output.last_hidden_state
        question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
        model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
        
        bos_ids = torch.full((bs, 1),fill_value=self.tokenizer.bos_token_id,device=self.device)
        
        # nucleus sampling
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
            **model_kwargs
        )
        
        answers = []    
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)    
            answers.append(answer)
        return answers
        

if __name__ == "__main__":
    model = BLIP_BEV_VQA()
    bev = torch.rand(1, 2500, 256, dtype=torch.float)
    question = "How are you?"
    ans = model.generate(bev, question)
    print(ans)

