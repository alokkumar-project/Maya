import numpy as np
from ml_model.representation import *
from ml_model.nueral_network import *
class Transformer_Encoder:
    def __init__(self,embedding_dim=64,hidden_nueron=256,embedding_object=None,learning_rate=0.01,multi_head=2):
        
        
        self.embedding_dim = embedding_dim
        self.hidden_nueron = hidden_nueron
        
        self.self_attention = Multi_head_SelfAttention(embedding_dim=embedding_dim,learning_rate=learning_rate,multi_head=multi_head)
    
        self.layer_norm1 = LayerNorm(learning_rate=learning_rate)
        self.add1= Add()
        
        self.feed_forward = Feed_Forward(input_dimension=self.embedding_dim,hidden_nueron=self.hidden_nueron)
        self.layer_norm2 = LayerNorm(learning_rate=learning_rate)
        self.add2 = Add()
       
    def forward(self,x,dropout=0,training='n'):
        
        self_attention_output = self.self_attention.forward(x)

        add_layer1 = self.add1.forward(self_attention_output,x)
        layer_norm1_output = self.layer_norm1.forward(add_layer1)
        
        feed_forward_output = self.feed_forward.forward(layer_norm1_output,dropout=dropout,training=training)
        add_layer2_output = self.add2.forward(feed_forward_output,layer_norm1_output)
        layer_norm2_output = self.layer_norm2.forward(add_layer2_output)
       
        return layer_norm2_output

    def backprop(self,prev_gradient):
        layer_norm2_back_output = self.layer_norm2.backprop(prev_grad=prev_gradient)
        
        residual_grad2,add2_grad = self.add2.backprop(layer_norm2_back_output)
        feed_forward_grad = self.feed_forward.backprop(add2_grad)
        combine_gradient = feed_forward_grad+residual_grad2
        layer_norm1_grad = self.layer_norm1.backprop(combine_gradient)
       
        residual_grad1,add1_grad = self.add1.backprop(layer_norm1_grad)
        self_attention_grad = self.self_attention.backprop(add1_grad)
        final_grad = residual_grad1+self_attention_grad
        return final_grad
    def update(self,batch_size):
        self.layer_norm2.update(batch_size=batch_size)
        self.feed_forward.update(batch_size=batch_size)
        self.layer_norm1.update(batch_size=batch_size)
        self.self_attention.update(batch_size=batch_size)
        
        

