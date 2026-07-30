from ml_model.representation import *
from ml_model.nueral_network import *
class TransformerDecoder:
    def __init__(self,embedding_dim=64,embedding_object=None,hidden_nueron=None,learning_rate=0.01,multi_head=2):
        
        self.masked_attention = Multi_head_MaskAttention(embedding_dim=embedding_dim,learning_rate=learning_rate,multi_head=multi_head)
        self.cross_attention = Multi_head_CrossAttention(embedding_dim=embedding_dim,learning_rate=learning_rate,multi_head=multi_head)
        self.add1 = Add()
        self.layer_norm1 = LayerNorm(learning_rate=learning_rate)
        self.add2 = Add()
        self.layer_norm2 = LayerNorm(learning_rate=learning_rate)
        self.add3=Add()
        self.layer_norm3 = LayerNorm(learning_rate=learning_rate)
        self.feed_forward = Feed_Forward(input_dimension=embedding_dim,hidden_nueron=hidden_nueron,learning_rate=learning_rate)
    def forward(self,encoder_input=None,target_token=None,dropout=0,training='n'):
       
        target_token_position_output = target_token
        
        masked_attention_output = self.masked_attention.forward(embedding_sentence=target_token_position_output)
        add1_output = self.add1.forward(residual=target_token_position_output,x=masked_attention_output)
        layer1_output = self.layer_norm1.forward(add1_output)
       

        crossed_attention_output = self.cross_attention.forward(encoder_embedding_sentence=encoder_input,decoder_embedding_sentence=layer1_output)
        add2_output = self.add2.forward(x=crossed_attention_output,residual=layer1_output)
        layer2_output = self.layer_norm2.forward(add2_output)
       


        feed_forward_output = self.feed_forward.forward(x=layer2_output,dropout=dropout,training=training)
        add3_output = self.add3.forward(x=feed_forward_output,residual=layer2_output)
        layer3_output = self.layer_norm3.forward(add3_output)
     
        return layer3_output
    def backprop(self,prev_grad):
        layer3_norm_grad = self.layer_norm3.backprop(prev_grad=prev_grad)
       
        add3_grad,residual3_grad = self.add3.backprop(prev_gradient=layer3_norm_grad)
        feed_forward_grad = self.feed_forward.backprop(add3_grad)
        combine_gradinet3 = feed_forward_grad+residual3_grad

        layer2_nomr_grad = self.layer_norm2.backprop(prev_grad=combine_gradinet3)
     
        add2_grad,residual2_grad = self.add2.backprop(layer2_nomr_grad)
        encoder_grad,dcoder_grad = self.cross_attention.backprop(add2_grad)
        combine_gradinet2 = dcoder_grad+residual2_grad

        layer1_grad = self.layer_norm1.backprop(combine_gradinet2)
       
        add1_grad,residual1_grad = self.add1.backprop(layer1_grad)
        mask_grad = self.masked_attention.backprop(add1_grad)
        combine_gradinet1 = residual1_grad+mask_grad
     
        return combine_gradinet1,encoder_grad
    
    def update(self,batch_size):
        self.masked_attention.update(batch_size=batch_size)
        self.cross_attention.update(batch_size=batch_size)
        self.feed_forward.update(batch_size=batch_size)
        self.layer_norm1.update(batch_size=batch_size)
        self.layer_norm2.update(batch_size=batch_size)
        self.layer_norm3.update(batch_size=batch_size)

        
