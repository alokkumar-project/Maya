import numpy as np
import pickle
from ml_model.nueral_network import Adam
import time
from collections import Counter


class Embedding:
    def __init__(self,vocab_size,embedding_dim=64,learning_rate=0.01):
        self.learning_rate=learning_rate
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.backprop_grad ={}
        self.embedding_weights=np.random.randn(vocab_size,embedding_dim)*np.sqrt(1/self.embedding_dim)
    def forward(self,token_sentence):
        self.token_sentence = token_sentence
        return self.embedding_weights[token_sentence]
    def backprop(self,grad):
        for idx,i in enumerate(self.token_sentence):
            if i not in self.backprop_grad:
                self.backprop_grad[i]=grad[idx]
            else:
                self.backprop_grad[i]+=grad[idx]
    def update(self,batch_size):
        for i in self.backprop_grad.keys():
            self.embedding_weights[i]-=self.learning_rate*self.backprop_grad[i]/batch_size
        self.backprop_grad={}


class Positional_encoder:
    def __init__(self,embedding_dim=64):
        self.embedding_dim = embedding_dim
        
    def forward(self,embedding_token):
        pos_embedding=[]

        for pos_idx,current_token in enumerate(embedding_token):
            current_embedding = current_token.copy()
            for i in range(0,self.embedding_dim,2):
                sin_value = np.sin(pos_idx/(10000)**(i/self.embedding_dim))
                cos_value = np.cos(pos_idx/(10000)**(i/self.embedding_dim))
                current_embedding[i]=sin_value+current_embedding[i]
                current_embedding[i+1]=cos_value+current_embedding[i+1]
            pos_embedding.append(current_embedding)
        
        return np.array(pos_embedding)

            
              
class Multi_head_SelfAttention:
    def __init__(self,embedding_dim=64,learning_rate=0.01,multi_head=4):

        self.embedding_dim = embedding_dim
        self.head_dim=embedding_dim//multi_head
        self.learning_rate=learning_rate
        self.head = multi_head

        self.query_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.key_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.value_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)

        self.final_matrix = np.random.randn(embedding_dim,embedding_dim)*np.sqrt(1.0/embedding_dim)

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final = 0
        
        self.adam_query = Adam(learning_rate=learning_rate)
        self.adam_key = Adam(learning_rate=learning_rate)
        self.adam_value = Adam(learning_rate=learning_rate)
        self.adam_final = Adam(learning_rate=learning_rate)


    def forward(self,embedding_sentence):
        self.embedding_sentence = embedding_sentence

        self.query_matrix = np.matmul(embedding_sentence[None],self.query_weights)
        self.key_matrix = np.matmul(embedding_sentence[None],self.key_weights)
        self.value_matrix = np.matmul(embedding_sentence[None],self.value_weights)

        self.score_matrix = np.matmul(self.query_matrix,self.key_matrix.transpose(0,2,1))/np.sqrt(self.head_dim)
      
        self.attention = self.softmax(self.score_matrix)

        self.pre_output = np.matmul(self.attention,self.value_matrix).transpose(1,0,2).reshape(embedding_sentence.shape[0],-1) # sare head ko concadinate karne ke liye 

        self.output = np.dot(self.pre_output,self.final_matrix)

        return self.output
        

    def softmax(self,z):
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def backprop(self,prev_grad):
        dfinal_matrix = np.dot(self.pre_output.T,prev_grad)
        dpre_out_after_transpose_reshape = np.dot(prev_grad,self.final_matrix.T)
        dpre_output_grad =dpre_out_after_transpose_reshape.reshape(self.embedding_sentence.shape[0],self.head,self.head_dim).transpose(1,0,2)
        dattention = np.matmul(dpre_output_grad,self.value_matrix.transpose(0,2,1))
        dvalue = np.matmul(self.attention.transpose(0,2,1),dpre_output_grad)
        dscore = (self.softmax_der(dattention))/np.sqrt(self.head_dim)

        dquery_matrix = np.matmul(dscore,self.key_matrix)
        dkey_matrix = np.matmul(dscore.transpose(0,2,1),self.query_matrix)

        # print('dattention_shape ',dattention.shape)
        # print('dscore ',dscore.shape)
        # print('dkey_matrix',dkey_matrix.shape)

        dkey_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dkey_matrix)
        dquery_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dquery_matrix)
        dvalue_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dvalue)

        # print('dkey_weights',dkey_weights.shape)
        dembedding = np.matmul(dquery_matrix,self.query_weights.transpose(0,2,1))+np.matmul(dkey_matrix,self.key_weights.transpose(0,2,1))+np.matmul(dvalue,self.value_weights.transpose(0,2,1))
        final_dembedding = np.sum(dembedding,axis=0)

        self.avg_dkey+=dkey_weights
        self.avg_dvalue+=dvalue_weights
        self.avg_dquery+=dquery_weights
        self.avg_final+=dfinal_matrix

        return final_dembedding
    def update(self,batch_size=1):

        self.avg_dquery = self.adam_query.update(self.avg_dquery)
        self.avg_dkey = self.adam_key.update(self.avg_dkey)
        self.avg_dvalue = self.adam_value.update(self.avg_dvalue)
        self.avg_final=self.adam_final.update(self.avg_final)

        self.query_weights = self.query_weights-self.avg_dquery/batch_size
        self.key_weights = self.key_weights-self.avg_dkey/batch_size
        self.value_weights = self.value_weights-self.avg_dvalue/batch_size
        self.final_matrix = self.final_matrix-self.avg_final/batch_size

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final=0
        



    def old_softmax_der(self,dattention):
        final =[]
        for current_head in range(dattention.shape[0]):
            current_row_grad=[]
            for current_row in range(dattention.shape[1]):
                jocobin = np.diag(self.attention[current_head][current_row])-np.outer(self.attention[current_head][current_row],self.attention[current_head][current_row])
                grad_attention = dattention[current_head][current_row]
           
                dscore = np.dot(jocobin,grad_attention)
                current_row_grad.append(dscore)
            final.append(current_row_grad)
        return np.array(final)
    
    def softmax_der(self,dattention):
        s = self.attention
        return s * (dattention - np.sum(dattention * s, axis=-1, keepdims=True))


class Multi_head_MaskAttention:
    def __init__(self,embedding_dim=64,learning_rate=0.01,multi_head=4):

        self.embedding_dim = embedding_dim
        self.head_dim=embedding_dim//multi_head
        self.learning_rate=learning_rate
        self.head = multi_head

        self.query_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.key_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.value_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)

        self.final_matrix = np.random.randn(embedding_dim,embedding_dim)*np.sqrt(1.0/embedding_dim)

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final = 0
        
        self.adam_query = Adam(learning_rate=learning_rate)
        self.adam_key = Adam(learning_rate=learning_rate)
        self.adam_value = Adam(learning_rate=learning_rate)
        self.adam_final = Adam(learning_rate=learning_rate)


    def forward(self,embedding_sentence):
        self.embedding_sentence = embedding_sentence

        self.query_matrix = np.matmul(embedding_sentence[None],self.query_weights)
        self.key_matrix = np.matmul(embedding_sentence[None],self.key_weights)
        self.value_matrix = np.matmul(embedding_sentence[None],self.value_weights)

        self.score_matrix = np.matmul(self.query_matrix,self.key_matrix.transpose(0,2,1))/np.sqrt(self.head_dim)

        masked_matrix = np.triu(np.ones_like(self.score_matrix),k=1)
        masked_matrix = np.where(masked_matrix==1,-np.inf,0)
        self.new_score_matrix = self.score_matrix+masked_matrix

 
        self.attention = self.softmax(self.new_score_matrix)

        self.pre_output = np.matmul(self.attention,self.value_matrix).transpose(1,0,2).reshape(embedding_sentence.shape[0],-1) # sare head ko concadinate karne ke liye 

        self.output = np.dot(self.pre_output,self.final_matrix)

        # print('output_shape', self.output.shape)
        # print('pre_output ',self.pre_output.shape)
        # print('value_matrix ',self.value_matrix.shape)
        

        return self.output
        

    def softmax(self,z):
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def backprop(self,prev_grad):
        dfinal_matrix = np.dot(self.pre_output.T,prev_grad)
        dpre_out_after_transpose_reshape = np.dot(prev_grad,self.final_matrix.T)
        dpre_output_grad =dpre_out_after_transpose_reshape.reshape(self.embedding_sentence.shape[0],self.head,self.head_dim).transpose(1,0,2)
        dattention = np.matmul(dpre_output_grad,self.value_matrix.transpose(0,2,1))
        dvalue = np.matmul(self.attention.transpose(0,2,1),dpre_output_grad)
        dscore = (self.softmax_der(dattention))/np.sqrt(self.head_dim)

        dquery_matrix = np.matmul(dscore,self.key_matrix)
        dkey_matrix = np.matmul(dscore.transpose(0,2,1),self.query_matrix)


        dkey_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dkey_matrix)
        dquery_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dquery_matrix)
        dvalue_weights = np.matmul(self.embedding_sentence[None].transpose(0,2,1),dvalue)

       
        dembedding = np.matmul(dquery_matrix,self.query_weights.transpose(0,2,1))+np.matmul(dkey_matrix,self.key_weights.transpose(0,2,1))+np.matmul(dvalue,self.value_weights.transpose(0,2,1))
        final_dembedding = np.sum(dembedding,axis=0)

        self.avg_dkey+=dkey_weights
        self.avg_dvalue+=dvalue_weights
        self.avg_dquery+=dquery_weights
        self.avg_final+=dfinal_matrix

        return final_dembedding
    def update(self,batch_size=1):

        self.avg_dquery = self.adam_query.update(self.avg_dquery)
        self.avg_dkey = self.adam_key.update(self.avg_dkey)
        self.avg_dvalue = self.adam_value.update(self.avg_dvalue)
        self.avg_final=self.adam_final.update(self.avg_final)

        self.query_weights = self.query_weights-self.avg_dquery/batch_size
        self.key_weights = self.key_weights-self.avg_dkey/batch_size
        self.value_weights = self.value_weights-self.avg_dvalue/batch_size
        self.final_matrix = self.final_matrix-self.avg_final/batch_size

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final=0
        



    def old_softmax_der(self,dattention):
        final =[]
        for current_head in range(dattention.shape[0]):
            current_row_grad=[]
            for current_row in range(dattention.shape[1]):
                jocobin = np.diag(self.attention[current_head][current_row])-np.outer(self.attention[current_head][current_row],self.attention[current_head][current_row])
                grad_attention = dattention[current_head][current_row]
                dscore = np.dot(jocobin,grad_attention)
                current_row_grad.append(dscore)
            final.append(current_row_grad)
        return np.array(final)
    
    def softmax_der(self,dattention):
        s = self.attention
        return s * (dattention - np.sum(dattention * s, axis=-1, keepdims=True))


    
class Multi_head_CrossAttention:
    def __init__(self,embedding_dim=64,learning_rate=0.01,multi_head=4):

        self.embedding_dim = embedding_dim
        self.head_dim=embedding_dim//multi_head
        self.learning_rate=learning_rate
        self.head = multi_head

        self.query_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.key_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)
        self.value_weights = np.random.randn(multi_head,embedding_dim,self.head_dim)*np.sqrt(1.0/embedding_dim)

        self.final_matrix = np.random.randn(embedding_dim,embedding_dim)*np.sqrt(1.0/embedding_dim)

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final = 0
        
        self.adam_query = Adam(learning_rate=learning_rate)
        self.adam_key = Adam(learning_rate=learning_rate)
        self.adam_value = Adam(learning_rate=learning_rate)
        self.adam_final = Adam(learning_rate=learning_rate)


    def forward(self,decoder_embedding_sentence,encoder_embedding_sentence):
        self.encoder_sentence = encoder_embedding_sentence
        self.decoder_sentence = decoder_embedding_sentence

        self.query_matrix = np.matmul(self.decoder_sentence[None],self.query_weights)
        self.key_matrix = np.matmul(self.encoder_sentence[None],self.key_weights)
        self.value_matrix = np.matmul(self.encoder_sentence[None],self.value_weights)

        self.score_matrix = np.matmul(self.query_matrix,self.key_matrix.transpose(0,2,1))/np.sqrt(self.head_dim)
        
        self.attention = self.softmax(self.score_matrix)

        self.pre_output = np.matmul(self.attention,self.value_matrix).transpose(1,0,2).reshape(self.decoder_sentence.shape[0],-1) # sare head ko concadinate karne ke liye 

        self.output = np.dot(self.pre_output,self.final_matrix)


        return self.output
        

    def softmax(self,z):
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def backprop(self,prev_grad):
        dfinal_matrix = np.dot(self.pre_output.T,prev_grad)

        dpre_out_after_transpose_reshape = np.dot(prev_grad,self.final_matrix.T)

        dpre_output_grad =dpre_out_after_transpose_reshape.reshape(self.decoder_sentence.shape[0],self.head,self.head_dim).transpose(1,0,2)

        dattention = np.matmul(dpre_output_grad,self.value_matrix.transpose(0,2,1))
        dvalue = np.matmul(self.attention.transpose(0,2,1),dpre_output_grad)
        dscore = (self.softmax_der(dattention))/np.sqrt(self.head_dim)

        dquery_matrix = np.matmul(dscore,self.key_matrix)
        dkey_matrix = np.matmul(dscore.transpose(0,2,1),self.query_matrix)


        dkey_weights = np.matmul(self.encoder_sentence[None].transpose(0,2,1),dkey_matrix)
        dquery_weights = np.matmul(self.decoder_sentence[None].transpose(0,2,1),dquery_matrix)
        dvalue_weights = np.matmul(self.encoder_sentence[None].transpose(0,2,1),dvalue)

        # print('dkey_weights',dkey_weights.shape)
        encoder_dembedding = np.matmul(dkey_matrix,self.key_weights.transpose(0,2,1))+np.matmul(dvalue,self.value_weights.transpose(0,2,1))
        decoder_dembedding = np.matmul(dquery_matrix,self.query_weights.transpose(0,2,1))

        decoder_final_dembedding = np.sum(decoder_dembedding,axis=0)
        encoder_final_dembedding = np.sum(encoder_dembedding,axis=0)

        self.avg_dkey+=dkey_weights
        self.avg_dvalue+=dvalue_weights
        self.avg_dquery+=dquery_weights
        self.avg_final+=dfinal_matrix

        return encoder_final_dembedding,decoder_final_dembedding
    def update(self,batch_size=1):

        self.avg_dquery = self.adam_query.update(self.avg_dquery)
        self.avg_dkey = self.adam_key.update(self.avg_dkey)
        self.avg_dvalue = self.adam_value.update(self.avg_dvalue)
        self.avg_final=self.adam_final.update(self.avg_final)

        self.query_weights = self.query_weights-self.avg_dquery/batch_size
        self.key_weights = self.key_weights-self.avg_dkey/batch_size
        self.value_weights = self.value_weights-self.avg_dvalue/batch_size
        self.final_matrix = self.final_matrix-self.avg_final/batch_size

        self.avg_dquery=0
        self.avg_dkey=0
        self.avg_dvalue=0
        self.avg_final=0
        
    def softmax_der(self,dattention):
        s = self.attention
        return s * (dattention - np.sum(dattention * s, axis=-1, keepdims=True))



class BPE:
    def __init__(self,vocab_size=None):
        letters= " abcdefghijklmnopqrstuvwxyz!'.,?:()😭😎🔥❤️😂😘😍🥹🥳😹🥰🎉👏👍🤣😡😠🤬😤🥲😌☠️💖💕😩🤡$*^-_"
        self.vocab={}
        self.vocab["<PAD>"]=0
        self.vocab["<UNK>"]=1
        self.vocab["<SOS>"]=2
        self.vocab["<EOS>"]=3
        self.vocab["maya"]=4
        self.vocab["alok"]=5
        for idx,letter in enumerate(letters):
            self.vocab[letter]=idx+6
        self.word_frequency ={}
        self.pair_frequency={}
        self.reverse_vocab={}
        self.merges = []
        self.vocab_size = vocab_size
        self.token =[]
    def build_vocab(self,sentence):
        self.token.append(list(sentence))
        
    def train(self,dataset=None):
        self.token=[]
        file = open(dataset,'r')
        for sentence in file:
            sentence = sentence.lower().strip()
            self.token.append(list(sentence))
            
            
        self.count_pair()
        
        
        while(len(self.vocab)<self.vocab_size and len(self.pair_frequency)>0):
            best = self.best_pair()
            merge = self.merge_pair(best=best)
            self.merges.append(best)
            if merge not in self.vocab:
                pos = len(self.vocab)
                self.vocab[merge]=pos
            
            self.count_pair()
            
        for key,value in self.vocab.items():
            self.reverse_vocab[value]=key


    def count_pair(self):
        self.pair_frequency = Counter()
    
        for token in self.token:
            self.pair_frequency.update(zip(token, token[1:]))
                

    def merge_pair(self,best):
        new_token=[]
        
        for current_word_token in (self.token):
            word_token=[]
            i=0
            while(i<len(current_word_token)-1):
                pair = (current_word_token[i],current_word_token[i+1])
                if(pair==best):
                    modi_pair =''.join(pair)
                    word_token.append(modi_pair)
                    i+=2
                else:
                    word_token.append(current_word_token[i])
                    i+=1
            if(i==len(current_word_token)-1):
                word_token.append(current_word_token[i])
            new_token.append(word_token)
        self.token=new_token
        return ''.join(best)

            

    def new_encode(self, sentence):

        sentence = sentence.strip()

        if not sentence:
            return []

        special_tokens = {
            "<SOS>", "<EOS>", "<PAD>", "<UNK>",
            "maya", "alok"
        }

        pieces = []

        words = sentence.split()

        for idx, word in enumerate(words):

          
            if word in special_tokens:
                pieces.append(word)

            else:
                word = word.lower()

                for ch in word:
                    pieces.append(ch)

            # Preserve spaces
            if idx != len(words) - 1:
                pieces.append(" ")


        tokens = pieces

        for left, right in self.merges:

            merged = left + right
            new_tokens = []

            i = 0

            while i < len(tokens):

                
                if tokens[i] in special_tokens:
                    new_tokens.append(tokens[i])
                    i += 1
                    continue

                if (
                    i + 1 < len(tokens)
                    and tokens[i + 1] not in special_tokens
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        unk = self.vocab["<UNK>"]

        return [self.vocab.get(token, unk) for token in tokens]

    def retrain(self, special_words, file_name=None,after_training_file_name=None):

    
        if not hasattr(self, "special_words"):
            self.special_words = set()

        for word in special_words:

            word = word.lower().strip()

            self.special_words.add(word)

            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word

        self.retrain_save_model(file_name=after_training_file_name)
        print("Special tokens added successfully!")
        

    
    def decode(self,encoded_sentence):
        decoded_sentence =[]
        for current_word in encoded_sentence:
            if current_word in self.reverse_vocab:
                pos = self.reverse_vocab[current_word]
                decoded_sentence.append(pos)
        return "".join(decoded_sentence)


    def new_decode(self, encoded_sentence):
        reverse = self.reverse_vocab
        return "".join(reverse.get(i, "") for i in encoded_sentence)

    def old_best_pair(self):
        keys=[]
        values=[]
        for key,value in self.pair_frequency.items():
            keys.append(key)
            values.append(value)
        values = np.array(values)
        max_value = np.argmax(values)
        best = keys[max_value]
        
       
        return best
        


    def best_pair(self):
        if not self.pair_frequency:
            return None
        return self.pair_frequency.most_common(1)[0][0]
    
    def pred_merge_pair(self,best,user_token):
        new_token=[]
        
        for current_word_token in (user_token):
            word_token=[]
            i=0
            while(i<len(current_word_token)-1):
                pair = (current_word_token[i],current_word_token[i+1])
                if(pair==best):
                    modi_pair =''.join(pair)
                    word_token.append(modi_pair)
                    i+=2
                else:
                    word_token.append(current_word_token[i])
                    i+=1
            if(i==len(current_word_token)-1):
                word_token.append(current_word_token[i])
            new_token.append(word_token)
        user_token=new_token
        return user_token

        
    def save_model(self,file_name='bpe_token.pkl'):
        model = {
            'vocab':self.vocab,
            'merge':self.merges,
            'reverse_vocab':self.reverse_vocab,
            
        }
        with open(file_name,'wb') as file:
            pickle.dump(model,file)
        print("model saved!")

    def retrain_save_model(self,file_name='bpe_token.pkl'):
        model = {
            'vocab':self.vocab,
            'merge':self.merges,
            'reverse_vocab':self.reverse_vocab,
            'special_words': list(self.special_words)
        }
        with open(file_name,'wb') as file:
            pickle.dump(model,file)
        print("model saved!")
    def load_model(self, file_name='bpe_token.pkl'):
        with open(file_name, 'rb') as file:
            model = pickle.load(file)

        self.vocab = model['vocab']
        self.merges = model['merge']
        self.reverse_vocab = model['reverse_vocab']
        print('model loaded successfully!')

                
        
            