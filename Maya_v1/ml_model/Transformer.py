from ml_model.encoder import *
from ml_model.decoder import *
from ml_model.representation import *
from ml_model.nueral_network import *
import time
import pickle

class Transformer:
    def __init__(self,encoder_unit=2,decoder_unit=2,source_file=None,target_file=None,token_file=None,embedding_dim=64,epoch=10,hidden_nueron=256,learning_rate=0.01,batch_size=32,vocab_size=4000,no_of_sentence=100,training_mode='n',multi_head_encoder=2,multi_head_decoder=2,dropout=0):
        self.epoch = epoch
        self.dropout = dropout
        self.batch_size = batch_size
        self.encoder_unit = encoder_unit
        self.decoder_unit = decoder_unit
        self.t = BPE(vocab_size=vocab_size)
        self.t.load_model(file_name=token_file)
        self.total_word = len(self.t.vocab)
        self.embedding_dim = embedding_dim
        self.dense = Dense(self.total_word,learning_rate=learning_rate)
        
        self.e_encoder = Embedding(vocab_size=self.total_word,embedding_dim=embedding_dim)
        self.e_decoder = Embedding(vocab_size=self.total_word,embedding_dim=embedding_dim)
        
        self.positional_encoder = Positional_encoder(embedding_dim=embedding_dim)
        
        self.softmax = Softmax()

        if(training_mode=='y'):
            self.token_encoder_sentence = []
            self.token_decoder_sentence = []
            
            target_file = open(target_file,'rb')
            self.token_decoder_sentence = pickle.load(target_file)
            target_file.close()
            
            source_file = open(source_file,'rb')
            self.token_encoder_sentence = pickle.load(source_file)
            source_file.close()
            
            self.token_encoder_sentence = self.token_encoder_sentence[:no_of_sentence]
            self.token_decoder_sentence = self.token_decoder_sentence[:no_of_sentence]
        
        self.transformer_encoder = {}
        self.transformer_dcoder={}
        
        for i in range(encoder_unit):
            self.transformer_encoder[i]=Transformer_Encoder(embedding_dim=embedding_dim,hidden_nueron=hidden_nueron,embedding_object=None,learning_rate=learning_rate,multi_head=multi_head_encoder)
        
        for i in range(decoder_unit):
            self.transformer_dcoder[i]=TransformerDecoder(embedding_object=None,embedding_dim=embedding_dim,hidden_nueron=hidden_nueron,learning_rate=learning_rate,multi_head=multi_head_decoder)
        


    def encoder_processing(self,input_sentence,training='n'):
        input_text = input_sentence
        for i in range(self.encoder_unit):
            process_sentence = self.transformer_encoder[i].forward(x=input_text,dropout=self.dropout,training=training)
            input_text = process_sentence
        return process_sentence
    def decoder_processing(self,encoder_input,target_token,training='n'):
        input_token = target_token
        for i in range(self.decoder_unit):
            process_information = self.transformer_dcoder[i].forward(encoder_input=encoder_input,target_token=input_token,dropout=self.dropout,training=training)
            input_token = process_information
        return process_information

    def train(self):
        total_sentence = len(self.token_encoder_sentence)
        intial_time = time.time()
        print("First 5 source sentences:")

        for i in range(5):

            print(self.t.decode(self.token_encoder_sentence[i]))

        print("\nFirst 5 target sentences:")

        for i in range(5):

            print(self.t.decode(self.token_decoder_sentence[i]))

        for i in (range(self.epoch)):
            epcoh_loss=0
            token_encoder_dataset = []
            token_decoder_dataset=[]
            random_number_series = np.random.permutation(len(self.token_encoder_sentence))
            for idx in random_number_series:
                token_encoder_dataset.append(self.token_encoder_sentence[idx])
                token_decoder_dataset.append(self.token_decoder_sentence[idx])
                
            start_time = time.time()
            for current_batch_epoch in (range(0,len(self.token_decoder_sentence),self.batch_size)):
                
                batch_token_encoder = token_encoder_dataset[current_batch_epoch:current_batch_epoch+self.batch_size]
                batch_token_decoder = token_decoder_dataset[current_batch_epoch:current_batch_epoch+self.batch_size]

                batch_size = len(batch_token_encoder)
                
                
                for current_encoder_sentence,current_decoder_sentence in zip(batch_token_encoder,batch_token_decoder):
                    # print(current_decoder_sentence)
                    # print(len(current_decoder_sentence))
                    embedding_encoder_output = self.e_encoder.forward(current_encoder_sentence)
                    embedding_decoder_output = self.e_decoder.forward(current_decoder_sentence[:-1])
                    
                    positional_output = self.positional_encoder.forward(embedding_encoder_output)
                    positional_decoder = self.positional_encoder.forward(embedding_decoder_output)
                    
                    encoder_output = self.encoder_processing(input_sentence=positional_output,training='y')
                    
                    decoder_output = self.decoder_processing(encoder_input=encoder_output,target_token=positional_decoder,training='y')
    
                    loss,final_grad = self.train_output_layer(decoder_prediction=decoder_output,real_output=current_decoder_sentence[1:])
                    epcoh_loss = epcoh_loss+loss
                    self.backprop(final_grad)
                    
                    

                self.update(batch_size=batch_size)
            end_time = time.time()
            
            print(f'Loss is {i} iteration is {epcoh_loss/total_sentence} | Time for this epoch is {end_time-start_time}')
        final_time = time.time()
        print("Total time :",final_time-intial_time)
    def gready_generate(self,sentence):
    
        encode_sentence = self.t.new_encode(sentence)
        start_token = self.t.new_encode('<SOS>')[0]
        end_token = self.t.new_encode('<EOS>')[0]
        self.output=[start_token]
        
        embedded_sentence = self.e_encoder.forward(encode_sentence)
        encode_sentence = self.positional_encoder.forward(embedded_sentence)# I was here!!
        
        encoder_output = self.encoder_processing(input_sentence=encode_sentence)

        decoder_inital_target = self.positional_encoder.forward(self.e_decoder.forward(np.array([start_token])))
        
                                         
        decoder_output = self.decoder_processing(encoder_input=encoder_output,target_token=decoder_inital_target)
        final_layer_output = self.generator_output(decoder_output)
        self.output.append(final_layer_output)
        
        while(final_layer_output!=end_token and len(self.output)<50):
            
            decoder_after_target = self.positional_encoder.forward(self.e_decoder.forward(np.array(self.output)))

            
            decoder_output = self.decoder_processing(encoder_input=encoder_output,target_token=decoder_after_target)

            final_layer_output = self.generator_output(decoder_output)
            self.output.append(final_layer_output)
            
        if(len(self.output)<50):
            final_result = self.t.decode(self.output[1:-1])
        else:
            final_result = self.t.decode(self.output)

        return final_result
        
            
    def generator_output(self,decorder_prediction):
        
        dense_output = self.dense.forward(decorder_prediction[-1].reshape(1,-1))
        softmax_output = self.softmax.forward(dense_output)
        output_token = np.argmax(softmax_output)

    
        return output_token

    def generate(self, sentence):
        # This is a beam Generator
       
        encode_sentence = self.t.new_encode(sentence)
        start_token = self.t.new_encode('<SOS>')[0]
        end_token = self.t.new_encode('<EOS>')[0]
        self.output = [start_token]

        embedded_sentence = self.e_encoder.forward(encode_sentence)
        encode_sentence = self.positional_encoder.forward(embedded_sentence)  

        encoder_output = self.encoder_processing(encode_sentence)

        decoder_inital_target = self.positional_encoder.forward(self.e_decoder.forward(np.array([start_token])))

        decoder_output = self.decoder_processing(encoder_input=encoder_output, target_token=decoder_inital_target)

        beam_candidate = []

        final_layer_output, first_prob = self.beam_generator_output(decoder_output)

        for token, prob in zip(final_layer_output, first_prob):
            beam = {
                'sequence': self.output + [token],
                'prob': -np.log(prob)
            }
            beam_candidate.append(beam)
        while True:
            finished = True

            all_child = []

            for i in beam_candidate:

                candidate_child = []
                if i['sequence'][-1] == end_token or len(i['sequence']) > 50:
                    all_child.append(i)
                    continue
                finished = False
                position_output = self.positional_encoder.forward(self.e_decoder.forward(i['sequence']))
                decoder_output = self.decoder_processing(target_token=position_output, encoder_input=encoder_output)
                output, prob = self.beam_generator_output(decorder_prediction=decoder_output)
                for current_token, current_prob in zip(output, prob):
                    beam = {
                        'sequence': i['sequence'] + [current_token],
                        'prob': i['prob'] - np.log(current_prob)
                    }
                    candidate_child.append(beam)
                all_child.extend(candidate_child)

            final_prob = []
            for current_beam in all_child:
                final_prob.append(current_beam['prob'])
            sorted_pro = np.argsort(np.array(final_prob))[:3]
            survived_candidate = []
            for index_prob in sorted_pro:
                survived_candidate.append(all_child[index_prob])
            beam_candidate = survived_candidate

            if finished:
                break
        final_beam_prob = []

        for current_sequence in beam_candidate:
            final_beam_prob.append(current_sequence['prob'] / (len(current_sequence['sequence']) ** (0.6)))
        location = np.argsort(np.array(final_beam_prob))[0]
        final = beam_candidate[location]['sequence'][1:-1]
        beam_decode = self.t.decode(final)
        beam_decode = beam_decode.strip()
        
        return beam_decode.capitalize()

    def beam_generator_output(self, decorder_prediction):
        dense_output = self.dense.forward(decorder_prediction[-1].reshape(1, -1))
        softmax_output = self.softmax.forward(dense_output)[0]
        output_token = np.argsort(softmax_output)[-3:][::-1]
        prob = softmax_output[output_token]
        
        return output_token, prob


    def decoder_backprop_processing(self,prev_grad):
        encoder_grad=[]
        input_value = prev_grad
        for i in reversed(range(self.decoder_unit)):
            decorder_grad,decoder_encoder_grad = self.transformer_dcoder[i].backprop(input_value)
            encoder_grad.append(decoder_encoder_grad)
            input_value = decorder_grad
        encoder_grad = np.stack(encoder_grad, axis=0)
        encoder_grad = np.sum(encoder_grad, axis=0)
        return decorder_grad,encoder_grad
            
    def encoder_back_processing(self,decoder_coming_grad):
        input_value = decoder_coming_grad
        for i in reversed(range(self.encoder_unit)):
            grad = self.transformer_encoder[i].backprop(input_value)
            input_value = grad
        return grad

    def backprop(self,final_grad):
        dense_grad = self.dense.backprop(final_grad)
        decorder_grad,decoder_encoder_grad = self.decoder_backprop_processing(dense_grad)
        self.e_decoder.backprop(decorder_grad)
        encoder_grad = self.encoder_back_processing(decoder_encoder_grad)
        self.e_encoder.backprop(encoder_grad)

    def expand_vocab(self,no_of_new_token):
        print('earlier vocab: ',len(self.t.vocab))
        print('earliear embedding ',self.e_encoder.embedding_weights.shape)

        old_encoder_embedding = self.e_encoder.embedding_weights
        dim_shape = self.e_encoder.embedding_weights.shape[1]
        new_embedding = np.random.randn(no_of_new_token,dim_shape)*np.sqrt(1/dim_shape)
        self.e_encoder.embedding_weights = np.vstack([old_encoder_embedding, new_embedding])

        old_decoder_embedding = self.e_decoder.embedding_weights
        new_decoder_embedding = np.random.randn(no_of_new_token,dim_shape)*np.sqrt(1/dim_shape)
        self.e_decoder.embedding_weights = np.vstack([old_decoder_embedding, new_decoder_embedding])

        old_dense = self.dense.weights
   

        new_dense = np.random.randn(old_dense.shape[0],no_of_new_token)*np.sqrt(2/old_dense.shape[0])
        self.dense.weights = np.hstack([old_dense, new_dense])

        new_bias = np.zeros((1, no_of_new_token))
        self.dense.bais = np.concatenate([self.dense.bais, new_bias],axis=1)
        
        self.dense.nueron += no_of_new_token

        if not np.isscalar(self.dense.adam_weights.m):

            self.dense.adam_weights.m = np.hstack([
                self.dense.adam_weights.m,
                np.zeros((self.dense.adam_weights.m.shape[0], no_of_new_token))
            ])

            self.dense.adam_weights.v = np.hstack([
                self.dense.adam_weights.v,
                np.zeros((self.dense.adam_weights.v.shape[0], no_of_new_token))
            ])

            self.dense.adam_weights.m_hat = np.hstack([
                self.dense.adam_weights.m_hat,
                np.zeros((self.dense.adam_weights.m_hat.shape[0], no_of_new_token))
            ])

            self.dense.adam_weights.v_hat = np.hstack([
                self.dense.adam_weights.v_hat,
                np.zeros((self.dense.adam_weights.v_hat.shape[0], no_of_new_token))
            ])

        if not np.isscalar(self.dense.adam_bais.m):

            self.dense.adam_bais.m = np.hstack([
                self.dense.adam_bais.m,
                np.zeros((1, no_of_new_token))
            ])

            self.dense.adam_bais.v = np.hstack([
                self.dense.adam_bais.v,
                np.zeros((1, no_of_new_token))
            ])

            self.dense.adam_bais.m_hat = np.hstack([
                self.dense.adam_bais.m_hat,
                np.zeros((1, no_of_new_token))
            ])

            self.dense.adam_bais.v_hat = np.hstack([
                self.dense.adam_bais.v_hat,
                np.zeros((1, no_of_new_token))
            ])

    def update(self,batch_size):
        for i in range(self.encoder_unit):
            self.transformer_encoder[i].update(batch_size=batch_size)
        
        for i in range(self.decoder_unit):
            self.transformer_dcoder[i].update(batch_size=batch_size)
     
        self.e_encoder.update(batch_size=batch_size)
        self.e_decoder.update(batch_size=batch_size)
        self.dense.update(batch_size=batch_size)

    def train_output_layer(self, decoder_prediction, real_output):

        dense_output = self.dense.forward(decoder_prediction)

        softmax_output = self.softmax.forward(z=dense_output)

        correct_prob = softmax_output[np.arange(len(real_output)), real_output]

        loss = -np.mean(np.log(correct_prob + 1e-8))
        
        # if np.random.rand() < 0.001:   # about 1% of training steps
        #     pred = np.argmax(softmax_output, axis=1)
        #     print("Prediction:", self.t.decode(pred[:10]))
        #     print("Target    :", self.t.decode(real_output[:10]))
        #     print("-" * 50)
        grad = softmax_output.copy()

        grad[np.arange(len(real_output)), real_output] -= 1

        grad /= len(real_output)

        return loss, grad
    

    def save_model(self, file_name="transformer.pkl"):
    
        model = {
            "dense": self.dense,
    
            "encoder_embedding": self.e_encoder,
            "decoder_embedding": self.e_decoder,
    
            "encoder": self.transformer_encoder,
            "decoder": self.transformer_dcoder,
    
            "tokenizer": self.t,
        }
    
        file = open(file_name, "wb")
        pickle.dump(model, file)
        file.close()
    
        print("Model saved successfully!")



    def load_model(self, file_name="transformer.pkl"):
    
        file = open(file_name, "rb")
        model = pickle.load(file)
        file.close()
    
        self.dense = model["dense"]
    
        self.e_encoder = model["encoder_embedding"]
        self.e_decoder = model["decoder_embedding"]
    
        self.transformer_encoder = model["encoder"]
        self.transformer_dcoder = model["decoder"]
    
        self.t = model["tokenizer"]
    
        print("Chatbot model loaded successfully!")
