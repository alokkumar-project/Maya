import numpy as np
class Dense:
    def __init__(self,nueron,learning_rate=0.01,weight_initialisation='He'):
        self.weights = None
        
        self.learning_rate=learning_rate
        self.bais  = np.zeros((1,nueron))
        self.nueron=nueron
        
        self.avg_dweights=0
        self.avg_dbais = 0
        self.adam_weights=Adam(learning_rate=learning_rate)
        self.adam_bais=Adam(learning_rate=learning_rate)
    def forward(self,x):
        self.input=x
        if(self.weights is None):
            self.weights = np.random.randn(x.shape[1],self.nueron)*np.sqrt(2/x.shape[1])
        self.z = np.dot(x,self.weights)+self.bais
        return self.z
    def backprop(self,pre_gradient):
        batch_size = self.input.shape[0]
        
        self.dweights = np.dot(self.input.T,pre_gradient)
        self.dbais = np.sum(pre_gradient,axis=0,keepdims=True)
        dinput = np.dot(pre_gradient,self.weights.T)
        # self.avg_dweights = (self.avg_dweights+self.avg_dweights)
        self.avg_dweights = self.avg_dweights+self.dweights
        self.avg_dbais +=self.dbais
        return dinput
    def update(self,batch_size):
        self.avg_dweights = self.adam_weights.update(self.avg_dweights)
        self.avg_dbais = self.adam_bais.update(self.avg_dbais)
        self.weights = self.weights-self.avg_dweights/batch_size
        self.bais = self.bais-self.avg_dbais/batch_size
        self.avg_dbais=0
        self.avg_dweights=0
        
class Leaky_Relu:
    def forward(self,x):
        self.input=x
        return np.where(x > 0, x,0.01*x)
    def backprop(self,pre_gradient):
        return pre_gradient*np.where(self.input > 0, 1, 0.01)
class Softmax:
    def __init__(self):
        pass
    def forward(self,z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def backprop(self,z):
        der_mat =[]
        for current_row in z:
            dummy = np.zeros((len(current_row),len(current_row)))
            
            for i in range(len(current_row)):
                for j in range(len(current_row)):
                    if(i==j):
                        dummy[i][j]=current_row[i]*(1-current_row[i])
                    else:
                        dummy[i][j]=-current_row[i]*current_row[j]
            der_mat.append(dummy)

        return np.array(der_mat)
                    
class Sigmoid:
    def __init__(self):
        pass
    def forward(self,z):
        value= np.clip(z,-500,500)
        self.output= 1/(1+np.exp(-value))
        return self.output
    def backprop(self,prev_grad):
        return prev_grad*(self.output*(1-self.output))
class BinaryEntropyLoss:
    def forward(self,y_hat,y_test):
        self.y_hat = y_hat
        self.y_test=y_test
        eps = 1e-6
        loss =-(y_test*(np.log(y_hat+eps))+(1-y_test)*(np.log(1-y_hat+eps)))
        return np.mean(loss)
    def backprop(self):
        eps = 1e-6
        return (self.y_hat - self.y_test) / ((self.y_hat + eps) * (1 - self.y_hat + eps))
class Add:
    def forward(self,x,residual):
        self.input=x
        self.residual=residual
        return x+residual
    def backprop(self,prev_gradient):
        return prev_gradient,prev_gradient
class Feed_Forward:
    def __init__(self,input_dimension,hidden_nueron,learning_rate=0.01):
        self.d1 = Dense(hidden_nueron,learning_rate=learning_rate)
        
        self.hidden_nueron=hidden_nueron
        
        self.leaky_relu = Leaky_Relu()
        self.d2= Dense(input_dimension,learning_rate=learning_rate)
    def forward(self,x,training='n',dropout=0):
        self.dropout = dropout
        d1_output = self.d1.forward(x)
        relu_output = self.leaky_relu.forward(d1_output)
        d2_output = self.d2.forward(relu_output)
        keep_prob =1-dropout
        if(dropout>0 and training=='y'):
            self.mask = (np.random.rand(*d2_output.shape)<keep_prob).astype(float)
            d2_output = self.mask*d2_output/keep_prob

        return d2_output
    def backprop(self,previous_gradient):
        if(self.dropout>0):
            keep_prob=1-self.dropout
            previous_gradient = previous_gradient*self.mask/keep_prob
        d2_gradient = self.d2.backprop(previous_gradient)
        relu_gradient = self.leaky_relu.backprop(d2_gradient)
        d1_gradient = self.d1.backprop(relu_gradient)
        return d1_gradient
    def update(self,batch_size):
        self.d1.update(batch_size=batch_size)
        self.d2.update(batch_size=batch_size)

class LayerNorm:
    def __init__(self,learning_rate=0.01):
        self.learning_rate=learning_rate
        self.avg_dgamma =0
        self.avg_dbeta=0
        self.gamma=None
        self.adam_gamma = Adam(learning_rate=learning_rate)
        self.adam_beta = Adam(learning_rate=learning_rate)
    def forward(self,x):
        
        self.input=x
        if(self.gamma is None):
            self.gamma = np.ones((1, x.shape[1]))
            self.beta = np.zeros((1,x.shape[1]))
        self.mean = np.mean(x,axis=1,keepdims=True)
        self.std = np.std(x,axis=1,keepdims=True)
        self.x_modi = (x-self.mean)/(self.std+1e-6)
        self.x_out = self.x_modi*self.gamma+self.beta
        return self.x_out
        
    def backprop(self, prev_grad):
    
        self.dgamma = np.sum(prev_grad * self.x_modi, axis=0, keepdims=True)
        self.dbeta = np.sum(prev_grad, axis=0, keepdims=True)
        
        self.avg_dbeta+=self.dbeta
        self.avg_dgamma+=self.dgamma
     
        dx_hat = prev_grad * self.gamma
        
       
        D = self.input.shape[1]
        

        term1 = D * dx_hat
        term2 = np.sum(dx_hat, axis=1, keepdims=True)
        term3 = self.x_modi * np.sum(dx_hat * self.x_modi, axis=1, keepdims=True)
 
        dx = (term1 - term2 - term3) / (D * (self.std + 1e-6))
        
        return dx

    def update(self,batch_size):
        self.avg_dbeta = self.adam_beta.update(self.avg_dbeta)
        self.avg_dgamma = self.adam_gamma.update(self.avg_dgamma)

        self.gamma = self.gamma-self.avg_dgamma/batch_size
        self.beta = self.beta-self.avg_dbeta/batch_size
        
        self.avg_dgamma =0
        self.avg_dbeta=0
class Adam:
    def __init__(self,beta1=0.9,beta2=0.999,learning_rate=0.01):
        self.beta1 = beta1
        self.beta2 =beta2
        self.m_hat=0
        self.v_hat=0
        self.m=0
        self.v=0
        self.t=0
        self.learning_rate=learning_rate
    def update(self,dw):
        self.m= self.beta1*self.m+(1-self.beta1)*dw
        self.v = self.beta2*self.v +(1-self.beta2)*(dw)**2
        self.t+=1
        self.m_hat = self.m/(1-self.beta1**self.t)
        self.v_hat = self.v/(1-self.beta2**self.t)
        final = self.learning_rate*self.m_hat/(np.sqrt(self.v_hat)+1e-6)
    
        return final