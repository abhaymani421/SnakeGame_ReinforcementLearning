import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import os 

class Linear_QNet(nn.Module) : # a feed forward neural network 

    def __init__(self, input_size, hidden_size, output_size) : 
        super().__init__() 
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size) 

    def forward(self, x) : 
        x = F.relu(self.linear1(x)) 
        x = self.linear2(x) 
        return x 
    
    def save(self, file_name = 'model.pth') : 
        model_folder_path = './model' 
        if not os.path.exists(model_folder_path) : 
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path,file_name) 
        torch.save(self.state_dict(), file_name)


class Qtrainer : 
    def __init__(self,model,lr,gamma):
        self.lr = lr 
        self.gamma = gamma 
        self.model = model 
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr) 
        self.criterion = nn.MSELoss() # the loss function 
    
    def train_step(self, state,action,reward,next_state,done) : 
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n,x) if there are multiple values then they are handeled other wise for 1 it is handeled below-->

        if len(state.shape) == 1: 
            # (1, x) 
            state = torch.unsqueeze(state, 0) # appends one direction in the begining 
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # tuple with only one value 

        # 1 : predicted Q values with current state 
        pred = self.model(state)
        target = pred.clone() 
        for idx in range((len(done))) : 
            Q_new = reward[idx] 
            if not done[idx] : 
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        # 2 : Q_new = r+y*max(next_predicted_q_value)--> only do this if not done 
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimizer.zero_grad() # to empty the gradient 
        loss = self.criterion(target,pred) 
        loss.backward()

        self.optimizer.step()