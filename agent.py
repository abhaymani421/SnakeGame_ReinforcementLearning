import torch 
import random 
import numpy as np 
from collections import deque # data structure used to store memory 
from game import SnakeGameAI,Direction,Point
from model import Linear_QNet, Qtrainer 
from helper import plot 


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

class Agent : 
    def __init__(self):
        self.n_games = 0 
        self.epsilon = 0 # parameter to control the randomness 
        self.gamma = 0.9 # discont rate 
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = Linear_QNet(11,200,3) # instance of the model 
        self.trainer = Qtrainer(self.model, lr = LR, gamma = self.gamma) 
        #model, trainer 
    
    def get_state(self, game) : 
        head = game.snake[0] # first item of the list is the head 
        # we create four points around the head 
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # boolean values for direction 
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #the 11 values of the state
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # to convert true and false booleand to 0 and 1 
    

    def remember(self,state,action, reward, next_state, done) : 
        self.memory.append((state,action, reward, next_state, done)) # pop left if maximum memory is reached 
        
    def train_long_memory(self) : 
        if len(self.memory) > BATCH_SIZE : 
            mini_sample =  random.sample(self.memory, BATCH_SIZE) # returns a list of tuple 
        else : 
            mini_sample =self.memory 
        states, actions, rewards, next_states, dones = zip(*mini_sample) # this basically puts every state , action............... all together
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self,state,action, reward, next_state, done) : 
        self.trainer.train_step(state,action, reward, next_state, done) # train for one game step

    def get_action(self,state) : # to get the action based on the state  
        # random moves : trade off between explorationa and exploitation 
        self.epsilon = 80 - self.n_games 
        final_move = [0,0,0]  
        #the more games we have the smaller our epsilon will get and the samller our epsilon will get the less frequent random.randint(0,200) will be less than epsilon 
        if random.randint(0,200) < self.epsilon : 
            move = random.randint(0,2)
            final_move[move] = 1 
        else : 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # predict the action based on one state
            move = torch.argmax(prediction).item() 
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()  # setting up an agent
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.n_games
            plot_mean_scores.append(mean_score)

            # Plotting every 10 games to avoid slowdown
            if agent.n_games % 10 == 0:
                plot(plot_scores, plot_mean_scores)

            # Optional: Reduce epsilon periodically for more exploitation
            if agent.n_games % 100 == 0:
                agent.epsilon *= 0.99



if __name__=='__main__' : 
    train()