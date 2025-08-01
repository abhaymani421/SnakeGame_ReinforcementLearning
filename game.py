import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np 

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#we need to implement a reset function so that our agent resets the game after it gets over
#reward
#play(action)--> computes the direction
#game_iteration
#is_collision
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20 # the size of one block of sanke in pixels 
SPEED = 20

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        

    def reset(self) : 
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)] # for the snake we use a list with three initial values
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # when the game resets the iteration becomes 0 
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action): 
        self.frame_iteration +=1  # with every action the frame iteration increases by 1 
        # 1. collect user input, which is the key we press
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
           
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        reward = 0 
        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration > 1000:
            game_over = True
            reward = -10 
            return reward,game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward,game_over, self.score
    
    def is_collision(self, pt = None): # the snake collision function should be public so the agent can use it 
        if pt is None : 
            pt = self.head ; 
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]: # if point is in the snake body 
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action): #based on the action we want to determine the next move 
        #[stratight,right,left] as mentioned in the notes ; 
        clockwise = [Direction.RIGHT, Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clockwise.index(self.direction)
        if np.array_equal(action,[1,0,0]) : 
            new_dir = clockwise[idx] # no change in direction 
        elif np.array_equal(action,[0,1,0]) : # right turn r->d->l->u
            next_idx = (idx+1)%4 ;  # idx = 4---> 0
            new_dir = clockwise[next_idx] 
        else : #[0,0,1]  left turn r->u->l->d
            next_idx = (idx-1)%4 ; 
            new_dir = clockwise[next_idx] 
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

