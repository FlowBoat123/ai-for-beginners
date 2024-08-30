import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

# Initialize Pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 640, 480
BLOCK_SIZE = 20
FPS = 10

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game')

# Define the Snake class
class Snake:
    def __init__(self):
        self.size = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])  # Start with a random direction
        self.color = GREEN
        self.score = 0

    def get_head_position(self):
        return self.positions[0]

    def turn(self, direction):
        # Prevent the snake from reversing
        if self.size > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * BLOCK_SIZE)) % WIDTH), (cur[1] + (y * BLOCK_SIZE)) % HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.size:
                self.positions.pop()

    def reset(self):
        self.size = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.score = 0

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

# Define the Food class
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, (WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE,
                         random.randint(0, (HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE)

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)

def reset_game_state():
    global snake, food
    snake = Snake()  # Reset the snake
    food = Food()    # Reset the food
    state = get_state()  # Get the initial state of the game
    return state

def game_step(action):
    global snake, food
    
    # Map action to snake direction
    if action == 0:  # UP
        snake.turn((0, -1))
    elif action == 1:  # DOWN
        snake.turn((0, 1))
    elif action == 2:  # LEFT
        snake.turn((-1, 0))
    elif action == 3:  # RIGHT
        snake.turn((1, 0))
    
    snake.move()
    
    reward = 0
    done = False
    
    # Check if the snake ate the food
    if snake.get_head_position() == food.position:
        snake.size += 1
        snake.score += 1
        reward = 10  # Positive reward for eating food
        food.randomize_position()
    
    # Check if the snake has collided with itself
    if len(snake.positions) > 2 and snake.get_head_position() in snake.positions[1:]:
        done = True
        reward = -10  # Negative reward for collision
    
    state = get_state()
    return state, reward, done, snake.score, {}

def get_state():
    # Example state representation
    head_x, head_y = snake.get_head_position()
    food_x, food_y = food.position
    distance_x = (food_x - head_x) / WIDTH
    distance_y = (food_y - head_y) / HEIGHT
    
    return np.array([head_x / WIDTH, head_y / HEIGHT, distance_x, distance_y])

# AI
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0) #1 predict
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0) #2 predict
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) #3 fit
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Run the AI-driven game and visualize the moves
if __name__ == "__main__":
    state_size = 4  # Example state size
    action_size = 4  # Example action size (up, down, left, right)
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(100):  # Number of episodes
        state = np.reshape(reset_game_state(), [1, state_size])
        for time in range(1000):
            # AI chooses the action
            action = agent.act(state)
            next_state, reward, done, score, _ = game_step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Draw the game state
            screen.fill(BLACK)
            snake.draw(screen)
            food.draw(screen)
            pygame.display.update()
            pygame.time.Clock().tick(FPS)
            
            if done:
                print(f"Episode {e}/{100}, Score: {score}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    pygame.quit()
