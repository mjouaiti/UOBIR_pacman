import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from pauser import Pause
import numpy as np
import random

# Training parameters
n_training_episodes = 10000
learning_rate = 0.7

# Evaluation parameters
n_eval_episodes = 100

# Environment parameters
max_steps = 999
gamma = 0.95
eval_seed = []

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

def initialize_q_table(state_space, action_space):
    Qtable = None
    return Qtable
    
def epsilon_greedy_policy(Qtable, state, epsilon):
    return
    
def greedy_policy(Qtable, state):
    return

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.score = 0
        self.level = 0
        self.lives = 5
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.pause = Pause(False)
        
        self.Qtable = initialize_q_table(SCREENWIDTH*SCREENHEIGHT, 5)
#        self.train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, self.Qtable)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)
        
    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background = self.mazesprites.constructBackground(self.background, self.level%5)
        self.nodes = NodeGroup("maze1.txt")
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)


    def update(self, action=-1):
        reward, done = 0, 0
        dt = self.clock.tick(30) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        
        if not self.pause.paused:
            self.pacman.update(dt, action)
            self.ghosts.update(dt)
            s = self.checkPelletEvents()
            reward += s
            g = self.checkGhostEvents()
            if g != 0:
                if g == S_GHOST:
                    reward -= 10
                    done = 1
                else:
                    reward += 10

        
        state = self.pacman.position.asInt()[0] + SCREENWIDTH * self.pacman.position.asInt()[1]
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()
        if self.score >= 2600:
            done = 1
        return state, reward, done
        
    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)
        if self.score >= 2600:
            return 1
        return 0

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
               self.ghosts.startFreight()
               return 5
            return 1
        return 0
        
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=False)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()
        
    def step(self, action):
        new_state, reward, done= self.update(action)
        
        return new_state, reward, done
    
        
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable):
        for episode in range(n_training_episodes):
            state = self.restartGame()
            step = 0
            done = False

            # repeat
            for step in range(max_steps):
                
                if done:
                    break
        return Qtable
        
    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        return self.pacman.position.asInt()[0] + SCREENWIDTH * self.pacman.position.asInt()[1]

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)
                
    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    ghost.startSpawn()
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    self.nodes.allowHomeAccess(ghost)
                    return S_GHOST_FREIGHT
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
                        return S_GHOST
        return 0

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()
        
    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
            
        pygame.display.update()



if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
