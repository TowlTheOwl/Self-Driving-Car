# Code Optimized and Inspired By: NeuralNine (Florian Dedov)

import pygame
import math
import numpy as np
import random
import ast


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


WINDOW_SIZE = (1920, 1080)
WIN = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Car")

TRACK = pygame.image.load("imgs/track_training.png").convert()

FPS = 60

CAR_SIZE_X = 60
CAR_SIZE_Y = 30

car_colors = {
    "red": "imgs/car_red.png",
    "green": "imgs/car_green.png",
    "blue": "imgs/car_blue.png",
    "duck": "imgs/car_duck.png",
}

A_USEFUL_ANGLE = math.degrees(math.atan(CAR_SIZE_Y / CAR_SIZE_X))
BORDER_COLOR = (255, 255, 255, 255)
FINISH_COLOR = (255, 255, 0, 255)

# font setup
pygame.font.init()
font1 = pygame.font.Font("Font/Unique.ttf", 100)
font2 = pygame.font.Font("Font/Unique.ttf", 50)


class Car:
    def __init__(self, color, parent_weights, mr, keep_weight=False):
        self.pos = [850, 820]
        self.center = (self.pos[0] + CAR_SIZE_X / 2, self.pos[1] + CAR_SIZE_Y / 2)
        self.corners = []
        self.sensors = []
        self.mr = mr

        # nn related
        self.keep_weight = keep_weight
        self.command = 0
        self.weights = ()
        self.parent_weights = parent_weights
        self.l1 = 7
        self.l2 = 6
        self.l3 = 5
        self.initialize_weights()

        # img
        self.sprite = pygame.image.load(car_colors[color]).convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # setup values
        self.angle = 0
        self.vel = 4
        self.center = [self.pos[0] + CAR_SIZE_X / 2, self.pos[1] + CAR_SIZE_Y / 2]
        self.max_vel = 12
        self.min_vel = 2
        self.rotation_vel = 2
        self.acceleration = 0.1

        # track status
        self.alive = True
        self.distance = 0
        self.time = 0
        self.finish = False

    def initialize_weights(self):
        # with bias term:
        # 1st layer: (5+1) x 5 = 30 values
        # 2nd layer: (5+1) x 4 = 24 values

        # without:
        # 1st layer: 5 x 5 = 25 values
        # 2nd layer: 5 x 4 = 20 values
        # output = array with 4 elements, the max values gets executed

        # initialize/mutate neural network
        weights = []
        if self.parent_weights is None:
            for _ in range((self.l1+1) * self.l2 + (self.l2+1) * self.l3):
                weights.append(random.random())
        else:
            parent_weight1 = self.parent_weights[0].flatten()
            parent_weight2 = self.parent_weights[1].flatten()
            self.parent_weights = np.concatenate((parent_weight1, parent_weight2))
            if self.keep_weight:
                weights = self.parent_weights
            else:
                for i in range((self.l1+1) * self.l2 + (self.l2+1) * self.l3):
                    action = random.randint(-1, 1)
                    weights.append(random.random()/mutation_rate * action + self.parent_weights[i])

        # format the array
        weight1 = weights[:(self.l1+1) * self.l2]
        new_weight1 = []
        for i in range(self.l2):
            l = []
            for j in range(self.l1+1):
                l.append(weight1[i*(self.l1+1) + j])
            new_weight1.append(l)

        weight2 = weights[(self.l1+1) * self.l2: (self.l1+1) * self.l2+(self.l2+1) * self.l3]
        new_weight2 = []
        for i in range(self.l3):
            l = []
            for j in range(self.l2+1):
                l.append(weight2[i*(self.l2+1) + j])
            new_weight2.append(l)

        self.weights = [np.array(new_weight1), np.array(new_weight2)]

    def draw(self, win):
        rect = self.rotated_sprite.get_rect(center=self.sprite.get_rect(topleft=self.pos).center)
        win.blit(self.rotated_sprite, rect)

    def life_status(self):
        return self.alive

    def move(self, track):
        radians = math.radians(self.angle)
        vertical = math.sin(radians) * self.vel
        horizontal = math.cos(radians) * self.vel

        y = self.pos[1]
        x = self.pos[0]
        y -= vertical
        x += horizontal
        self.pos = [x, y]
        self.update_corners()
        self.check_collision(track)

        self.time += 1
        self.distance += self.vel

    def accelerate(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)

    def decelerate(self):
        self.vel = max(self.vel - self.acceleration, self.min_vel)

    def turn_left(self):
        self.angle += self.rotation_vel

    def turn_right(self):
        self.angle -= self.rotation_vel

    def check_collision(self, track):
        self.alive = True
        for corner in self.corners:
            if track.get_at((int(corner[0]), int(corner[1]))) == BORDER_COLOR:
                self.alive = False
                break
            elif track.get_at((int(corner[0]), int(corner[1]))) == FINISH_COLOR:
                self.alive = False
                self.finish = True
                break

    def check_sensors(self, degree, track):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not track.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # pythagorean theorem
        distance = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.sensors.append([(x, y), distance])

    def draw_radar(self, win):
        for radar in self.sensors:
            position = radar[0]
            pygame.draw.line(win, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(win, (0, 255, 0), position, 5)

    def nn(self):
        inp = np.array([sensor[1] for sensor in self.sensors])
        one = np.array([1]).T

        # --- FIRST LAYER --- #

        inp = np.concatenate((one, inp))
        l1_weight = self.weights[0]
        layer2 = l1_weight @ inp.T
        layer2 = np.array([sigmoid(x) for x in layer2])

        # --- SECOND LAYER --- #

        layer2 = np.concatenate((one, layer2))
        l2_weight = self.weights[1]
        out = l2_weight @ layer2
        out = np.array([sigmoid(x) for x in out])
        result = np.where(out == max(out))
        self.command = int(result[0])

    def update_corners(self):
        # calculate corners
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + A_USEFUL_ANGLE))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + A_USEFUL_ANGLE))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + (180 - A_USEFUL_ANGLE)))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + (180 - A_USEFUL_ANGLE)))) * length]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + (180 + A_USEFUL_ANGLE)))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + (180 + A_USEFUL_ANGLE)))) * length]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + (360 - A_USEFUL_ANGLE)))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + (360 - A_USEFUL_ANGLE)))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

    def update(self, win, track):
        if self.alive:
            self.sensors = []
            for degree in range(-90, 91, 30):
                self.check_sensors(degree, track)

            if show_radar:
                self.draw_radar(win)

            self.nn()

            # determine action
            to_execute = [self.accelerate, self.decelerate, self.turn_left, self.turn_right]
            if self.command != 4:
                to_execute[self.command]()
            # if command is not None:
            #     to_execute[command]()

            # update variables
            self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
            self.center = (self.pos[0] + CAR_SIZE_X / 2, self.pos[1] + CAR_SIZE_Y / 2)

            # move
            self.move(track)

        self.draw(win)


run = True
clock = pygame.time.Clock()

images = [(TRACK, (0, 0))]


def update(win, imgs, gen):
    for img, pos in imgs:
        win.blit(img, pos)
    comp_status = "OFF"
    cs_color = (200, 100, 100)
    if comp_mode:
        comp_status = "ON"
        cs_color = (100, 200, 100)
    text = font1.render(f"Gen #: {gen}", True, (0, 0, 0))
    win.blit(text, (20, 20))
    text2 = font2.render(f"Competition Mode: {comp_status}", True, cs_color)
    stat = font2.render(f"Best distance: {round(best_dist)}", True, (0, 0, 0))
    stat2 = font2.render(f"Best time: {round(best_time)}", True, (0, 0, 0))
    tsf = font2.render(f"Time since first: {time_since_first}", True, (0, 0, 0))
    win.blit(text2, (20, 110))
    win.blit(stat, (1000, 900))
    win.blit(stat2, (1000, 950))
    win.blit(tsf, (20, 900))


def create_offsprings(weight, base_mr, num_competitor, num_good, num_normal, num_mutator):
    offsprings = [Car("green", weight, base_mr, True)]
    for _ in range(num_competitor):
        offsprings.append(Car("blue", weight, base_mr*2))
    for _ in range(num_good):
        offsprings.append(Car("blue", weight, base_mr))
    for _ in range(num_normal):
        offsprings.append(Car("blue", weight, base_mr/4))
    for _ in range(num_mutator):
        offsprings.append(Car("red", weight, base_mr/8))
    return offsprings


# base_weight = (-1.4335413488674602, 4.105591988921176, 2.6865865516041993, -1.7547456838357256, -4.534929158847147, 0.44518875360402754, -3.603922193460095, 0.41571418359920265, -0.14350646083446594, -0.5435286978805753, -1.31166493805272, 0.09329964680541702, 3.6359471825278393, 1.6184452592125913, 2.1900281906324603, -0.6187658576423736, -0.7456546399731752, 3.0807122836862018, -1.4109226488148408, 3.446271947992158, -0.5725176066336232, 1.2125367254408406, 1.753768169757701, 2.9919606244098773, 2.210022238418384, 0.28765925552161475, -3.0972647930648187, 0.6631111886226815, -0.24458761780627358, -4.391081883860312, -1.0330634964721104, 0.7696979405469375, -1.2279108504779348, -5.054137983949918, -1.6352164520612344, -1.1735799111152214, 1.1601602752066746, -0.16786399316383138, 4.691708772417237, 0.15473688914607525, -2.4725193752597256, 0.9676467375005491, -3.882656283194523, 2.3387221099122883, -1.2862881943845001, -3.4808522523093943, 1.3701492439190834, -3.0283290296090706, -0.7306210191288608, 1.619883367820715, -2.4212393667893997, -0.21979819674455547, -1.7554386138867817, 1.5796187887135824, -0.44364931006980834, 5.970629833503108, -0.3600281065924301, -1.6313103843075216, -1.4029575175190674, 5.225272788145438, -0.652040983810081, 1.7145553987792312, -2.5673128043238425, -1.2220787502099808, 3.7611577896666297, 0.8324687720694954, 3.9441475618879283, 1.7004315707025408, 0.02512653288348432, -3.413964523764866, 1.6767073679803945, 2.6566017262237405, 4.012163716278209, 3.862822836454882, -4.280502707438524, -1.4723145933818294, -0.22247932953071847, 2.8200244667573418, 3.4405654027725494, -0.3697413221773955, 2.779025769425674, -0.7618166931352719, 0.9611603398973064)
base_weight = """[array([[ 0.63624245,  1.55331488,  0.49317642, -0.25911412, -0.20866609,
         0.53653983,  1.48171649,  0.37091641],
       [-0.59257176, -1.0095923 , -1.37349735,  2.05840837,  1.21114301,
         1.25159171, -0.55686389,  0.62418297],
       [ 2.4216849 , -0.27459679,  0.70120914, -0.83790039, -0.07249071,
         0.60263128,  0.63773902, -0.12944863],
       [-1.30176087, -1.54786603,  0.58206923,  3.25385717,  2.88510229,
         0.5969085 ,  0.86913439, -0.65081809],
       [ 1.98946334, -0.63201611, -0.07758247,  2.11998538,  0.31166168,
         0.90688185,  1.34623798,  0.11245827],
       [-1.49653937,  2.095834  , -0.32550697,  1.12222189, -1.92902   ,
        -0.4149984 ,  2.30318566, -0.76537695]]), array([[-0.42341454,  0.55737474,  0.68906518, -0.69482925,  1.29517508,
        -1.03705237, -0.8838846 ],
       [ 0.12816755,  0.93808577, -0.81158128, -0.55252959,  2.81138374,
        -3.02826877,  0.21514645],
       [ 1.08565381, -0.26879112,  2.22112428,  1.55343107,  0.52710632,
        -0.05391861,  0.7102321 ],
       [-0.58482581,  2.50279178,  0.78566516, -0.1348877 ,  0.18206798,
         1.1382043 ,  1.56683157],
       [-0.34748698,  1.12974537,  2.43800868,  0.62178248,  0.14156704,
        -0.14481793,  0.21123123]])]"""
base_weight = base_weight.replace("array(", "")
base_weight = base_weight.replace(")", "")
base_weight = ast.literal_eval(base_weight)
for i in range(len(base_weight)):
    base_weight[i] = np.array(base_weight[i])

gen = 1

# bigger rate = less change
comp_mode = False
mutation_rate = 1
best_dist = 0
best_time = 0
time_since_first = 0
sim_data_dist = []
sim_data_time = []

blue_car = Car("blue", base_weight, mutation_rate)
green_car = Car("green", base_weight, mutation_rate, True)
# duck = Car("duck", base_weight, mutation_rate)
car_list = [blue_car, green_car]

show_radar = False

while run:
    FPS = 60
    car_status = []
    car_dist = []
    car_time = []
    car_finish = []

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_BACKSPACE]:
        run = False
        break
    if keys[pygame.K_1]:
        show_radar = True
    if keys[pygame.K_0]:
        show_radar = False
    if keys[pygame.K_z]:
        FPS = 200

    update(WIN, images, gen)

    for car in car_list[1:]:
        car.update(WIN, TRACK)
    car_list[0].update(WIN, TRACK)

    pygame.display.flip()

    clock.tick(FPS)

    for car in car_list:
        car_status.append(car.life_status())
        car_finish.append(car.finish)

    if any(car_finish):
        comp_mode = True
        time_since_first += 1

    if (not any(car_status) and car_status != []) or time_since_first > 200:
        gen += 1

        for car in car_list:
            car_dist.append(car.distance)
            car_time.append(car.time)
            car_finish.append(car.finish)

        best_dist = max(car_dist)
        if car_dist.count(best_dist) > 1:
            index_pos_list = []
            index_pos = 0
            while True:
                try:
                    # Search for item in list from indexPos to the end of list
                    index_pos = car_dist.index(best_dist, index_pos)
                    # Add the index position in list
                    index_pos_list.append(index_pos)
                    index_pos += 1
                except ValueError as e:
                    break

            best_cars_time = []
            for i in range(len(car_list)):
                if i in index_pos_list:
                    best_cars_time.append(car_time[i])
            best_time = min(best_cars_time)
            best_idx = car_time.index(best_time)

        elif car_finish.count(True) > 1:
            index_pos_list = []
            index_pos = 0
            while True:
                try:
                    index_pos = car_finish.index(True, index_pos)
                    index_pos_list.append(index_pos)
                    index_pos += 1
                except ValueError as e:
                    break

            best_cars_time = []
            for i in range(len(car_list)):
                if i in index_pos_list:
                    best_cars_time.append(car_time[i])
            best_time = min(best_cars_time)
            best_idx = car_time.index(best_time)
        else:
            best_idx = car_dist.index(best_dist)

        best_car = car_list[best_idx]
        best_weight = best_car.weights

        for car in car_list:
            del car
        car_list = []
        time_since_first = 0
        sim_data_dist.append(round(best_dist, 3))
        sim_data_time.append(round(min(car_time), 3))
        mutation_rate = 1
        if comp_mode:
            mutation_rate = 10
            car_list = create_offsprings(best_weight, mutation_rate, 10, 5, 0, 0)
        else:
            if best_dist < 1000:
                mutation_rate = 0.5
                car_list = create_offsprings(best_weight, mutation_rate, 3, 3, 7, 7)
            elif best_dist < 2000:
                car_list = create_offsprings(best_weight, mutation_rate, 7, 7, 3, 3)
            else:
                mutation_rate = 3
                car_list = create_offsprings(best_weight, mutation_rate, 10, 5, 2, 2)

pygame.quit()
print(best_weight)
print(sim_data_dist)
print(sim_data_time)
