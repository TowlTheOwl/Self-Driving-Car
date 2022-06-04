import pygame
import math
import numpy as np
import random

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


tracks = {
    "default": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_default.png",
        "pos": [850, 820],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
    "difficult": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_difficult.png",
        "pos": [850, 820],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
    "lake": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_lake.png",
        "pos": [850, 820],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
    "lake2": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_lake2.png",
        "pos": [850, 820],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
    "training": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_training.png",
        "pos": [850, 820],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
    "training2": {
        "img": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/track_training2.png",
        "pos": [405, 400],
        "border": (255, 255, 255, 255),
        "finish": (255, 255, 0, 255)
    },
}

car_colors = {
    "red": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/car_red.png",
    "green": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/car_green.png",
    "blue": "C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/imgs/car_blue.png",
}

def car_color(age):
    if age == 0:
        return "red"
    elif age < 5:
        return "green"
    else:
        return "blue"

WINDOW_SIZE = (1920, 1080)
WIN = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Natural Selection")
pygame.font.init()

chosen_track = "training"

TRACK = pygame.image.load(tracks[chosen_track]["img"]).convert()
FPS = 60

CAR_SIZE_X = 60
CAR_SIZE_Y = 30

font1 = pygame.font.Font("C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/Font/Unique.ttf", 100)
font2 = pygame.font.Font("C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/Font/Unique.ttf", 50)
font3 = pygame.font.Font("C:/Users/henry/Coding/Python/Self Driving Car/Self-Driving-Car-main/Font/Unique.ttf", 30)

A_USEFUL_ANGLE = math.degrees(math.atan(CAR_SIZE_Y / CAR_SIZE_X))
BORDER_COLOR = tracks[chosen_track]["border"]
FINISH_COLOR = tracks[chosen_track]["finish"]

show_radar = False

clock = pygame.time.Clock()

images = [(TRACK, (0, 0))]

run = True

best_dist = 0
best_time = 0
time_since_first = 0

sp = tracks[chosen_track]["pos"]

class Car:
    def __init__(self, name, parent_weights, start_pos, mr, keep_weight=False):
        self.name = name
        self.start_pos = start_pos
        self.pos = self.start_pos
        self.center = (self.pos[0] + CAR_SIZE_X / 2, self.pos[1] + CAR_SIZE_Y / 2)
        self.corners = []
        self.sensors = []
        self.mr = mr
        self.age = 0

        # nn related
        self.keep_weight = keep_weight
        self.command = 0
        self.weights = ()
        self.parent_weights = parent_weights
        self.layer = [7, 6, 5]
        self.initialize_weights()

        # img
        self.sprite = pygame.image.load(car_colors[car_color(self.age)]).convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # setup values
        self.angle = 0
        self.vel = 4
        self.max_vel = 12
        self.min_vel = 2
        self.rotation_vel = 2
        self.acceleration = 0.1

        # track status
        self.alive = True
        self.distance = 0
        self.time = 0
        self.finish = False

    def return_weights(self):
        return self.weights

    def increase_age(self):
        self.pos = self.start_pos
        self.center = (self.pos[0] + CAR_SIZE_X / 2, self.pos[1] + CAR_SIZE_Y / 2)
        self.age += 1

        self.sprite = pygame.image.load(car_colors[car_color(self.age)]).convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.angle = 0
        self.vel = 4

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
        num_weights = 0
        for i in range(len(self.layer)-1):
            num_weights += (self.layer[i]+1)*self.layer[i+1]

        if self.parent_weights is None:
            for _ in range(num_weights):
                action = random.randint(-1,1)
                weights.append(random.random()*action)
        else:
            parent_weights = ()

            for j in range(len(self.parent_weights)):
                parent_weights += (self.parent_weights[j].flatten(),)
            new_parent_weights = parent_weights[0]
            for weight_num in range(1, len(parent_weights)):
                new_parent_weights = np.concatenate(parent_weights)
            self.parent_weights = new_parent_weights

            if self.keep_weight:
                weights = self.parent_weights
            else:

                for i in range(num_weights):
                    action = random.randint(-1, 1)
                    weights.append(random.random()/self.mr * action + self.parent_weights[i])

        # format the array
        self.weights = []
        prev_layers = 0
        for i in range(len(self.layer)-1):
            weight = weights[prev_layers:prev_layers+((self.layer[i]+1) * self.layer[i+1])]
            new_weight = []


            for j in range(self.layer[i+1]):
                l = []
                for k in range(self.layer[i]+1):
                    l.append(weight[j * (self.layer[i]+1) + k])
                new_weight.append(l)
            self.weights.append(np.array(new_weight))

            prev_layers += (self.layer[i]+1) * self.layer[i+1]

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

    def return_name(self):
        return self.name

    def return_age(self):
        return self.age

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

    def return_distance(self):
        if self.finish:
            return float('inf')
        else:
            return self.distance

    def return_time(self):
        return self.time

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
        out = np.array([sensor[1] for sensor in self.sensors])
        one = np.array([1]).T

        for i in range(len(self.layer)-1):
            inp = np.concatenate((one, out))
            weight = self.weights[i]
            out = weight @ inp
            out = np.array([sigmoid(x) for x in out])

        result = np.where(out == max(out))[0].tolist()
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


def update(win, imgs, gen, age_list, car_data):
    for img, pos in imgs:
        win.blit(img, pos)
    text = font1.render(f"Gen #: {gen}", True, (0, 0, 0))
    tsf = font2.render(f"Time since first: {time_since_first}", True, (0, 0, 0))
    age_title = font2.render("Current cars: ", True, (0, 0, 0))
    ages = []
    for i in range(len(age_list)):
        ages.append(font3.render(f"{i+1}: {age_list[i][0]}, {age_list[i][1]}", True, (0, 0, 0)))
    ranking = []
    for i in range(len(car_data)):
        dist = car_data[i][0]
        if dist == float('inf'):
            dist = 'finish'
        else:
            dist = int(car_data[i][0])
        ranking.append(font2.render(f"{i+1}: {car_data[i][1].return_name()}, {dist}", True, (0, 0, 0)))
    win.blit(text, (700, 20))
    win.blit(tsf, (20, 900))
    win.blit(age_title, (1300, 10))
    for i, p in enumerate(ages):
        win.blit(p, (1200 + 400* (i // 5), 70+30*(i%5)))
    for i, e in enumerate(ranking):
        win.blit(e, (20, 80 + 60*(i)))


def create_offsprings(cars, car_dist, cars_time, mr, num_survive):
    offsprings = []
    total_cars = len(cars)
    only_dist = np.array([data[0] for data in car_dist])
    passed = np.where(only_dist == float('inf'))[0].tolist()
    dead_names = []
    for car in cars:
        dead_names.append(car.return_name())

    # no cars has finished
    if len(passed) == 0:
        sorted_car_dist = sorted(car_dist, key=lambda x: x[0], reverse=True)
        for i in range(num_survive):
            offsprings.append(sorted_car_dist[i][1])

    elif len(passed) <= num_survive:
        passed_cars = [data[1] for ind, data in enumerate(car_dist) if ind in passed]
        offsprings.extend(passed_cars)
        sorted_car_dist = sorted(car_dist, key=lambda x: x[0], reverse=True)

        i=0
        while len(offsprings) < survive:
            car_check = sorted_car_dist[i][1]
            if car_check not in offsprings:
                offsprings.append(car_check)
            i += 1


    # > num_survive cars finished
    else:
        passed_cars_time = [data for ind, data in enumerate(cars_time) if ind in passed]
        times = sorted(passed_cars_time, key=lambda x: x[0])
        for i in range(num_survive):
            offsprings.append(times[i][1])

    num_to_make = total_cars - num_survive
    all_weights = []
    for car in offsprings:
        name_idx = dead_names.index(str(car.return_name()))
        dead_names.pop(name_idx)
        all_weights.append(car.return_weights())
        car.increase_age()

    for car in cars:
        if car not in offsprings:
            del car

    for i in range(num_to_make):
        parent = random.choice(all_weights)
        baby_name = random.choice(dead_names)
        dead_names.remove(baby_name)
        baby_car = Car(baby_name, parent, sp, mr)
        offsprings.append(baby_car)
    return offsprings

def get_car_age(cars):
    age_list = []
    for car in cars:
        age_list.append((car.return_name(), car.return_age()))
    return age_list

gen = 1
mutation_rate = 1.5
sim_data_dist = np.array([])
sim_data_time = np.array([])

names = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
]

car_list = []

total_car_num = 0

for name in names:
    new_car = Car(name, None, sp, mutation_rate)
    car_list.append(new_car)
    total_car_num += 1

zoom = True
survive = 6

age_list = get_car_age(car_list)

while run:
    FPS = 60
    car_status = []
    all_car_dist = []
    all_car_time = []
    car_data = []

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
        zoom = True
    if keys[pygame.K_x]:
        zoom = False

    if zoom:
        FPS = 0

    for car in car_list:
        car_status.append(car.life_status())
        all_car_dist.append((car.return_distance(), car))
        all_car_time.append((car.return_time(), car))


    old_cars = all_car_dist[:survive]
    new_cars = all_car_dist[survive:]

    new_car_status = []
    for car in new_cars:
        new_car_status.append(car[1].life_status())

    worst_old = min(old_cars, key=lambda x: x[0])
    try:
        best_new = max(new_cars, key=lambda x: x[0])
    except Exception as exc:
        print(all_car_dist)
        raise exc

    sorted_car_data = sorted(all_car_dist, key=lambda x: x[0], reverse=True)

    update(WIN, images, gen, age_list, sorted_car_data)

    for car in car_list:
        car.update(WIN, TRACK)

    pygame.display.flip()
    only_dist = [data[0] for data in all_car_dist]

    if float('inf') in only_dist:
        time_since_first += 1

    if (not any(new_car_status)) and worst_old[0] > best_new[0] and gen != 1:
        gen += 1
        time_since_first = 0

        new_car_list = []
        new_car_list = [car[1] for car in old_cars]
        survivor_names = [car.return_name() for car in new_car_list]
        num_to_make = total_car_num - survive
        dead_names = []
        for name in names:
            if name not in survivor_names:
                dead_names.append(name)
        all_weights = []
        for car in car_list:
            if car not in new_car_list:
                del car
        for car in new_car_list:
            all_weights.append(car.return_weights())
            car.increase_age()

        for i in range(num_to_make):
            parent = random.choice(all_weights)
            baby_name = random.choice(dead_names)
            dead_names.remove(baby_name)
            baby_car = Car(baby_name, parent, sp, mutation_rate)
            new_car_list.append(baby_car)

        car_list = new_car_list[:]
        age_list = get_car_age(car_list)
        top_3 = [x[1] for x in age_list[:4]]
        old = [y for y in top_3 if y > 8]
        if len(old) == 4:
            mutation_rate = 2
        else:
            mutation_rate = 1.5

    elif (not any(car_status) and car_status != []) or time_since_first > 200:
        gen += 1

        car_list = create_offsprings(car_list, all_car_dist, all_car_time, mutation_rate, survive)
        time_since_first = 0
        age_list = get_car_age(car_list)
        top_3 = [x[1] for x in age_list[:4]]
        old = [y for y in top_3 if y > 8]
        if len(old) == 4:
            mutation_rate = 2
        else:
            mutation_rate = 1.5
    clock.tick(FPS)

print(sorted_car_data[0][1].return_weights())
pygame.quit()
