# Imports
import csv
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from sklearn import linear_model
import json
import random
random.seed(2019)

dataset = []
with open('../dataset/flights_CA.json', "r") as f:
    for l in f.readlines():
        dataset.append(json.loads(l))
for d in dataset:
    d['DEPARTURE_DELAY'] = 0.0 if d['DEPARTURE_DELAY'] == '' else float(d['DEPARTURE_DELAY'])
meanDelay = np.mean([float(d['DEPARTURE_DELAY']) for d in dataset])

airports = []
airlines = []
with open('../dataset/airlines_airports_CA.json', "r") as f:
    obj = json.loads(f.readline())
    airports = obj['airports']
    airlines = obj['airlines']

airportGamma = {}
airlineGamma = {}
K = 3

for a in airports:
    airportGamma[a] = [random.random() * 1.0 - 0.5 for _ in range(K)]

for a in airlines:
    airlineGamma[a] = [random.random() * 1.0 - 0.5 for _ in range(K)]

def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])

def pack(airportGamma, airlineGamma):
    theta = []
    for a in airports:
        theta += np.array(airportGamma[a])
    for a in airlines:
        theta += np.array(airlineGamma[a])
    return theta

def unpack(theta):
    global airportGamma
    global airlineGamma
    index = 0
    for a in airports:
        airportGamma[a] = theta[index:(index+K)].tolist()
        index += K
    for a in airlines:
        airlineGamma[a] = theta[index:(index+K)].tolist()
        index += K

batchSize = 10000
def sample(arg):
    global samples
    temp = random.sample(dataset, batchSize)
    samples = [(d['AIRLINE'], d['ORIGIN_AIRPORT'], d['DESTINATION_AIRPORT'], float(d['DEPARTURE_DELAY'])) for d in temp]

samples = []
sample(0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost(theta, lamb):
    unpack(theta)
    cost = np.mean([np.square(
        inner(airlineGamma[airline], airportGamma[origin])
        + inner(airlineGamma[airline], airportGamma[destination])
        + meanDelay - delay
    ) for (airline, origin, destination, delay) in samples])
    return cost

def derivative(theta, lamb):
    unpack(theta)
    dAirportGamma = {}
    dAirlineGamma = {}
    for a in airports:
        dAirportGamma[a] = [0.0 for k in range(K)]
    for a in airlines:
        dAirlineGamma[a] = [0.0 for k in range(K)]
    for airline, origin, destination, delay in samples:
        for k in range(K):
            s = (
                inner(airlineGamma[airline], airportGamma[origin])
                + inner(airlineGamma[airline], airportGamma[destination])
                + meanDelay - delay)
            dAirlineGamma[airline][k] += (airportGamma[origin][k] + airportGamma[destination][k]) * (2.0 * s / len(samples))
            dAirportGamma[origin][k] += (airlineGamma[airline][k]) * (2.0 * s / len(samples))
            dAirportGamma[destination][k] += (airlineGamma[airline][k]) * (2.0 * s / len(samples))
    dtheta = []
    for a in airports:
        dtheta += dAirportGamma[a]
    for a in airlines:
        dtheta += dAirlineGamma[a]
    return np.array(dtheta)

theta = np.array([random.random() * 1.0 - 0.5 for k in range(K*(len(airlines)+len(airports)))])

for iter in range(50000):
    c = cost(theta, 0.0)
    der = derivative(theta, 0.0)
    theta = np.subtract(theta, der * 0.001)
    sample(0)
    if iter % 100 == 0:
        print('iteration: %d, cost: %.6f' % (iter, c))
        with open('../dataset/Latent_%d_train_%d.txt' % (K, iter), 'w') as file:
            file.write(json.dumps({'airlines': airlines, 'airports': airports, 'airlineGamma': airlineGamma, 'airportGamma': airportGamma}) + '\n')
