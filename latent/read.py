# Imports
import csv
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from sklearn import linear_model
import json

path = '../dataset/airports.csv'
reader = csv.reader(open(path, 'rt'), delimiter=',')
airports = defaultdict(str)
header = next(reader)
for line in reader:
    d = dict(zip(header, line))
    if d['IATA_CODE'].isdigit(): continue
    airports[d['IATA_CODE']] = d['STATE']

# Read dataset
path = '../dataset/flights.csv'
reader = csv.reader(open(path, 'rt'), delimiter=',')
dataset = []
airlines_CA = set()
airports_CA = set()
header = next(reader)
to_remove_fields = ['YEAR', 'TAXI_OUT', 'TAIL_NUMBER', 'TAXI_OUT', 'WHEELS_OFF', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN',
                    'DIVERTED','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY']
for line in reader:
    d = dict(zip(header, line))
    if d['ORIGIN_AIRPORT'].isdigit() or d['DESTINATION_AIRPORT'].isdigit(): continue
    if airports[d['ORIGIN_AIRPORT']] != 'CA' or airports[d['DESTINATION_AIRPORT']] != 'CA': continue
    for field in to_remove_fields:
        d.pop(field, None)
    dataset.append(d)
    airlines_CA.add(d['AIRLINE'])
    airports_CA.add(d['ORIGIN_AIRPORT'])
    airports_CA.add(d['DESTINATION_AIRPORT'])


with open("../dataset/flights_CA.json", "w") as f:
    for d in dataset:
        f.write(json.dumps(d) + '\n')

with open("../dataset/airlines_airports_CA.json", "w") as f:
    obj = {'airlines': list(airlines_CA), 'airports': list(airports_CA)}
    f.write(json.dumps(obj) + '\n')
