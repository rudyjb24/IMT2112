import multiprocessing
import time as tm
import numpy as np


def calculateDistance(coordinate1):
    global coordinates

    maxDistance = -1
    
    for coordinate2 in coordinates:
        newDistance = calculateHaversineDistance(coordinate1[0], coordinate1[1], coordinate2[0], coordinate2[1])
        if newDistance > maxDistance:
            maxDistance = newDistance

    return maxDistance

def calculateHaversineDistance(latitude1, longitude1, latitude2, longitude2):
    earthRatioInKilometers = 6371

    latitudeDistance = (latitude1 - latitude2) * np.pi / 180
    longitudeDistance = (longitude1 - longitude2) * np.pi / 180

    haversineValue = np.sin(latitudeDistance / 2)**2 + np.sin(longitudeDistance / 2)**2 
    haversineValue *= np.cos(latitude1 * np.pi / 180) * np.cos(latitude2 * np.pi / 180)

    haversineDistance = 2 * np.arcsin(np.sqrt(haversineValue)) * earthRatioInKilometers

    return haversineDistance


def calculateMaximum(distances):
    maxDistance = -1

    for distance in distances:
        if distance > maxDistance:
            maxDistance = distance

    return maxDistance




t0 = tm.time()

numberOfWorkers = 8

with open("coordenadas1000.txt", 'r') as file:
    file.readline()
    coordinates = file.readlines()

coordinates = [coordinate.strip().split(',') for coordinate in coordinates]
coordinates = [(float(coordinate[0]), float(coordinate[1])) for coordinate in coordinates]

p = multiprocessing.Pool(numberOfWorkers)

distances = list(p.map(calculateDistance, coordinates))



chuncksSize = int(len(distances)/numberOfWorkers)
chuncks = []

for i in range(numberOfWorkers-1):
    chuncks.append(distances[i*chuncksSize:(i+1)*chuncksSize])

chuncks.append(distances[(i+1)*chuncksSize:])


maxDistances = list(p.map(calculateMaximum, chuncks))

maxDistance = calculateMaximum(maxDistances)

t1 = tm.time()

print('Tiempo de ejecucion total: ', t1-t0)
print('Distancia maxima: ', maxDistance)


"""
1.000  puntos - 1.7019586563110352 s - 44.746050984398096 km
10.000 puntos - 188.71962118148804 s - 44.829257919253564 km
"""