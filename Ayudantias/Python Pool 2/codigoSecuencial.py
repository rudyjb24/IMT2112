import time as tm
import numpy as np

def calculateHaversineDistance(latitude1, longitude1, latitude2, longitude2):
    earthRatioInKilometers = 6371

    latitudeDistance = (latitude1 - latitude2) * np.pi / 180
    longitudeDistance = (longitude1 - longitude2) * np.pi / 180

    haversineValue = np.sin(latitudeDistance / 2)**2 + np.sin(longitudeDistance / 2)**2 
    haversineValue *= np.cos(latitude1 * np.pi / 180) * np.cos(latitude2 * np.pi / 180)

    haversineDistance = 2 * np.arcsin(np.sqrt(haversineValue)) * earthRatioInKilometers

    return haversineDistance

t0 = tm.time()

coordinates = []
maxDistance = -1

with open("coordenadas10000.txt", 'r') as file:

    file.readline()

    for linea in file:

        coordenatesInfo = linea.strip().split(',')
        latitude = float(coordenatesInfo[0])
        longitude = float(coordenatesInfo[1])
        
        coordinates.append((latitude, longitude))


for coordinate1 in coordinates:
    for coordinate2 in coordinates:
        
        distance = calculateHaversineDistance(coordinate1[0], coordinate1[1], coordinate2[0], coordinate2[1])

        if distance > maxDistance:
            maxDistance = distance



t1 = tm.time()


print('Tiempo de ejecucion: ', t1-t0)
print('Distancia maxima: ', maxDistance)














"""
1.000  puntos - 6.218794584274292 s - 44.746050984398096 km
10.000 puntos - 579.1201915740967 s - 44.829257919253564 km
"""