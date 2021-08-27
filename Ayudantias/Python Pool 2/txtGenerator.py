import random as rd


numberOfCoordenates = 1000

with open('coordenadas{}.txt'.format(numberOfCoordenates), 'w') as file:

    evaluaciones = 'latitude,longitude\n'
    file.write(evaluaciones)
    
    for coordanatesIndex in range(numberOfCoordenates):

        coordanates = str(rd.uniform(-33.641718, -33.299945)) + ','
        coordanates += str(rd.uniform(-70.820771, -70.471303)) + '\n'

        file.write(coordanates)