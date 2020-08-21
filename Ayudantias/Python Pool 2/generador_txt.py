import random as rd


n_alumnos = 1000

with open('notas.txt', 'w') as file:

    evaluaciones = 'I1,I2,E,T1,T2,T3,T4,T5,T6,T7\n'
    file.write(evaluaciones)
    
    for a in range(n_alumnos):

        notas = ''
        for n in range(10):
            nota = min(7, round(1.4*(rd.random()*6 + 1), 2))
            notas = notas + str(nota) + ","

        notas = notas[:-1] + '\n'
        file.write(notas)