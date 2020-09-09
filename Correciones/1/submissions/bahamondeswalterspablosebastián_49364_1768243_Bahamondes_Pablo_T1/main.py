if __name__ =="__main__":
    # me surge cierto problema con los workers que hace que todo se ejecute varias
    # veces, no sé como solucionarlo pero aqui esta
    import numpy as np
    import matplotlib.pyplot as plt

    from multiprocess import Pool
    from time import time

    from k_mean_algorithm import k_mean as kalg
    from k_mean_numpy import k_mean as knp
    from k_mean_multiprocess import k_mean as kmp

    """
    K-mean de 3 formas para 5000 datos en R^2 (funciona en dimensionalidad mayor)
    con 4 centros. Las formas son respectivamente:

        * Loops python
        * Utilizando numpy
        * Utilizando multiprocess

    Se puede ver los códigos en detalle y comentado en los otros archivos.
    """
    colores = ["r","g","b","cyan","gray","black","yellow"]
    cantidad_datos = 5000
    data = np.random.rand(cantidad_datos,2)
    k = 4
    n_workers = 4


    t0 = time()
    w = Pool(n_workers)
    means = kmp(k,data,centriods = None, tol = 0,workers = w,n_w= n_workers)
    print(time()-t0)

    # Plot
    for i in range(k):
        for dato in means[i][1]:
            plt.plot(dato[0],dato[1],"*",c=colores[i])
    for i in range(k):
        plt.plot(float(means[i][0][0]),float(means[i][0][1]),"*",c="black")
    plt.title(f"K-means con {k} centros y {cantidad_datos} datos\n"
                f"Usando k_mean_multiprocess (y {n_workers} n_trabajadores)")
    plt.show()



    t0 = time()
    means = kalg(k,data, tol = 0.0)
    print(time()-t0)

    for i in range(k):
        for dato in means[i][1]:
            plt.plot(dato[0],dato[1],"*",c=colores[i])
    for i in range(k):
        plt.plot(float(means[i][0][0]),float(means[i][0][1]),"*",c="black")
    plt.title(f"K-means con {k} centros y {cantidad_datos} datos\n"
                f"Usando k_mean_algorithm (bucles)")

    plt.figure()

    t0 = time()
    means = knp(k,data, tol = 0.0)
    print(time()-t0)

    for i in range(k):
        for dato in means[i][1]:
            plt.plot(dato[0],dato[1],"*",c=colores[i])
    for i in range(k):
        plt.plot(float(means[i][0][0]),float(means[i][0][1]),"*",c="black")
    plt.title(f"K-means con {k} centros y {cantidad_datos} datos\n"
                f"Usando k_mean_numpy (usando librería numpy)")
    plt.show()
