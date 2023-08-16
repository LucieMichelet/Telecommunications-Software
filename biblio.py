# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:16:56 2022

@author: Lucie
"""


import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd


def esperance(Xi):
    """
    fonction qui calcul un estimateur sans biais de l'esperence d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: l'esperence du vecteur Xi
    """

    m = Xi.shape[0]
    return np.sum(Xi) / m



def variance(Xi):
    """
    fonction qui calcul un estimateur avec un biais asymptotique de la variance d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: la variance du vecteur Xi
    """

    m = Xi.shape[0]
    Xi_bar = esperance(Xi)
    return np.sum((Xi - Xi_bar)**2) / (m - 1)


# PARTIE 2

def centre_red(R):
    """
    fonction qui a partir une variable aleatoire R de loi differente, centre et reduit les Xi pour qu'ils soient plus
    homogène a etudier
    :param R: un vecteur aléatoire de taille (m, n)
    :return Rcr: le vecteur aléatoire R modifié de facon que l'esperance de Xi soient 0 et leurs variance 1
    """

    # on récupère les dimensions de R pour créer la matrice résultat Rcr de même dimension
    m, n = R.shape
    Rcr = np.zeros((m, n))

    # pour chaque colonne de R on centre et réduit indépendamment les données car les Xi ne suivent pas les mêmes lois
    for i in range(n):
        Xi = R[:, i]

        # on calcule l'espérence et la variance de chaque Xi
        E = esperance(Xi)
        var = variance(Xi)

        # on "décale" les données de chaque colonnes indépendamment
        Rcr[:, i] = (Xi - E) / np.sqrt(var)

    # on retourne la matrice Rcr qui contient les données centrées et réduites
    return Rcr




def approx(R, k):
    """
    le but de la fonction est de decomposer en vecteur propre la matrice R suivant k direction que l'on doit déterminer,
    apres avoir projeté R sur ces vecteurs la matrice résultante sera dans un EV de dimension plus faible donc
    possiblement affichable sur un plan
    :param R: le vecteur aléatoire, matrice de données de taille (m, n)
    :param k: le nombre de dimension dans lequel on souhaite projeter R
    :return proj: un matrice de taille (m, k)
    """

    # on récupère les dimensions de la matrice R et on créer la matrice proj qui contiendra le résultat
    m, n = R.shape
    proj = np.zeros((m, k))

    # on centre et réduit le vecteur aléatoire R pour que les composantes soient toutes homogènes
    Rcr = centre_red(R)

    # on fait la décomposition SVD de la matrice Rcr, le vecteur U de taille (m, n) et s de taille (n, 1) nous interesse
    # U est une base othonormé de Rcr
    # s contient les valeurs singulière / variance de Rcr trier par ordre décroissante d'importance
    U, s, VT = np.linalg.svd(Rcr)
    u = U[:, :k]

    # on concerve dans proj uniquement les k-composantes les plus importante de la nouvelle base U (variances
    # les plus élevés) : sigma**2 * uj
    for j in range(k):
        proj[:, j] = (s[j]**2) * u[:, j]

    return proj


def correlationdirprinc(R, k):
    """
    parameter R : tableau de données numérique de taille [m;n]
    parameter k : entier inférieur à n
    return Cor : matrice correlation de taille [k:n]

    """
    m, n = R.shape
    Cor = np.zeros((k, n))

    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    

    for j in range(k):
        Yk = Y[:, j]
        for i in range(n):
            Xi = Rcr[:, i]
            Xi = Xi.reshape(m, 1)
            Cor[j, i] = (Yk.T@Xi/(np.sqrt(variance(Yk))))

    return Cor


def ACP2D(R, labelsligne, labelscolonne):
    """
    parameter R : le tableau de données numériques
    parameter labelsligne : noms des lignes (s’ils existent, si non on prendra un vecteur d’entier de 1 à m),
    parameter labelscolonne : noms des colonnes (s’ils existent si non on prendra un vecteur d’entier de 1 à n),
    return :  le graphe qui représente les valeurs des variances σk² et le graphe qui représente le pourcentage de l’explication de la variance de chaque k−composante principal

    """
    m, n = R.shape
    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    
        

    Yi = []
    for i in range(1, n+1):
        Yi.append('Y{}'.format(i))

    # Affichage des deux premiers graphqiues

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    for i in range(n):
        ax1.bar(Yi[i], variance(Y[:, i]), color='#77b5fe', edgecolor='k')
        ax1.text(Yi[i], variance(Y[:, i]), round(variance(Y[:, i]), 2), horizontalalignment='center')

    ax1.set_ylabel('Variance(Yk)')
    ax1.set_xlabel('Composantes')
    ax1.set_title('Variance des composantes principales')

    ax2.pie(s**2, labels=Yi, autopct='%1.1f%%')
    ax2.set_title('Participation à la variance totale')

    # Affichage des deux seconds graphqiues

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

    Proj = approx(R, 2)
    ax3.scatter(Proj[:, 0], Proj[:, 1], c=labelsligne.astype('float'))
    ax3.set_ylabel('Y2')
    ax3.set_xlabel('Y1')
    ax3.set_title('Analyse en composantes principales pour k = 2')

    Cor = correlationdirprinc(R, 2)
    for i in range(n):
        X = Cor[0, i]/(np.sqrt((Cor[0, i]**2+Cor[1, i]**2)))
        Y = Cor[1, i]/(np.sqrt((Cor[0, i]**2+Cor[1, i]**2)))
        ax4.arrow(0, 0, X, Y, width=0.02, length_includes_head=True)
        ax4.text(X, Y, labelscolonne[i])

    ax4.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
             np.sin(np.linspace(0, 2*np.pi, 100)))
    ax4.grid()
    ax4.set_ylabel('Y2')
    ax4.set_xlabel('Y1')
    ax4.set_title('Cercle de corrélation')

    plt.show()


def ACP3D(R, labelsligne, labelscolonne):
    """
    parameter R : le tableau de données numériques
    parameter labelsligne : nom des lignes
    parameter labelscolonne : nom des colonnes
    return :

    """

    m, n = R.shape
    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    
    # Affichage des deux premiers graphqiues 2D

    Yi = []
    for i in range(1, n+1):
        Yi.append('Y{}'.format(i))

    # Affichage des deux premiers graphqiues

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    for i in range(n):
        ax1.bar(Yi[i], variance(Y[:, i]), color='#77b5fe', edgecolor='k')
        ax1.text(Yi[i], variance(Y[:, i]), round(variance(Y[:, i]), 2), horizontalalignment='center')

    ax1.set_ylabel('Variance(Yk)')
    ax1.set_xlabel('Composantes')
    ax1.set_title('Variance des composantes principales')

    ax2.pie(s**2, labels=Yi, autopct='%1.1f%%')
    ax2.set_title('Participation à la variance totale')

    # Affichage des deux seconds graphqiues 3D

    fig = plt.figure(figsize=(12, 6))

    ax3 = fig.add_subplot(121, projection='3d')
    Proj = approx(R, 3)
    ax3.scatter(Proj[:, 0], Proj[:, 1], Proj[:, 2], c=labelsligne.astype('float'))
    ax3.set_xlabel('Y1')
    ax3.set_ylabel('Y2')
    ax3.set_zlabel('Y3')
    ax3.set_title("Analyse en composantes principales pour k = 3")

    Cor = correlationdirprinc(R, 3)
    ax4 = fig.add_subplot(122, projection='3d')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax4.plot_surface(x, y, z, color="b", alpha=0.2)
    ax4.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
             np.sin(np.linspace(0, 2*np.pi, 100)), -1, color="grey")
    ax4.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.ones(100),
             np.sin(np.linspace(0, 2*np.pi, 100)), color="grey")
    ax4.plot(-np.ones(100), np.cos(np.linspace(0, 2*np.pi, 100)),
             np.sin(np.linspace(0, 2*np.pi, 100)), color="grey")

    for i in range(n):
        X = Cor[0, i]/(np.sqrt((Cor[0, i]**2+Cor[1, i]**2+Cor[2, i]**2)))
        Y = Cor[1, i]/(np.sqrt((Cor[0, i]**2+Cor[1, i]**2+Cor[2, i]**2)))
        Z = Cor[2, i]/(np.sqrt((Cor[0, i]**2+Cor[1, i]**2+Cor[2, i]**2)))
        ax4.quiver(0, 0, 0, X, Y, Z, color="b")
        ax4.quiver(0, 1, 0, X, 0, Z, color="r")
        ax4.quiver(-1, 0, 0, 0, Y, Z, color="r")
        ax4.quiver(0, 0, -1, X, Y, 0, color="r")
        ax4.text(X, Y, Z, labelscolonne[i])

    ax4.set_xlabel('Y1')
    ax4.set_ylabel('Y2')
    ax4.set_zlabel('Y3')
    ax4.set_title("Cercle de corrélation et ses projections")

    plt.show()


def ACP(R, labelsligne, labelscolonne, k=0, epsilon=10**-1):

    m, n = np.shape(R)
    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    
    # Affichage des deux premiers graphqiues 2D

    Yi = []
    for i in range(1, n+1):
        Yi.append('Y{}'.format(i))

    # Affichage des deux premiers graphqiues

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    for i in range(n):
        ax1.bar(Yi[i], variance(Y[:, i]), color='#77b5fe', edgecolor='k')
        ax1.text(Yi[i], variance(Y[:, i]), round(variance(Y[:, i]), 2), horizontalalignment='center')

    ax1.set_ylabel('Variance(Yk)')
    ax1.set_xlabel('Composantes')
    ax1.set_title('Variance des composantes principales')

    ax2.pie(s**2, labels=Yi, autopct='%1.1f%%')
    ax2.set_title('Participation à la variance totale')


    # Kaiser
    for i in range(n):
        if variance(Y[:,i])>=1-epsilon :
            k +=1
        elif variance(Y[:,i+1])<1-epsilon :
            break

    
    print(k)

    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    img = ax3.imshow(correlationdirprinc(R, k))
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.yticks(ticks=range(k), labels=[f'Y{i + 1}' for i in range(k)])
    plt.xticks(ticks=range(n), labels=labelscolonne)
    fig2.colorbar(img)
    plt.plot()


# PARTIE 3


def cercle_cor(mat_cor, col_names, show=True):
    n, _ = mat_cor.shape
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), ls='--')

    arrow_style = {"width": 0.02, "length_includes_head": True}

    for i in range(n):
        plt.arrow(0, 0, mat_cor[i, 0], mat_cor[i, 1], **arrow_style)
        plt.annotate(col_names[i], xy=(mat_cor[i, 0], mat_cor[i, 1]))

    plt.title('Cercle de corrélation')
    plt.grid()

    if show:
        plt.show()


def matrice_cor(mat_cor, col_names, show=True):
    n, k = mat_cor.shape

    plt.imshow(mat_cor.T)
    plt.yticks(ticks=range(k), labels=[f'Y{i + 1}' for i in range(k)])
    plt.xticks(ticks=range(n), labels=col_names)
    plt.title('Matrice de corrélation')
    plt.colorbar()

    if show:
        plt.show()


def plot_2D(proj, label, show=True):
    x = proj[:, 0]
    y = proj[:, 1]

    plt.scatter(x, y, c=label)
    plt.title("Analyse en composantes principales 2D")
    plt.xlabel('Y1')
    plt.ylabel('Y2')

    if show:
        plt.show()


def barycentre(S):
    # retourne un matrice de taille (k, p) ou chaque ligne est le barycentre de tous les points d'un catégorie S[k]
    return np.array([np.sum(s, axis=0, dtype=float) / len(s) for s in S])


def norme(x):
    # norme 2 de x
    return np.linalg.norm(x)


def Kmoy(A, k, err=0.01, show=False, save=False):
    """
    L'algorithme des k-means permette de catégoriser / classifier les objets d'un EV en fonction de leurs distances
    euclidiennes relative. Dans notre cas il permet de séparer en k catégorie un nuage de point que l'on
    sait 'relativement ordonné' grace a un travail de décomposition en vecteur propre ACP effectué précédements.
    :param A: la matrice de taille (m, p) que souhaite classifier
    :param k: le nombre de groupe / catégorie que l'on souhaite créer
    :param err: condition d'arrete de l'algorithme : c'est la variation minimal entre les barycentres muu entre 2
                ittérations pour stopper l'algorithme
    :param show: if True affiche en temps réel l'évolution de l'algorithme ittération par ittération
    :param save: if True enregistre le résultat que l'on affiche sur un plan (fonction pour p = 2)
    :return S: la meilleur partition des lignes de A,
               une liste de longueur k qui contient les lignes de A trier selon k catégories exemple,
               pour k = 3: S = [[.., .., ..], [.., .., .., .., ..], [.., ..]]
    """

    # on récupère les dimensions de A
    m, p = A.shape
    # on stocke quelque couleur pour le future affichage
    color = ['b', 'g', 'k', 'y', 'm', 'pink']

    # ------------------------------------- initialistation de l'algorithme --------------------------------------
    # on choisie k ligne différente de A au hasard
    # on recommence tant que l'on a pas k indice différent
    indice = np.random.choice(m, k)
    while (np.unique(indice) != indice).all():
        indice = np.random.choice(m, k)

    # muu est un matrice de taille (k, p) qui contient les coordonnées des barycentres des catégories que l'on cherche
    # on initialise ces barycentres de manière aléatoire a partir des indices
    muu = A[indice, :]
    # on fixe delta_muu est notre variable de souvenir (ittération n-1)
    delta_muu = muu

    # ------------------------------------------- algorithme k-means -------------------------------------------
    # tant que l'algorithme n'est pas stable, que les barycentres se deplacent encore on ittere l'algo
    while norme(delta_muu) > err:
        # on initialise notre partition des lignes de A en k catégorie : [[], [], .., []]
        S = [[] for l in range(k)]

        if show:
            plt.clf()

        # pour chaque ai les ligne de A, on clacule sa distance aux k-barycentres et on ajoute ai au nuage de point
        # dont le barycentre est le plus proche
        for i in range(m):
            ai = A[i, :]

            # on calcule les distances entre ai et les k-barycentres avec la norme 2
            D = [norme(ai - muu[j, :]) for j in range(k)]
            # on récupère l'indice du nuage le plus proche de ai, avec j la catégorie qui lui conviendrait le mieux
            j, _ = min(enumerate(D), key=lambda x: x[1])
            # on ajoute la ligne ai a la meilleur catégorie au vu des barycentres de cette ittération
            S[j].append(ai)

            if show:
                plt.scatter(ai[0], ai[1], c=color[j])

        # il se peut qu'un catégorie de S soit vide, pour éviter les problèmes on ajoute un point 0 Rp, qui
        # n'aura pas d'impacte sur les barycentres muu
        while [] in S:
            S.remove([])
            S.append([np.zeros(p)])

        # on calcule le déplacement moyen des barycentres delta_muu pour la condition d'arrêt ainsi que les nouveaux
        # barycentres muu
        delta_muu = muu - barycentre(S)
        muu = barycentre(S)
        # print(norme(delta_muu))

        if show:
            [plt.scatter(muuk[0], muuk[1], c='r', s=100) for muuk in muu]
            plt.pause(0.2)

    # --------------------------------------------- fin de l'algorithme ----------------------------------------------
    # on calcule l'efficaciter du partionnement S en calculant Val un scalaire
    Val = 0
    for j in range(k):
        for i in range(len(S[j])):
            Val += norme(S[j][i] - muu[j])**2

    # on affiche tout les points si show == True avec la couleur de leurs catégories respectives
    if show:
        plt.title(f'Val = {Val}')
        plt.show()

    # si save == True on enregiste le plot avec comme nom la qualité de la partition S donné par val
    if save:
        [[plt.scatter(S[j][i][0], S[j][i][1], c=color[j]) for i in range(len(S[j]))] for j in range(k)]
        [plt.scatter(muuk[0], muuk[1], c='r', s=100) for muuk in muu]

        plt.title(f'Val = {Val}')
        plt.savefig(str(Val) + '.png')
        plt.close()

    # on retourne le partitionnement S trouver par l'algorithme des k-moyens
    return S
