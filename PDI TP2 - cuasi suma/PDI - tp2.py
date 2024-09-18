import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
# import tkinter

imgA = imageio.imread('paisaje-A.jpg')/255
imgB = imageio.imread('paisaje-B.jpg')/255

def sumaclam(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =np.clip(imgA[:,:,0]+imgB[:,:,0],0,1)
    imgC[:,:,1] =np.clip(imgA[:,:,1]+imgB[:,:,1],0,1)
    imgC[:,:,2] =np.clip(imgA[:,:,2]+imgB[:,:,2],0,1)
    return imgC
    
imgclam = sumaclam(imgA, imgB)#llamo a la funcion sumapro

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgclam)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Suma RGB clam')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

def sumapro(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =(imgA[:,:,0]+imgB[:,:,0])/2
    imgC[:,:,1] =(imgA[:,:,1]+imgB[:,:,1])/2
    imgC[:,:,2] =(imgA[:,:,2]+imgB[:,:,2])/2
    return imgC

imgcpro = sumapro(imgA, imgB)#llamo a la funcion sumapro

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgcpro)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Suma RGB pro')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

####        RESTA         ############
def restaclam(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =np.clip(imgA[:,:,0]-imgB[:,:,0],0,1)
    imgC[:,:,1] =np.clip(imgA[:,:,1]-imgB[:,:,1],0,1)
    imgC[:,:,2] =np.clip(imgA[:,:,2]-imgB[:,:,2],0,1)
    return imgC
    
imgclamR = restaclam(imgA, imgB)

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgclamR)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('resta RGB clam')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

def restapro(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =(imgA[:,:,0]-imgB[:,:,0])/2
    imgC[:,:,1] =(imgA[:,:,1]-imgB[:,:,1])/2
    imgC[:,:,2] =(imgA[:,:,2]-imgB[:,:,2])/2
    return imgC

imgcproR = restapro(imgA, imgB)

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgcproR)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('resta RGB pro')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

def RGBtoYIQ(img):
    yiq = np.zeros(img.shape)
    yiq[:,:,0] = np.clip(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2],0,1) # Luminancia Y
    yiq[:,:,1] = np.clip(0.595*img[:,:,0] - 0.274*img[:,:,1] - 0.321*img[:,:,2],-0.5957, 0.5957) # In-phase I
    yiq[:,:,2] = np.clip(0.211*img[:,:,0] - 0.523*img[:,:,1] + 0.312*img[:,:,2],-0.5226,0.5226) # Quadrature Q
    return yiq
################ YIQ suma clampeada ###################
# YC := YA + YB; If YC > 1 then YC:=1; 
# IC := (YA * IA + YB * IB) / (YA + YB) ;
# QC := (YA * QA + YB * QB) / (YA + YB) ; 
def sumaYIQclam(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =np.clip(imgA[:,:,0]+imgB[:,:,0],0,1)
    imgC[:,:,1] =((imgA[:,:,0])*imgA[:,:,1] + imgB[:,:,0]*imgB[:,:,1])/(imgA[:,:,0]+imgB[:,:,0])
    imgC[:,:,2] =((imgA[:,:,0])*imgA[:,:,2] + imgB[:,:,0]*imgB[:,:,2])/(imgA[:,:,0]+imgB[:,:,0])
    return imgC
################ YIQ suma promediada ###################
# YC := (YA + YB) / 2;
def sumaYIQpro(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =(imgA[:,:,0]+imgB[:,:,0])/2
    imgC[:,:,1] =((imgA[:,:,0])*imgA[:,:,1] + imgB[:,:,0]*imgB[:,:,1])/(imgA[:,:,0]+imgB[:,:,0])
    imgC[:,:,2] =((imgA[:,:,0])*imgA[:,:,2] + imgB[:,:,0]*imgB[:,:,2])/(imgA[:,:,0]+imgB[:,:,0])
    return imgC

yiqclam = sumaYIQclam(RGBtoYIQ(imgA), RGBtoYIQ(imgB))
yiqpro = sumaYIQpro(RGBtoYIQ(imgA), RGBtoYIQ(imgB))

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(yiqclam)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Suma YIQ clam')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()
fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(yiqpro)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Suma YIQ prom')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

# MULTIPLICACION Y DIVISION DE IMAGENES imgA imgB
imgMult = np.multiply(imgA,imgB)
imgDiv = np.divide(imgA,imgB)

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgMult)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Multiplicacion RGB')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgDiv)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('Division RGB')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()


#    RESTA ABSOLUTA

def restaAbs(imgA,imgB):
    imgC = np.zeros(imgA.shape)
    imgC[:,:,0] =abs(imgA[:,:,0]-imgB[:,:,0])
    imgC[:,:,1] =abs(imgA[:,:,1]-imgB[:,:,1])
    imgC[:,:,2] =abs(imgA[:,:,2]-imgB[:,:,2])
    return imgC

imgAbs = restaAbs(imgA, imgB)

fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(imgAbs)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('resta RGB pro')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

# IF LIGTHER
def sumaYIQligther(imgA, imgB):
    # Crear una imagen de salida con la misma forma que imgA
    C = np.zeros(imgA.shape)
    
    # Usar np.where para seleccionar el valor más oscuro de cada pixel
    mask = imgA[:, :, 0] > imgB[:, :, 0]
    
    # Si imgA es más oscuro, tomar los valores de imgA, si no, tomar de imgB
    C[:, :, 0] = np.where(mask, imgA[:, :, 0], imgB[:, :, 0])
    C[:, :, 1] = np.where(mask, imgA[:, :, 1], imgB[:, :, 1])
    C[:, :, 2] = np.where(mask, imgA[:, :, 2], imgB[:, :, 2])

    return C


yiqligther = sumaYIQligther(RGBtoYIQ(imgA), RGBtoYIQ(imgB))
fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(yiqligther)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('If Ligther')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()

# IF DARKER
# def sumaYIQdarker(imgA,imgB):
#     C = np.zeros(imgA.shape)
#     if imgA[:,:,0] < imgB[:,:,0]:
#         C[:,:,0] = imgA[:,:,0]
#         C[:,:,1] = imgA[:,:,1]
#         C[:,:,2] = imgA[:,:,2]
#     else:
#         C[:,:,0] = imgB[:,:,1]
#         C[:,:,1] = imgB[:,:,1]
#         C[:,:,2] = imgB[:,:,2]
#     return C
# El error que estás viendo ocurre porque en NumPy no puedes 
# comparar directamente arrays enteros con operadores lógicos como if. 
# En su lugar, necesitas realizar una comparación elemento por elemento 
# y luego aplicar la lógica que deseas utilizando funciones de NumPy, 
# como np.where.
# Puedes usar np.where para aplicar la comparación a cada pixel de las 
# imágenes imgA e imgB. Aquí tienes una versión corregida de la función:

def sumaYIQdarker(imgA, imgB):
    # Crear una imagen de salida con la misma forma que imgA
    C = np.zeros(imgA.shape)
    
    # Usar np.where para seleccionar el valor más oscuro de cada pixel
    mask = imgA[:, :, 0] < imgB[:, :, 0]
    
    # Si imgA es más oscuro, tomar los valores de imgA, si no, tomar de imgB
    C[:, :, 0] = np.where(mask, imgA[:, :, 0], imgB[:, :, 0])
    C[:, :, 1] = np.where(mask, imgA[:, :, 1], imgB[:, :, 1])
    C[:, :, 2] = np.where(mask, imgA[:, :, 2], imgB[:, :, 2])
    return C

yiqdarker = sumaYIQdarker(RGBtoYIQ(imgA), RGBtoYIQ(imgB))
fig, axs = plt.subplots(1, 3) # nº filas, nº columnas
axs[0].imshow(imgA)# fila 1, columna 1
axs[0].axis('off') # Desactivar los ejes (reglas y números)
axs[0].set_title('Imagen A')
axs[1].imshow(imgB)# fila 1, columna 2
axs[1].axis('off') # Desactivar los ejes (reglas y números)
axs[1].set_title('Imagen B')
axs[2].imshow(yiqdarker)# fila 1, columna 2
axs[2].axis('off') # Desactivar los ejes (reglas y números)
axs[2].set_title('If Darker')
# Mostrar todo el conjunto de subplots
plt.tight_layout()# Ajustar el espacio entre subplots
plt.show()














