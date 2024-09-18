import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

img = imageio.imread('paisaje1.jpg')/255
print(img.shape,img.dtype)
plt.figure()
plt.imshow(img)

#distintos canales RGB
plt.figure()
plt.imshow(img[:,:,0],) #canal R
plt.figure()
plt.imshow(img[:,:,1],) #canal G
plt.figure()
plt.imshow(img[:,:,2],) #canal B

#pasar de RGB a YIQ
img = imageio.imread('paisaje1.jpg')/255
plt.imshow(img)

def RGBtoYIQ(img):
    yiq = np.zeros(img.shape)
    yiq[:,:,0] = np.clip(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2],0,1) # Luminancia Y
    yiq[:,:,1] = np.clip(0.595*img[:,:,0] - 0.274*img[:,:,1] - 0.321*img[:,:,2],-0.5957, 0.5957) # In-phase I
    yiq[:,:,2] = np.clip(0.211*img[:,:,0] - 0.523*img[:,:,1] + 0.312*img[:,:,2],-0.5226,0.5226) # Quadrature Q
    return yiq

yiq = RGBtoYIQ(img)
#mostrar la imagen en YIQ en cada canal
plt.figure()
plt.imshow(yiq)
plt.figure()
plt.imshow(yiq[:,:,0]) # Y
plt.figure()
plt.imshow(yiq[:,:,1]) # I
plt.figure()
plt.imshow(yiq[:,:,2]) # Q

#punto4 modificar la imagen,obtener imagen mas oscura y aumentar saturacion a y b

def modificarYIQ(yiq,a,b):
    yiq[:,:,0] = np.clip(yiq[:,:,0]*a,0,1)
    yiq[:,:,1] = np.clip(yiq[:,:,1]*b,-0.5957,0.5957)
    yiq[:,:,2] = np.clip(yiq[:,:,2]*a,-0.5226,0.5226)
    return yiq

a = 0.5 #luminancia
b = 1.3 #saturacion
yiq2 = modificarYIQ(yiq, a, b)

plt.figure(4)
plt.imshow(yiq2)


#punto 7 YIQ a RGB
def YIQtoRGB(yiq):
    rgb = np.zeros(yiq.shape)
    rgb[:,:,0] = 1*yiq[:,:,0]+0.9663*yiq[:,:,1]+0.6210*yiq[:,:,2] # R
    rgb[:,:,1] = 1*yiq[:,:,0]-0.2721*yiq[:,:,1]-0.6474*yiq[:,:,2] # G
    rgb[:,:,2] = 1*yiq[:,:,0]-1.1070*yiq[:,:,1]+1.7046*yiq[:,:,2] # B
    return rgb

rgb2= YIQtoRGB(yiq2)
#mostrar imagen original y final
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(rgb2)