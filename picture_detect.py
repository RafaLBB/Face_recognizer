import cv2

classificador = cv2.CascadeClassifier("/home/rafaela/Documents/Rafaela_VC/semana_prof_2021.1/cascades/haarcascade_frontalface_default.xml")

imagem = cv2.imread("/home/rafaela/Documents/Rafaela_VC/semana_prof_2021.1/pessoas/beatles.jpg")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(30, 30))

for(x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 255, 255), 2)
    
    '''cv2.imshow("Face", imagem)
    cv2.waitKey(1)'''
    

cv2.imshow("Face", imagem)
cv2.waitKey()
