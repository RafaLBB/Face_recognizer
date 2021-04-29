import cv2

classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
amostra = 0
numeroAmostras = 15
id = input("Digite seu identificador")
largura, altura = 220, 165                          #Tamanho da foto que eu vou tirar


while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    #Já que ele conseguiu fazer a detecção de uma foto, ele entrará no comando for para a captura de fotos

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 255, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #Sempre que apertar a tecla 'q', o comando abaixo será  selecionado (Salvar ImagemFace)
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite("dataset/" + str(id) + "." + str(amostra) + ".jpg", imagemFace)
            print("[foto" + str(amostra) + "capturada com sucesso]")
            amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if amostra >= numeroAmostras + 1:
        break
print("Faces capturadas com sucesso")

cv2.imshow("Face", imagem)
#cv2.waitKey(1)
camera.release()
