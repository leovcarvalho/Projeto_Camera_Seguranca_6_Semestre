import cv2
import os
import numpy as np

PASTA_PESSOAS = "Pessoas"
ARQUIVO_MODELO = "modelo.yml"
CLASSIFICADOR_ROSTO = "haarcascade_frontalface_default.xml"

def treinoModelo():
    print("\n🧠 Iniciando treinamento do modelo facial...")

    # Inicializa o reconhecedor LBPH
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CLASSIFICADOR_ROSTO)

    faces = []
    ids = []
    nomes = []

    # Percorre as pastas dentro de "Pessoas"
    for id_pessoa, nome_pessoa in enumerate(sorted(os.listdir(PASTA_PESSOAS))):
        caminho_pessoa = os.path.join(PASTA_PESSOAS, nome_pessoa)
        if not os.path.isdir(caminho_pessoa):
            continue

        for arquivo in os.listdir(caminho_pessoa):
            if arquivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                caminho_img = os.path.join(caminho_pessoa, arquivo)
                img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Imagem inválida: {caminho_img}")
                    continue

                # Detecta rostos dentro da imagem
                rostos = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in rostos:
                    rosto = cv2.resize(img[y:y+h, x:x+w], (200, 200))
                    faces.append(rosto)
                    ids.append(id_pessoa)
                    nomes.append(nome_pessoa)

    if len(faces) == 0: 
        print("❌ Nenhuma imagem de rosto encontrada! Cadastre pessoas primeiro.")
        return

    # Treina o modelo com os rostos coletados
    print(f"Treinando com {len(faces)} rostos...")
    reconhecedor.train(faces, np.array(ids))
    reconhecedor.write(ARQUIVO_MODELO)

    # Salva os nomes associados aos IDs
    with open("nomes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(nomes))

    print(f"\n✅ Treinamento concluído com sucesso!")
    print(f"📁 Modelo salvo em: {ARQUIVO_MODELO}")
    print(f"👥 Pessoas reconhecidas: {', '.join(nomes)}")