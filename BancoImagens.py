import cv2
import os
import numpy as np
import time
from Treino_Modelo import treinoModelo


# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
PASTA_PESSOAS = "Pessoas"
CLASSIFICADOR_ROSTO = "haarcascade_frontalface_default.xml"
QUANTIDADE_IMAGENS = 40

os.makedirs(PASTA_PESSOAS, exist_ok=True)

# ==========================
# FUNÇÃO: CADASTRAR NOVA PESSOA
# ==========================
def cadastrar_novas_pessoa():
    while True:
            
        nome = input("Digite o nome da pessoa que deseja cadastrar: ").strip()
        
        # Validação se nome está em branco
        if not nome:
            print("Nome inválido! \n\n")
            
            escolha = input("Deseja digitar o nome novamente se sim digite(S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
            if escolha == "s":
                print()
                continue
            else:
                print()
                return
        
        pasta_pessoa = os.path.join(PASTA_PESSOAS, nome)
        
        # Validação se nome já existe
        if os.path.exists(pasta_pessoa):
            print(f"O nome '{nome}' já existe!")
            escolha = input("Caso desejar digitar o nome novamete digite (S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
        
            # Retorna ao começo do código
            if escolha == "s":
                print()
                continue
            
            # Retorna ao menu de cadastro
            else:
                print("Retornando ao menu de cadastro . . .", end="\n\n")
                return
            
        # Caso o nome seja válido
        os.makedirs(pasta_pessoa)
        break

    # ==========================
    # INÍCIO DA CAPTURA DE IMAGENS
    # ==========================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera!", end="\n\n\n")
        return

    detector = cv2.CascadeClassifier(CLASSIFICADOR_ROSTO)
    contador = 0

    print("\n Iniciando a captura de imagens . . .")
    print("Pressione 'q' para encerrar o cadastro caso desejar interromper o processo.")
    print("Olhe para a câmera em diferentes ângulos e expressões.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            contador += 1
            caminho_img = os.path.join(pasta_pessoa, f"{contador}.jpg")
            cv2.imwrite(caminho_img, rosto)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Foto {contador}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            time.sleep(0.3)

        cv2.imshow("Cadastro de Pessoa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or contador >= QUANTIDADE_IMAGENS:
            break

    print(f"\n✅ {contador} fotos salvas em {pasta_pessoa}")
        
    cap.release()
    cv2.destroyAllWindows()
    print()
    treinoModelo()

# ============================================
# FUNÇÃO: ADICIONAR MAIS IMAGENS A UMA PESSOA
# ============================================
def ComplementarBancoImgs():
    while True:
            
        nome = input("Digite o nome da pessoa que deseja cadastrar: ").strip()
        
        # Validação se nome está em branco
        if not nome:
            print("Nome inválido!")
            print()
            escolha = input("Caso desejar digitar o nome novamete digite (S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
            if escolha == "s":
                print()
                continue
            else:
                print()
                return
        
        pasta_pessoa = os.path.join(PASTA_PESSOAS, nome)
        
        # Validando caso nome não existir
        if not os.path.exists(pasta_pessoa):
            print(f"O nome '{nome}' não existe em nosso banco de dados!")
            print()
            escolha = input("Caso desejar digitar o nome novamete digite (S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
            if escolha == "s":
                print()
                continue
            else:
                print()
                return
        
        # Caso seja válido
        break

    # ==========================
    # INÍCIO DA CAPTURA DE IMAGENS
    # ==========================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera!", end="\n\n\n")
        return

    detector = cv2.CascadeClassifier(CLASSIFICADOR_ROSTO)
    contador = os.listdir(pasta_pessoa)
    contador = len(contador)
    maximo = contador + 15

    print("\n Iniciando a captura de imagens . . .")
    print("Pressione 'q' para encerrar o cadastro caso desejar interromper o processo.")
    print("Olhe para a câmera em diferentes ângulos e expressões.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            contador += 1
            caminho_img = os.path.join(pasta_pessoa, f"{contador}.jpg")
            cv2.imwrite(caminho_img, rosto)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Foto {contador}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            time.sleep(0.3)

        cv2.imshow("Cadastro de Pessoa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or contador >= maximo:
            break

    print(f"\n✅ {contador} fotos salvas em {pasta_pessoa}")
        
    cap.release()
    cv2.destroyAllWindows()
    print()
    treinoModelo()

# ============================================
# FUNÇÃO: SOBREESCREVER UMA PASTA DO BANCO
# ============================================
def SobreescreverBancoImgs():
    while True:
            
        option = os.listdir(PASTA_PESSOAS)
        print("Pessoas cadastradas:")
        for pessoa in option:
            print(f"- {pessoa}")
        print()
        nome = input("Digite o nome da pessoa que deseja cadastrar: ").strip()
        
        # Validação se nome está em branco
        if not nome:
            print("Nome inválido!")
            print()
            escolha = input("Caso desejar digitar o nome novamete digite (S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
            if escolha == "s":
                print()
                continue
            else:
                print()
                return
        
        pasta_pessoa = os.path.join(PASTA_PESSOAS, nome)
        
        # Validando caso nome não existir
        if not os.path.exists(pasta_pessoa):
            print(f"O nome '{nome}' não existe em nosso banco de dados!")
            print()
            escolha = input("Caso desejar digitar o nome novamete digite (S) ou caso desejar voltar ao menu de cadastro digite qualquer outra tecla:").strip().lower()
            if escolha == "s":
                print()
                continue
            else:
                print()
                return
        
        # Caso seja válido
        break

    # ==========================
    # INÍCIO DA CAPTURA DE IMAGENS
    # ==========================
    
    # Limpando imagens que estavam na pasta antigamente
    for arquivo in os.listdir(pasta_pessoa):
        caminho = os.path.join(pasta_pessoa, arquivo)
        if os.path.isfile(caminho):
            os.remove(caminho)
    print(f"\n Todas as imagens antigas de {nome} foram apagadas! \n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera!", end="\n\n\n")
        return

    detector = cv2.CascadeClassifier(CLASSIFICADOR_ROSTO)
    contador = 0

    print("\n Iniciando a captura de imagens . . .")
    print("Pressione 'q' para encerrar o cadastro caso desejar interromper o processo.")
    print("Olhe para a câmera em diferentes ângulos e expressões.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            contador += 1
            caminho_img = os.path.join(pasta_pessoa, f"{contador}.jpg")
            cv2.imwrite(caminho_img, rosto)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Foto {contador}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            time.sleep(0.3)

        cv2.imshow("Cadastro de Pessoa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or contador >= QUANTIDADE_IMAGENS:
            break

    print(f"\n✅ {contador} fotos salvas em {pasta_pessoa}")
                
    cap.release()
    cv2.destroyAllWindows()
    treinoModelo()

def lista_pessoas():

    
    # Checa se a pasta realmente existe
    os.makedirs(PASTA_PESSOAS, exist_ok=True)
    
    pessoas = [p for p in os.listdir(PASTA_PESSOAS) if os.path.isdir(os.path.join(PASTA_PESSOAS, p))]
    
    if not pessoas:
        print("Nenhuma pessoa ainda foi cadastrada!", end="\n\n")
    else:
        print("=" * 90)
        print("Lista de pessoas cadastradas:")
        print("=" * 90, end="\n\n")
        for i, pessoa in enumerate(sorted(pessoas), start=1):
            print(f"{i:02d} - {pessoa}")
        print("")

def BancoImagens(action):
    if action == "cadastrar":
        cadastrar_novas_pessoa()
    elif action == "sobreescrever":
        SobreescreverBancoImgs()
    elif action == "complementar":
        ComplementarBancoImgs()
    elif action == "listar":
        lista_pessoas()