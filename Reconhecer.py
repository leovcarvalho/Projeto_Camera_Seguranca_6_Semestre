import cv2
import os
import time
import deepface as DF

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
PASTA_PESSOAS = "Pessoas"
ARQUIVO_MODELO = "modelo.yml"
CLASSIFICADOR_ROSTO = "haarcascade_frontalface_default.xml"
LIMIAR_CONFIANCA = 45  # Quanto menor, mais rigoroso

# ==========================
# FUNÇÃO: RECONHECIMENTO EM TEMPO REAL
# ==========================
def reconhecer():
    if not os.path.exists(ARQUIVO_MODELO):
        print("Modelo não encontrado! Treine o modelo primeiro.")
        return

    tempo_mensagem = 5

    # Cria o reconhecedor
    modelo = cv2.face.LBPHFaceRecognizer_create()

    # Tenta ler o modelo treinado
    try:
        modelo.read(ARQUIVO_MODELO)  # alternativa: modelo.load(ARQUIVO_MODELO)
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        print("Verifique se o arquivo de modelo está correto e foi gerado pelo treino.")
        return

    # Carrega o classificador Haar
    rosto_base = cv2.CascadeClassifier(CLASSIFICADOR_ROSTO)

    # Lista de Pessoas (mapeamento de índices)
    pasta_princ = PASTA_PESSOAS
    nomes = [nome for nome in os.listdir(pasta_princ) if os.path.isdir(os.path.join(pasta_princ, nome))]
    nomes.sort()
    print("Pessoas que estão no banco de dados:", nomes)

    # Abre a câmera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro ao abrir câmera")
        return

    mensagens_ativas = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível carregar a câmera. Saindo...")
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = rosto_base.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            try:
                id_previsto, confianca = modelo.predict(rosto)
            except cv2.error as err:
                print("Erro no predict:", err)
                continue

            # Verifica se id_previsto está dentro do range de nomes
            nome = None
            if 0 <= id_previsto < len(nomes) and confianca < LIMIAR_CONFIANCA:
                nome = nomes[id_previsto]
                mensagem = f"Reconhecido como: {nome}"
                cor = (0, 255, 0)
            else:
                mensagem = "Rosto não reconhecido."
                cor = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, f"{mensagem} ({confianca:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            if nome not in mensagens_ativas:
                print(mensagem, f" - Confiança: {confianca:.1f}")
                mensagens_ativas[nome] = time.time()

        # Remove mensagens antigas
        for chave in list(mensagens_ativas.keys()):
            if time.time() - mensagens_ativas[chave] > tempo_mensagem:
                del mensagens_ativas[chave]

        cv2.imshow("Reconhecimento Facial (pressione 'q' para sair)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

    # FINALIZAÇÃO (fora do loop)
    cap.release()
    cv2.destroyAllWindows()