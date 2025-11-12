import cv2
import numpy as np
import os
import time
from deepface import DeepFace
import mediapipe as mp
from ReconhecimentoYOLO import reconhecimento_yolo



def ReconhecerGiovanna():

    LEFT_EYE = [33, 133, 159, 145]
    RIGHT_EYE = [362, 263, 386, 374]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

    # Limpar terminal
    def clear_terminal():
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')


    # Tempo de verificação de sexo.
    tempo_s = 3600.0 
    #tempo de verificação para as emoções.
    tempo_emocao = 1550.0

    limitar_confianca = 50

    # Carrega o modelo LBPH (seu modelo treinado)
    modelo = cv2.face.LBPHFaceRecognizer_create()
    try:
        modelo.read("modelo.yml")
    except cv2.error:
        print("Aviso: 'modelo.yml' não encontrado. O reconhecimento facial LBPH não funcionará.")
        modelo = None

    # Carrega a base de rostos (Haar Cascade)
    rosto_base = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Lista de Pessoas possíveis 
    pasta_princ = "Pessoas"
    if os.path.exists(pasta_princ):
        nomes = [nome for nome in os.listdir(pasta_princ) if os.path.isdir(os.path.join(pasta_princ, nome))]
        nomes.sort()
    else:
        nomes = []

    # Armazena a última análise de genero
    cache_s = {} 
    # Armazena a ultima análise de emoçao
    cache_emocao = {}
    # Para controle de log no terminal
    log_deepface_timestamp = {} 

    # Abre a câmera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir camêra")
        exit()

    # Limpa o terminal na inicialização 
    clear_terminal()
    print(f"Sistema inicializado. A análise DeepFace só ocorrerá se o LBPH reconhecer a pessoa.")
    print(f"Gênero será checado a cada {tempo_s/60:.1f} minutos. Emoção a cada {tempo_emocao:.1f} segundos.")


    # Roda a camera e atualiza constantemente
    while True:
        ret, frame = cap.read()
        height, weight = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not ret:
            print("Não foi possível carregar sua câmera, verifique!")
            break
        
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = rosto_base.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        current_faces = {}
        agora = time.time()
        
        nova_analise_concluida = False
        
        for (x, y, w, h) in rostos:
            
            face_key = (x, y) 
            current_faces[face_key] = True
            face_frame = None
            if results.multi_face_landmarks:
                face_frame = results.multi_face_landmarks[0]
            else:
                continue
                

            # Retorna coordenadas dos olhos, multiplicadas pela largura e altura do frame
            def eye_region(indices):
                return np.array([(int(face_frame.landmark[i].x * weight), int(face_frame.landmark[i].y * height)) for i in indices])

            left = eye_region(LEFT_EYE)
            right = eye_region(RIGHT_EYE)

            # Máscara para olhos
            mask = np.zeros((height, weight), dtype=np.uint8)
            cv2.fillPoly(mask, [left], 255)
            cv2.fillPoly(mask, [right], 255)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = cv2.mean(gray, mask=mask)[0]

            # --- DETECÇÃO DE ÓCULOS ESCUROS ---
            if brightness < 20:
                eye_status, eye_color = "Oculos escuros", (0, 255, 0)
            else:
                eye_status, eye_color = "Sem Oculos", (0, 0, 255)

            # --- DETECÇÃO DE CHAPÉU ---
            # Pontos aproximados do topo da cabeça
            top_y = int(face_frame.landmark[10].y * height)
            crown_y = int(face_frame.landmark[152].y * height)
            x_coords = [int(p.x * weight) for p in face_frame.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            # Região acima da cabeça (aproximadamente 1.5x a altura da cabeça)
            y_top = max(0, top_y - (crown_y - top_y) * 2)
            roi_hat = frame[y_top:top_y, x_min:x_max]
            gray_hat = cv2.cvtColor(roi_hat, cv2.COLOR_BGR2GRAY)
            mean_brightness_hat = cv2.mean(gray_hat)[0]

            if mean_brightness_hat < 20:  # ajuste conforme iluminação
                hat_status, hat_color = "Possivel Chapéu", (0, 255, 0)
            else:
                hat_status, hat_color = "Sem chapeu", (0, 0, 255)

            # Textos
            cv2.putText(frame, eye_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)
            cv2.putText(frame, hat_status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, hat_color, 2)

            # --- Parte A: Reconhecimento Facial (LBPH) ---
            nome = "N/A (LBPH)" 
            confianca = 100.0
            cor = (0, 0, 255) 
            
            reconhecido = False
            if modelo and nomes:
                rosto_lbph = cinza[y:y+h, x:x+w]
                rosto_lbph = cv2.resize(rosto_lbph, (200, 200)) 
                id_previsto, confianca = modelo.predict(rosto_lbph)

                if confianca < limitar_confianca:
                    nome = nomes[id_previsto]
                    cor = (0, 255, 0)
                    reconhecido = True 
                else:
                    nome = "Não Reconhecido"
                    cor = (0, 0, 255)

            # Analise de emoção
            genero = "..." 
            emocao = "..."
            deepface_executado = False
            
            if reconhecido: 
                rosto_deepface = frame[y:y+h, x:x+w] 

                
                
                if face_key not in cache_s or (agora - cache_s[face_key]['timestamp'] >= tempo_s):
                    # Cache Gênero Inválido: Executa análise
                    try:
                        analise_gen = DeepFace.analyze(rosto_deepface, actions=['gender'], enforce_detection=False)[0] 
                        genero = analise_gen['dominant_gender']
                        cache_s[face_key] = {'gender': genero, 'timestamp': agora}
                    except Exception:
                        genero = "Erro Gen"
                        cache_s[face_key] = {'gender': genero, 'timestamp': agora}
                else:
                    # Cache Gênero Válido: Recupera
                    genero = cache_s[face_key]['gender']


                
                if face_key not in cache_emocao or (agora - cache_emocao[face_key]['timestamp'] >= tempo_emocao):
                    # Cache Emoção Inválido: Executa análise
                    try:
                        analise_emo = DeepFace.analyze(rosto_deepface, actions=['emotion'], enforce_detection=False)[0] 
                        emocao = analise_emo['dominant_emotion']
                        if emocao == 'neutral':
                            emocao = 'neutro'
                        elif emocao == 'happy':
                            emocao = 'feliz'
                        elif emocao == 'sad':
                            emocao = 'triste'
                        elif emocao == 'angry':
                            emocao = 'bravo'
                        elif emocao == 'fear':
                            emocao = 'com medo'
                        elif emocao == 'surprise':
                            emocao = 'surpreso'
                        elif emocao == 'disgust':
                            emocao = 'nojo'
                        
                        cache_emocao[face_key] = {'emotion': emocao, 'timestamp': agora}
                        
                        # Marca que a análise DINÂMICA (Emoção) foi feita
                        deepface_executado = True
                        nova_analise_concluida = True 
                    except Exception:
                        emocao = "Erro Emo"
                        cache_emocao[face_key] = {'emotion': emocao, 'timestamp': agora}
                else:
                    # Cache Emoção Válido: Recupera
                    emocao = cache_emocao[face_key]['emotion']

            else:
                genero = "N/A"
                emocao = "N/A"


    
            if deepface_executado:
                mensagem_log = f"--- NOVA ANÁLISE DE EMOÇÃO CONCLUÍDA ---\n"
                mensagem_log += f"  > Nome LBPH: {nome} (Conf: {confianca:.1f})\n"
                mensagem_log += f"  > DeepFace: Gênero (Fixado): {genero}, Emoção (Atualizada): {emocao}\n"
                mensagem_log += f"--- A próxima análise de EMOÇÃO será em {tempo_emocao:.1f} segundos ---\n"
                
                # 2. Registra o timestamp do log
                log_deepface_timestamp[face_key] = agora
                
                # O log será impresso FORA deste 'for'


            
            # Linha 1: Nome (LBPH) e Confiança
            texto_nome = f"{nome} ({confianca:.1f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, texto_nome, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            
            if reconhecido:
                # Gênero - usa a variável 'genero' (do cache estático)
                texto_genero = f"Genero: {genero}"
                cv2.putText(frame, texto_genero, (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Emoção - usa a variável 'emocao' (do cache dinâmico)
                texto_emocao = f"Emocao: {emocao}"
                cv2.putText(frame, texto_emocao, (x, y+h+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "DeepFace inativo (Nao Reconhecido)", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        
        # 2. CONTROLE CENTRAL DE LOG E LIMPEZA DE TELA
        if nova_analise_concluida:
            clear_terminal() 
            print(mensagem_log) 

        # 3. Limpa os caches de rostos que não estão mais no frame 
        keys_to_delete = [key for key in cache_s if key not in current_faces]
        for key in keys_to_delete:
            del cache_s[key]
            if key in cache_emocao:
                del cache_emocao[key]
            if key in log_deepface_timestamp:
                del log_deepface_timestamp[key]
                

        # Mostra o vídeo
        cv2.imshow("Reconhecimento e Analise Facial (pressione 'q' para sair)", frame)

        # Sai com 'q'
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        if tecla == ord('r'):
            cv2.destroyAllWindows()
            reconhecimento_yolo(cap)
            

    # Finaliza a camera
    cap.release()
    cv2.destroyAllWindows()