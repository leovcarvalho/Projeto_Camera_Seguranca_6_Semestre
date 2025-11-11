import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import time



def reconhecimento_yolo(videoSource):

    # Dicionário de tradução
    traducao_classes = {
        'person': 'pessoa',
        'knife': 'faca',
        'backpack': 'mochila',
        'handbag': 'bolsa',
        'glasses': 'oculos',
        'sunglasses': 'oculos escuros',
        'hat': 'bone',
        'gun': 'arma',
        'cup': 'copo',
        'mask': 'mascara',
        'bottle': 'garrafa',
        'suitcase': 'mala',
        'chair': 'cadeira'
    }

    # Lista de objetos monitorados
    objetos = ['pessoa', 'faca', 'mochila', 'oculos', 'bone', 'arma', 'copo', 'mascara', 'bolsa', 'garrafa', 'mala', 'cadeira']

    # Caminho da pasta de gravação
    caminho_gravacao = 'Gravacoes_YOLO'
    os.makedirs(caminho_gravacao, exist_ok=True)

    # 🧠 Carrega múltiplos modelos
    modelos = [
        YOLO('yolo11n.pt'),
        YOLO("yolo11s.pt")
    ]

    # Exibe classes do primeiro modelo
    print("Classes disponíveis no primeiro modelo YOLO:")
    print(modelos[0].names)
    print("-" * 50)


    fps_cap = videoSource.get(cv2.CAP_PROP_FPS) or 30.0
    gravando = False
    monitorando = False
    saida_video = None
    tempo_ultima_detecção = None
    intervalo_gravacao = 5
    objetos_detectados_total = set()

    print("Pressione 'S' para iniciar o monitoramento e gravação automática.")
    print("Pressione 'Esc' para sair.")
    print(f"Objetos monitorados: {', '.join(objetos)}")

    while True:
        sucesso, frame = videoSource.read()
        if not sucesso:
            break

        if monitorando:
            deteccao_na_iteracao_atual = False
            annotated_frame = frame.copy()

            # 🔹 Executa todos os modelos no mesmo frame
            for model in modelos:
                resultados = model(frame, conf=0.3, imgsz=640, verbose=False)
                annotated_frame = resultados[0].plot()

                for r in resultados:
                    for box in r.boxes:
                        classe_nome = model.names[int(box.cls)]
                        confianca = box.conf.item()
                        classe_traduzida = traducao_classes.get(classe_nome.lower(), classe_nome.lower())

                        if classe_traduzida in objetos:
                            deteccao_na_iteracao_atual = True
                            objetos_detectados_total.add(classe_traduzida)

            # 🔸 Gravação automática
            if deteccao_na_iteracao_atual and not gravando:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = os.path.join(caminho_gravacao, f"gravação_{timestamp}.mp4")

                altura, largura, _ = annotated_frame.shape
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                saida_video = cv2.VideoWriter(nome_arquivo, codec, fps_cap, (largura, altura))

                print(f"🎥 Iniciando gravação: {nome_arquivo}")
                gravando = True
                tempo_ultima_detecção = datetime.now()

            if gravando:
                if saida_video.isOpened():
                    saida_video.write(annotated_frame)
                if deteccao_na_iteracao_atual:
                    tempo_ultima_detecção = datetime.now()
                elif (datetime.now() - tempo_ultima_detecção).total_seconds() > intervalo_gravacao:
                    print("⏹ Parando gravação - sem detecção recente.")
                    gravando = False
                    saida_video.release()
                    saida_video = None

            cv2.putText(annotated_frame,
                        "GRAVANDO..." if gravando else "Monitorando...",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255) if gravando else (0, 255, 0),
                        3)
            cv2.imshow("Monitoramento de Objetos", annotated_frame)
            time.sleep(0.09)  # Pequena pausa para melhorar a performance
        else:
            cv2.putText(frame, "Pressione 'S' para iniciar", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Monitoramento de Objetos", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            resultado = print("Programa finalizado pelo usuário.")
            if len(objetos_detectados_total) > 0:
                diaAtual = datetime.now().strftime("%Y%m%d_%H%M%S")
                caminho_arquivo = os.path.join('Gravacoes_YOLO/Relatorio_Da_Deteccao', f"detecção_{diaAtual}.txt")
                with open(caminho_arquivo, "w", encoding="utf-8") as f:
                    for obj in objetos_detectados_total:
                        f.write(f"{obj}\n")
            cv2.destroyAllWindows()
            return resultado
        elif tecla == ord('s'):
            print(objetos_detectados_total)
            for obj in objetos_detectados_total:
                print(f" - {obj}")
            monitorando = not monitorando
            if monitorando:
                # objetos_detectados_total.clear()
                print("▶️ Monitoramento iniciado.")
            else:
                print("⏸️ Monitoramento pausado.")
                if gravando:
                    gravando = False
                    saida_video.release()
                    saida_video = None
        elif tecla == ord('q'):
            break

    videoSource.release()
    if saida_video:
        saida_video.release()
    cv2.destroyAllWindows()