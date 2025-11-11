from Menu_Cadastro import menu_cadastro
from Treino_Modelo import treinoModelo
from ReconhecerGiovanna import ReconhecerGiovanna

def menu_principal():
    while True:    
        print("")
        print("=" * 90, end="\n\n")
        print("MENU PRINCIPAL", end="\n\n")
        print("=" * 90, end="\n\n")
        print("Escolha oque deseja realizar agora:", end="\n\n")
        print("1º - Menu de cadastro.", end="\n\n")
        print("2º - Treinar modelo novamente (sempre executar antes de ir para reconhecimento!", end="\n\n")
        print("3º - Iniciar sistema de reconhecimento e procura!", end="\n\n")
        print("4º - Finalizar programa!", end="\n\n")

        escolha = input("Digite aqui qual opção deseja: ")

        print("")
        print("=" * 90, end="\n\n")
        
        if escolha == "1":
            print("Abrindo menu de cadastro . . . . .", end="\n\n")
            menu_cadastro()
        elif escolha == "2":
            print("Realizando o treinamento do seu novo projeto . . . . .", end="\n\n")
            treinoModelo()
        elif escolha == "3":
            print("Inciiando programa de reconhecimento . . . . .", end="\n\n")
            ReconhecerGiovanna()
        elif escolha == "4":
            print("Programa finalizado com sucesso!!", end="\n\n")
            break
        else:
            print("Essa opção não é valida pedimos gentilmente que escolha uma das opções acima válida ou finalize o programa!!!", end="\n")
            

if __name__ == "__main__":
    menu_principal()
    
    
    
    