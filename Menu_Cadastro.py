from BancoImagens import BancoImagens

def menu_cadastro():
    while True:
        print("=" * 90, end="\n\n")
        print("MENU DE CADASTRO", end="\n\n")
        print("=" * 90, end="\n\n")
        print("Escolha oque deseja realizar agora:", end="\n\n")
        print("1º - Listar Pessoas Cadastradas.", end="\n\n")
        print("2º - Adicionar Nova Pessoa.", end="\n\n")
        print("3º - Sobreescrever uma pasta de pessoa existente.", end="\n\n")
        print("4º - Adicionar novas fotos a uma pessoa existente.", end="\n\n")
        print("5º - Voltar ao menu principal.", end="\n\n")

        escolha = input("Digite aqui qual opção deseja: ")

        print("")
        print("=" * 90, end="\n\n")
        
        if escolha == "1":
            print("Segue a lista de todas as pessoas cadastradas:", end="\n\n")
            BancoImagens("listar")
            
        elif escolha == "2":
            print("Iniciando programa para cadastrar uma nova pessoa . . . . .", end="\n\n")
            BancoImagens("cadastrar")
            break
        
        elif escolha == "3":
            print("Iniciando programa para reescrever uma pessoa existente . . . . .", end="\n\n")
            BancoImagens("sobreescrever")
            break
        
        elif escolha == "4":
            print("Iniciando programa para adicionar mais imagens a uma pessoa existente", end="\n\n")
            BancoImagens("complementar")
            break
        
        elif escolha == "5":
            print("Retornando ao menu principal . . . .", end="\n")
            break
        
        else:
            print("Essa opção não é valida pedimos gentilmente que escolha uma das opções acima válida ou finalize o programa!!!", end="\n\n")