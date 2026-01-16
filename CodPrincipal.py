import eel
import tkinter as tk
from tkinter import filedialog
#todas as funções de back-end feitas aqui
'''
Para esse programa quero integrar as 3 analises em um só grafico.
'''
import matplotlib.pyplot as plt
import numpy as np
import lmfit as fit
import pandas as pd
import os

x = [0,1,2,3,4,5]
y = [0,1,4,9,16,25]
#Funções auxiliares
def linear(x,a,b):
    return a*x+b

def Eir(E,i,r):
        return E-0.95*i*r

def abrir_diretorio(diretorio):
    arquivos = os.listdir(diretorio)
    return arquivos

def converter_para_RHE(E,ph = 13.65):
    return E + (0.059 * ph) + 0.1976

def eq_para_o_fitting(Rs,Qdl,Qad,Rp,Rad):
        return  Rs*(Qdl*Rp)*(Qad*Rad)

def eq_polinomial_4ordem(x, a, res1, res2, cap1, cap2):
    """Polinômio de 4ª ordem: a + b*x + c*x² + d*x³ + e*x⁴"""
    return a + res1*x + res2*x**2 + cap1*x**3 + cap2*x**4

#parte 1 - LSV
def LSV(diretorio,nome_da_amostra):
    
    aquivos = abrir_diretorio(diretorio)
    for nome_arquivo in aquivos:
        if nome_arquivo.endswith(".txt"):
            arquivo = os.path.join(diretorio, nome_arquivo)
            conteudo = pd.read_csv(arquivo,sep=';',decimal = ",")

            #Parte 1
            J = np.array(conteudo["J (mA/cm²)"])
            EIR = np.array(conteudo["E-iR vs. RHE (V)"])

            plt.figure(figsize=(10,10))
            plt.plot(EIR,J,label=nome_da_amostra)
            plt.xlabel("EIR vs. RHE (V)")
            plt.ylabel("J (mA/cm²)")
            plt.legend()
            plt.show()

            #Parte 2
            alvos = [10,50,100,200,300]
            idx_prox = []
            for alvo in alvos:
                idx = np.abs(J - alvo).argmin()
                idx_prox.append(idx)

            CP = np.array(conteudo["Corrected potential (V)"])
            CP_alvo = []
            for idxs in idx_prox:
                CP_alvo.append(CP[idxs])
            CP_alvo = np.array(CP_alvo)

            eta = (converter_para_RHE(CP_alvo) - 1.23)*1000#v
            plt.figure(figsize=(10,10))
            plt.bar(alvos,eta,width=20,label=nome_da_amostra)
            plt.xlabel("miliAmpere")
            plt.ylabel(r"$\eta$")
            plt.legend()
            plt.show()

            #parte 3 - Fazer o fitting ainda

            J_log = np.log10(J)

            plt.figure(figsize=(10,10))
            plt.plot(J_log, EIR, label=nome_da_amostra)
            plt.xlim(-0.50,J_log[-1])
            plt.ylim(1.35,EIR[-1])
            plt.xlabel(r"$\log_{10} J\;(\mathrm{mA/cm^2})$")
            plt.ylabel("EIR vs. RHE (V)")
            plt.legend()

            print("Selecione DOIS pontos no gráfico (intervalo de Tafel)")

            pontos = plt.ginput(2)
            plt.close()

            if len(pontos) != 2:
                raise RuntimeError(
                    f"Esperado 2 pontos, mas foram selecionados {len(pontos)}."
                )

            ponto1, ponto2 = pontos


            x1, x2 = ponto1[0], ponto2[0]

            # Ordenar para evitar inversão
            x_min, x_max = sorted([x1, x2])

            # Criar máscara para selecionar o intervalo
            mask = (J_log >= x_min) & (J_log <= x_max)

            # Dados filtrados
            EIR_intervalo = EIR[mask]
            J_intervalo = J_log[mask]
            print("Intervalo selecionado:")
            print("x_min =", x_min)
            print("x_max =", x_max)
            print("Número de pontos selecionados:", len(J_intervalo))
            plt.figure(figsize=(10,10))
            plt.plot(J_intervalo,EIR_intervalo,'ro',label='Pontos Selecionados')
            plt.xlabel(r"$Log_{10}$ J (mA/cm²)")
            plt.ylabel("EIR vs. RHE (V)")
            plt.legend()
            plt.show()

            modelo = fit.Model(linear)
            params = modelo.make_params(a=0.1,b=1)
            resultado = modelo.fit(EIR_intervalo,params,x=J_intervalo)
            print(resultado.fit_report())
            plt.figure(figsize=(10,10))
            plt.plot(J_log,EIR,'b-',label=nome_da_amostra,lw=3)
            plt.plot(J_intervalo,resultado.best_fit,'r-',label='Fitting Linear',lw=2)
            plt.xlim(-0.50,J_log[-1])
            plt.ylim(1.35,EIR[-1])
            plt.xlabel(r"$Log_{10}$ J (mA/cm²)")
            plt.ylabel("EIR vs. RHE (V)")
            plt.legend()
            plt.show()

    return J, J_log, EIR, J_intervalo, EIR_intervalo, CP, eta, resultado, alvos

#Parte 2 - EIS
def EIS(diretorio,nome_da_amostra):
    aquivos = abrir_diretorio(diretorio)
    Z2linhas = []
    Zlinhas = []
    Rss = []
    resultados = []
    for nome_arquivo in aquivos:
        if nome_arquivo.endswith(".txt"):
            arquivo = os.path.join(diretorio, nome_arquivo)

            conteudo = pd.read_csv(arquivo,sep=";",decimal = ",")
            #Parte 1
            Zlinha = np.array(conteudo["Z' (Ω)"])
            Z2linha = np.array(conteudo["-Z'' (Ω)"])

            # Opção 1: Usar [0][0] para extrair o valor escalar
            idxminimo = np.where(Z2linha == np.min(Z2linha))[0][0]
            print("Índice do mínimo de Z''", idxminimo)

            Rs = max(Zlinha[:idxminimo])

            # Fatiamento CORRETO
            Zlinha = Zlinha[idxminimo+5:]
            Z2linha = Z2linha[idxminimo+5:]

            modelo = fit.Model(eq_polinomial_4ordem)
            params = modelo.make_params(a=1, res1=0.1, res2=0.001, cap1=0.0001, cap2=0.00001)
            resultado = modelo.fit(Z2linha, params, x=Zlinha)
            print(resultado.fit_report())
            print("Rs estimado:", Rs)
            plt.figure(figsize=(10,10))
            plt.plot(Zlinha, Z2linha,"o")
            plt.plot(Zlinha,resultado.best_fit)
            plt.xlabel("Z' (Ω)")
            plt.ylabel("-Z'' (Ω)")
            plt.show()
            Z2linhas.append(Z2linha)
            Zlinhas.append(Zlinha)
            Rss.append(Rs)
            resultados.append(resultado)

    return Zlinhas, Z2linhas, resultados, Rss

#Parte 3 - CV

def CV(diretorio, nome_da_amostra, Rs):
    arquivos = os.listdir(diretorio)
    
    scan_rate = []
    pontos_maximos = []

    # VAMOS ARMAZENAR TODOS OS DADOS PROCESSADOS
    lista_voltagens = []
    lista_correntes = []

    for nome_arquivo in arquivos:
        if nome_arquivo.endswith(".txt"):
            arquivo = os.path.join(diretorio, nome_arquivo)

            dados = pd.read_csv(arquivo, sep=";", decimal=",")
            voltagem = np.array(dados["WE(1).Potential (V)"])
            corrente = np.array(dados["WE(1).Current (A)"])
            tempo = np.array(dados["Time (s)"])

            # --- Scan rate ---
            sr = abs((voltagem[1] - voltagem[0]) / (tempo[1] - tempo[0]))
            scan_rate.append(sr)

            # --- Pico ---
            idx_max = np.where(voltagem == max(voltagem))
            pontos_maximos.append(corrente[idx_max[0][0]])

            # --- Corte seguro ---
            inicio = max(0, idx_max[0][0] - 5)
            voltagem = voltagem[inicio:]
            corrente = corrente[inicio:]

            # --- Correção E - iR ---
            voltagem_corrigida = Eir(voltagem, corrente, Rs) + 1

            # --- GUARDAR OS DADOS PARA O GRÁFICO FINAL ---
            lista_voltagens.append(voltagem_corrigida)
            lista_correntes.append(corrente)

    # --- Fitting linear ---
    modelo = fit.Model(linear)
    params = modelo.make_params(a=0.1, b=0)
    resultado = modelo.fit(pontos_maximos, params, x=scan_rate)

    # --- Plot local (pode deixar ou remover) ---
    plt.figure(figsize=(10, 10))
    for V, I in zip(lista_voltagens, lista_correntes):
        plt.plot(V, I, lw=1)

    plt.xlabel("E - iR (V)")
    plt.ylabel("J (A/cm²)")
    plt.title("CV - Todas as curvas")
    plt.show()

    return (
        lista_voltagens,   # <-- TODAS as curvas corrigidas
        lista_correntes,   # <-- TODAS as correntes
        scan_rate,
        pontos_maximos,
        resultado
    )

def escolher_diretorio(titulo):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=titulo)

#Fim das funções back-end
eel.init('/home/gabriel/Documentos/Programas/Eel/FrontEnd')#chama os arquivos front-end

nome_amostra = None
massa_amostra = None
diretorio_lsv = None
diretorio_EIS = None
diretorio_CV = None
diretorio_saida = None

@eel.expose
def set_nome_amostra(nome):
    global nome_amostra
    nome_amostra = nome
    print("Nome da amostra:", nome_amostra)
    return f"Nome da amostra salvo: {nome_amostra}"

@eel.expose
def set_massa(massa):
    global massa_amostra
    massa_amostra = massa
    print("Massa:", massa_amostra)
    return f"Massa salva: {massa_amostra}"

@eel.expose
def escolher_diretorio_lsv():
    global diretorio_lsv

    root = tk.Tk()
    root.withdraw()  # não mostra janela principal

    caminho = escolher_diretorio("Selecione o diretório LSV")

    if caminho:
        diretorio_lsv = caminho
        print("Diretório LSV absoluto:", diretorio_lsv)
        return diretorio_lsv
    else:
        return "Nenhum diretório selecionado"

@eel.expose
def escolher_diretorio_EIS():
    global diretorio_EIS

    root = tk.Tk()
    root.withdraw()  # não mostra janela principal

    caminho = escolher_diretorio("Selecione o diretório EIS")

    if caminho:
        diretorio_EIS = caminho
        print("Diretório EIS absoluto:", diretorio_EIS)
        return diretorio_EIS
    else:
        return "Nenhum diretório selecionado"
    
@eel.expose
def escolher_diretorio_CV():
    global diretorio_CV

    root = tk.Tk()
    root.withdraw()  # não mostra janela principal

    caminho = escolher_diretorio("Selecione o diretório CV")

    if caminho:
        diretorio_CV = caminho
        print("Diretório CV absoluto:", diretorio_CV)
        return diretorio_CV
    else:
        return "Nenhum diretório selecionado"
    
@eel.expose
def escolher_diretorio_saida():
    global diretorio_saida

    root = tk.Tk()
    root.withdraw()  # não mostra janela principal

    caminho = escolher_diretorio("Selecione o diretório de saída")

    if caminho:
        diretorio_saida = caminho
        print("Diretório de saída absoluto:", diretorio_saida)
        return diretorio_saida
    else:
        return "Nenhum diretório selecionado"

@eel.expose
def processar_dados():
    # ---------- VALIDAÇÃO ----------
    if not nome_amostra:
        return "Erro: nome da amostra não informado"

    if not massa_amostra:
        return "Erro: massa da amostra não informada"

    if not diretorio_lsv:
        return "Erro: diretório LSV não selecionado"

    if not diretorio_EIS:
        return "Erro: diretório EIS não selecionado"

    if not diretorio_CV:
        return "Erro: diretório CV não selecionado"

    if not diretorio_saida:
        return "Erro: diretório de saída não selecionado"

    # ---------- CONVERSÕES ----------
    try:
        massa_da_amostra = float(massa_amostra)
    except ValueError:
        return "Erro: massa inválida"

    nome_da_amostra = nome_amostra

    print("Processando dados com:")
    print("Nome:", nome_da_amostra)
    print("Massa:", massa_da_amostra)
    print("LSV:", diretorio_lsv)

    # ---------- NOMES DE SAÍDA ----------
    nome_saida1 = f"Analise_EIS_{nome_da_amostra}.png"
    nome_saida2 = f"Analise_Integrada_{nome_da_amostra}.png"

    saida1 = os.path.join(diretorio_saida, nome_saida1)
    saida2 = os.path.join(diretorio_saida, nome_saida2)

    # ---------- EXECUÇÃO CIENTÍFICA ----------
    try:
        J, J_log, EIR, J_intervalo, EIR_intervalo, CP, eta, resultado_LSV, alvos = LSV(diretorio_lsv, nome_da_amostra)
        Zlinha, Z2linha, resultado_EIS, Rss = EIS(diretorio_EIS, nome_da_amostra)
        lista_voltagens, lista_correntes, scan_rate, pontos_maximos, resultado_CV = CV(diretorio_CV, nome_da_amostra, Rss[0])
    except Exception as e:
        return f"Erro durante processamento: {e}"

    # ---------- PLOTS ----------

    #Primeiro as figuras individuais - EIS
    fig, axs = plt.subplots(1,2,figsize=(12,12))
    axs[0].plot(Zlinha[0], Z2linha[0], "o")
    axs[0].plot(Zlinha[0],resultado_EIS[0].best_fit)
    axs[0].set_xlabel("Z' (Ω)")
    axs[0].set_ylabel("-Z'' (Ω)")
    axs[1].plot(x,y, "o")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    plt.savefig(saida1)
    plt.show()
    

    #Depois a figura composta
    fig, axs = plt.subplots(2, 3, figsize=(12, 12))
    # LSV
    axs[0, 0].plot(EIR, J, label=nome_da_amostra)
    axs[0, 0].set_xlabel("EIR vs. RHE (V)")
    axs[0, 0].set_ylabel("J (mA/cm²)")
    axs[0, 0].legend()
    # Eta
    axs[0, 1].bar(alvos, eta, width=20, label=nome_da_amostra)
    axs[0, 1].set_xlabel("miliAmpere")
    axs[0, 1].set_ylabel(r"$\eta$")
    axs[0, 1].legend()
    # Tafel
    axs[0, 2].plot(J_log, EIR, 'b-', label=nome_da_amostra, lw=3)
    axs[0, 2].plot(J_intervalo, resultado_LSV.best_fit, 'r-', label='Fitting Linear', lw=2)
    axs[0, 2].set_xlim(-0.50, J_log[-1])
    axs[0, 2].set_ylim(1.35, EIR[-1])
    axs[0, 2].set_xlabel(r"$Log_{10}$ J (mA/cm²)")
    axs[0, 2].set_ylabel("EIR vs. RHE (V)")
    axs[0, 2].legend()
    # CV
    for V, I in zip(lista_voltagens, lista_correntes):
        axs[1, 0].plot(V, I, lw=1)
    axs[1, 0].set_xlabel("E - iR (V)")
    axs[1, 0].set_ylabel("J (A/cm²)")
    # Scan Rate vs Pico
    axs[1, 1].plot(scan_rate, resultado_CV.best_fit, label='Fitting Linear', color='red')
    axs[1, 1].plot(scan_rate, pontos_maximos, 'o')
    axs[1, 1].set_xlabel("Scan Rate (V/s)")
    axs[1, 1].set_ylabel("J máximo (A/cm²)")
    # Fig não concluida (função linear para ficar de exemplo)
    axs[1, 2].plot(x, y, 'o')
    axs[1, 2].set_xlabel("X")
    axs[1, 2].set_ylabel("Y")
    plt.tight_layout()
    plt.savefig(saida2)
    plt.show()

    return "Processamento concluído com sucesso"

eel.start('pagPrincipal.html') #inicia a pagina html