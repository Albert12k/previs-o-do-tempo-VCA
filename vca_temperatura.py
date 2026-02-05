import pandas as pd #* Excel do Python serve para criar e organizar a tabela do python
from sklearn.linear_model import LinearRegression #* é o cérebro, é aqui que fica a I.A matematica que traça a linha da tendencia
import matplotlib.pyplot as plt #* serve só pra desenhar o grafico

# --- 1. TREINAMENTO (pega o historico da temperatura da cidade) ---
# ensina o computador como a temperatura se comporta nos meses  de Jan a Jun
dados = {
    'mes_numero': [1, 2, 3, 4, 5, 6],
    'nome_mes': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun'],
    'temperatura_media': [23.0, 23.5, 23.5, 22.5, 21.0, 20.0] 
}
df = pd.DataFrame(dados)

# Separando as variáveis
X = df[['mes_numero']]       # O tempo (Mês)
y = df['temperatura_media']  # A temperatura

# Criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# --- 2. PREVISÃO PARA MARÇO, ABRIL E MAIO DE 2026 ---
meses_alvo = [3, 4, 5]
nomes_alvo = ['Março', 'Abril', 'Maio']

# Pedindo para a IA calcular
X_futuro = pd.DataFrame(meses_alvo, columns=['mes_numero'])
previsoes = modelo.predict(X_futuro)

print("-" * 40)
print("--- Previsão IA para 2026 ---")
for i in range(3):
    print(f"{nomes_alvo[i]}: {previsoes[i]:.1f}°C")
print("-" * 40)

# --- 3. GERANDO O GRÁFICO ---
plt.figure(figsize=(10, 6))

# Bolinhas Azuis: A Realidade (Média Histórica)
plt.scatter(X, y, color='blue', alpha=0.6, s=80, label='Histórico VCA')

# Linha Vermelha: A "Régua" da IA (Tendência)
plt.plot(X, modelo.predict(X), color='red', linestyle='--', label='Tendência Linear')

# X Verde: A Previsão específica para 2026
plt.scatter(X_futuro, previsoes, color='green', marker='X', s=200, label='Previsão 2026')

# Escrevendo os valores no gráfico para ficar chique
for i in range(3):
    plt.annotate(f"{previsoes[i]:.1f}°C", 
                 (meses_alvo[i], previsoes[i]), 
                 textcoords="offset points", xytext=(0,15), ha='center',
                 fontweight='bold', color='green')

# Título atualizado com o ano
plt.title('Previsão de Temperatura VCA: Março, Abril e Maio de 2026')
plt.xlabel('Mês')
plt.ylabel('Temperatura (°C)')
plt.xticks(list(range(1, 7)), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun'])
plt.legend()
plt.grid(True, alpha=0.3)

# --- 4. SALVANDO O ARQUIVO ---
nome_arquivo = 'previsao_vca_2026.png'
plt.savefig(nome_arquivo)
print(f"✅ SUCESSO! O gráfico foi salvo como '{nome_arquivo}' na sua pasta.")