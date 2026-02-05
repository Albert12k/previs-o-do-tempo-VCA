# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS (Nossas Ferramentas) ---

# Pandas: O "Excel" do Python.
# Usamos para organizar os dados em tabelas (linhas e colunas) que a gente consegue ler.
import pandas as pd

# load_breast_cancer: O "Banco de Dados".
# Essa função traz os dados reais de biópsias que já vêm salvos dentro do Scikit-Learn.
from sklearn.datasets import load_breast_cancer

# train_test_split: O "Separador".
# Ele serve para dividir nossos dados em dois pedaços:
# 1. Treino (para a IA estudar) e 2. Teste (para a gente aplicar a prova nela depois).
from sklearn.model_selection import train_test_split

# RandomForestClassifier: O "Cérebro" (O Modelo de IA).
# É o algoritmo de Classificação. "Random Forest" (Floresta Aleatória) é muito bom
# porque ele cria várias "árvores de decisão" e faz uma votação para decidir se é benigno ou maligno.
from sklearn.ensemble import RandomForestClassifier

# accuracy_score e ConfusionMatrixDisplay: Os "Avaliadores".
# accuracy_score: Calcula a nota (ex: 95%).
# ConfusionMatrixDisplay: Desenha aquele gráfico mostrando onde ele acertou e errou.
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Matplotlib: O "Pintor".
# Usamos apenas para salvar o gráfico como imagem no final.
import matplotlib.pyplot as plt


# --- 2. CARREGANDO OS DADOS ---
# Carregando os dados de câncer de mama
dados = load_breast_cancer()

# Criando a tabela (DataFrame) para visualizar
df = pd.DataFrame(dados.data, columns=dados.feature_names)
df['diagnostico'] = dados.target # 0 = Maligno, 1 = Benigno

# Traduzindo para ficar fácil de ler no print
df['nome_diagnostico'] = df['diagnostico'].replace({0: 'Maligno', 1: 'Benigno'})

print("--- Amostra dos Dados (Primeiras 5 linhas) ---")
print(df[['mean radius', 'mean texture', 'nome_diagnostico']].head())
print("-" * 40)

# --- 3. SEPARANDO TREINO E TESTE ---
X = dados.data   # As medidas (Raio, Textura, etc.)
y = dados.target # A resposta (0 ou 1)

# Separando 30% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 4. TREINAMENTO ---
modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino) # O robô estuda aqui

# --- 5. AVALIAÇÃO ---
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do Modelo: {acuracia:.2%}")

# --- 6. GRÁFICO (Matriz de Confusão) ---
disp = ConfusionMatrixDisplay.from_estimator(
    modelo, 
    X_teste, 
    y_teste,
    display_labels=['Maligno', 'Benigno'],
    cmap=plt.cm.Blues
)
plt.title("Matriz de Confusão: Biópsia")

# Salvando
plt.savefig('grafico_biopsia.png')
print("✅ SUCESSO! Gráfico salvo como 'grafico_biopsia.png'")