import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

##### PEGAR INFOS DE 2012-01 A 2020-12 ##########
# Placeholder para carregar e preparar os dados (substitua pelo seu dataset)
dolar = pd.read_csv('dolar.csv')  # Substitua pelo caminho do arquivo
dolar = dolar[dolar['tipoBoletim']=='Fechamento']
dolar = dolar[dolar['dataHoraCotacao']>= '2000-01-01']
dolar = dolar[['dataHoraCotacao','cotacaoVenda']]
# # Converter a coluna 'data' para datetime
dolar['dataHoraCotacao'] = pd.to_datetime(dolar['dataHoraCotacao'])
# Adicionar uma coluna de mês/ano para agrupar
dolar['mes_ano'] = dolar['dataHoraCotacao'].dt.to_period('M')
# Selecionar a menor data de cada mês
resultado = dolar.loc[dolar.groupby('mes_ano')['dataHoraCotacao'].idxmin()]
# Remover a coluna 'mes_ano' se não for necessária
dolar = resultado.drop(columns=['mes_ano'])
dolar['cotacaoVenda'] = dolar['cotacaoVenda'].str.replace(',','.').astype(float)
dolar = dolar[dolar['dataHoraCotacao'] >= '2012-01-01']
dolar_teste = dolar[dolar['dataHoraCotacao'] >= '2021-01-01']
dolar_teste = dolar_teste[dolar_teste['dataHoraCotacao'] <= '2023-12-31']
dolar = dolar[dolar['dataHoraCotacao'] <= '2020-12-31']

# print(resultado)

combustivel = pd.read_csv('combustiveis.csv')
combustivel = combustivel[combustivel['referencia']>= '2000-01']
combustivel = combustivel[['referencia','oleo_diesel_preco_revenda_avg']]
combustivel = combustivel[combustivel['referencia'] >= '2012-01']
combustivel_teste = combustivel[combustivel['referencia'] >= '2021-01']
combustivel_teste = combustivel_teste[combustivel_teste['referencia'] <= '2023-12']
combustivel = combustivel[combustivel['referencia'] <= '2020-12']

##AJUSTAR A PARTIR DAQUI
juros = pd.read_csv('juros.csv',delimiter=';', decimal=',')
juros['Data'] = pd.to_datetime(juros['Data'])
juros_teste = juros[(juros['Data'] >= '01/01/2021') & (juros['Data'] <= '31/12/2023')]
juros = juros[(juros['Data'] >= '01/01/2012') & (juros['Data'] <= '31/12/2020')]


satisfacao = pd.read_csv('bcdatasgs.csv',delimiter=';', decimal=',')
satisfacao['data'] = pd.to_datetime(satisfacao['data'])
satisfacao_teste = satisfacao[(satisfacao['data'] >= '01/01/2021') & (satisfacao['data'] <= '31/12/2023')]
satisfacao = satisfacao[(satisfacao['data'] >= '01/01/2012') & (satisfacao['data'] <= '31/12/2020')]
# Supondo que seu dataset tenha as colunas:
# 'dolar', 'salario_minimo', 'preco_arroz', 'clima', 'area_plantada', 'produtividade', 'preco_combustivel', 'taxa_juros', 'subsidiado', 'confiança', 'concorrência', 'vendas'

# Separando dados de entrada (X) e saída (y)
#X = data[['dolar', 'salario_minimo', 'preco_arroz', 'clima', 'area_plantada', 'produtividade', 'preco_combustivel', 'taxa_juros', 'subsidiado', 'confiança', 'concorrência']]
X = [dolar['cotacaoVenda'].values, combustivel['oleo_diesel_preco_revenda_avg'].values, juros['Taxa media'].values, satisfacao['valor'].values]
X_teste = [dolar_teste['cotacaoVenda'].values, combustivel_teste['oleo_diesel_preco_revenda_avg'].values, juros_teste['Taxa media'].values, satisfacao_teste['valor'].values]
X_teste = pd.DataFrame(X_teste)

y = pd.read_csv('vendas.csv',delimiter=';')
y = y[12:]
y = y['Colheitadeiras']

# Simulando dados
n_samples = 9  # Exemplo: 10 anos de dados
timesteps = 12  # 12 meses por ano
n_features = 4  # Exemplo: 3 variáveis mensais (dólar, salário, arroz)

# Dados simulados (substitua pelos seus)
X = np.random.rand(n_samples, timesteps, n_features)


# Criando o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(timesteps, n_features)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Saída: valor anual
])

model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
model.fit(X, y, epochs=50, batch_size=1)


X_novo = np.array(X_teste).reshape(1, 36, len(X_teste[0]))  # (1, 12, 3)

# Fazer a previsão
previsao = model.predict(X_novo)

# Resultado da previsão
print(f"Valor previsto para o próximo ano: {previsao[0][0]:.2f}")
