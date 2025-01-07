# import numpy as np
# import matplotlib.pyplot as plt
# from pandas import read_csv
# import math
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)
# # fix random seed for reproducibility
# tf.random.set_seed(7)
# # load the dataset
# dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# dataset = dataframe.values
# dataset = dataset.astype('float32')

# plt.figure()
# plt.plot(dataset)
# plt.show()

# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# plt.figure()
# plt.plot(dataset)
# plt.show()

# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# plt.figure()
# plt.plot(history.history['loss'], label='loss')
# plt.ylim([0, 0.01])
# plt.xlabel('Epoch')
# plt.ylabel('Error [passangers]')
# plt.legend()
# plt.grid(True)

# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and 
# plt.figure()
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# import tensorflow as tf

# # Configurações iniciais do Streamlit
# st.title("Previsão de Séries Temporais com LSTM")
# st.write("""
# Este aplicativo utiliza Redes Neurais Recorrentes (LSTM) para prever séries temporais. 
# Carregue os dados, visualize os resultados e explore o modelo com novos datasets.
# """)

# # Função para converter valores em datasets X e Y
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)

# # Configurações do Random Seed
# tf.random.set_seed(7)

# # Campo para upload de dataset personalizado
# st.sidebar.header("Configurações")
# uploaded_file = st.sidebar.file_uploader("Carregar Dataset CSV", type=["csv"])

# # Opções de look_back
# look_back = st.sidebar.slider("Quantidade de Passos Anteriores (look_back)", min_value=1, max_value=10, value=5)

# # Upload do dataset ou uso do exemplo padrão
# if uploaded_file:
#     st.write("### Dataset Carregado")
#     dataframe = pd.read_csv(uploaded_file)
# else:
#     st.write("### Usando Dataset Exemplo (Airline Passengers)")
#     dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python', names=['Passengers'], header=0)

# st.write(dataframe.head())

# # Converter dados para numpy array
# dataset = dataframe.values.astype('float32')

# # Normalizar os dados
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# # Divisão em treino e teste
# train_size = int(len(dataset) * 0.67)
# train, test = dataset[:train_size, :], dataset[train_size:, :]

# # Criar datasets X e Y
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# # Ajustar o shape para [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# # Criação e treinamento do modelo LSTM
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# st.write("### Treinando o Modelo")
# with st.spinner("Treinando..."):
#     history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
# st.success("Treinamento Concluído!")

# # Exibir a curva de perda durante o treinamento
# st.write("### Curva de Perda Durante o Treinamento")
# fig, ax = plt.subplots()
# ax.plot(history.history['loss'], label='Erro de Treinamento')
# ax.set_xlabel('Épocas')
# ax.set_ylabel('Erro (Loss)')
# ax.set_title('Erro por Época')
# ax.legend()
# st.pyplot(fig)

# # Fazer predições
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

# # Reverter normalização
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

# # Calcular RMSE
# trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
# st.write(f"### Erro Quadrático Médio (RMSE)")
# st.write(f"- Treinamento: {trainScore:.2f}")
# st.write(f"- Teste: {testScore:.2f}")

# # Preparar predições para visualização
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# # Plotar os resultados
# st.write("### Comparação de Valores Reais e Predições")
# fig, ax = plt.subplots()
# ax.plot(scaler.inverse_transform(dataset), label='Valores Reais', color='blue')
# ax.plot(trainPredictPlot, label='Previsões - Treino', color='green')
# ax.plot(testPredictPlot, label='Previsões - Teste', color='red')
# ax.set_title("Comparação de Previsões")
# ax.legend()
# st.pyplot(fig)

# # Permitir ao usuário salvar as previsões
# st.sidebar.header("Salvar Resultados")
# if st.sidebar.button("Salvar Previsões"):
#     pred_train_df = pd.DataFrame(trainPredict, columns=['Train Predictions'])
#     pred_test_df = pd.DataFrame(testPredict, columns=['Test Predictions'])
#     pred_train_df.to_csv("train_predictions.csv", index=False)
#     pred_test_df.to_csv("test_predictions.csv", index=False)
#     st.sidebar.success("Previsões Salvas como CSV!")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Configuração de seed para reprodutibilidade
tf.random.set_seed(7)

# Título do aplicativo
st.title("Previsão de Passagens Aéreas com LSTM")

# Explicação do problema e solução
st.markdown("""
## Descrição do Problema

Este modelo usa uma Rede Neural Recorrente (RNN) com LSTM (Long Short-Term Memory) para prever a quantidade de passageiros com base em dados históricos de passageiros em um determinado período. O objetivo é prever os valores futuros, considerando os dados passados como entradas.

### Etapas do Modelo:
1. **Carregar Dados**: O modelo carrega os dados históricos de passageiros.
2. **Preprocessamento**: Os dados são normalizados para melhorar a precisão do modelo.
3. **Treinamento**: A rede LSTM é treinada com os dados de treinamento.
4. **Previsão e Avaliação**: O modelo faz previsões e calcula o erro médio quadrado (RMSE).
5. **Visualização**: O gráfico final mostra as previsões comparadas com os dados reais.

Agora, você pode carregar seu próprio arquivo CSV e visualizar as previsões do modelo.
""")

# Upload de arquivo CSV
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Carregar os dados do arquivo
    df = pd.read_csv(uploaded_file)
    st.write("Primeiras linhas do arquivo:", df.head())

    # Verificar se existe uma coluna de datas e transformar
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'])
    
    # Verificar se existe uma coluna numérica
    if 'total_passengers' in df.columns:
        values = df['total_passengers'].values
    else:
        st.error("A coluna 'Passengers' não foi encontrada. Verifique o nome da coluna no arquivo CSV.")
        st.stop()

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = values.reshape(-1, 1)
    dataset = scaler.fit_transform(values)

    st.write("Dados Normalizados")
    st.line_chart(dataset)

    # Dividir em treino e teste
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Função para criar dataset com look_back
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Definir look_back
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape para [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Criar e treinar o modelo LSTM
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Treinamento
    history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    st.write("Histórico de Erro do Treinamento (Loss)")
    st.line_chart(history.history['loss'])

    # Fazer previsões
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverter previsões
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calcular erro médio quadrado (RMSE)
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    st.write(f"Train Score (RMSE): {trainScore:.2f}")
    st.write(f"Test Score (RMSE): {testScore:.2f}")

    # Plotar previsões
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    st.write("Previsões vs Valores Reais")
    plt.figure()
    plt.plot(scaler.inverse_transform(dataset), label='Valores Reais')
    plt.plot(trainPredictPlot, label='Previsões de Treinamento')
    plt.plot(testPredictPlot, label='Previsões de Teste')
    plt.title('Comparação de Previsões e Valores Reais')
    plt.legend()
    st.pyplot(plt)


