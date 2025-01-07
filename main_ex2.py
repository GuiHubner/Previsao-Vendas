import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Função para criar dataset com look_back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Função para o código antigo com explicação e alteração do look_back
def old_code():
    st.markdown("""
    ## Descrição do Problema

    Este modelo usa uma Rede Neural Recorrente (RNN) com LSTM (Long Short-Term Memory) para prever a quantidade de passageiros com base em dados históricos de passageiros em um determinado período. O objetivo é prever os valores futuros, considerando os dados passados como entradas.

    ### Etapas do Modelo:
    1. **Carregar Dados**: O modelo carrega os dados históricos de passageiros.
    2. **Preprocessamento**: Os dados são normalizados para melhorar a precisão do modelo.
    3. **Treinamento**: A rede LSTM é treinada com os dados de treinamento.
    4. **Previsão e Avaliação**: O modelo faz previsões e calcula o erro médio quadrado (RMSE).
    5. **Visualização**: O gráfico final mostra as previsões comparadas com os dados reais.

    """)
    
    st.markdown("""
    **Alterando o valor de `look_back`**: 
    O parâmetro `look_back` define a quantidade de dados anteriores a serem considerados para fazer a previsão. Você pode alterar esse valor para ver como o modelo se comporta com diferentes quantidades de dados históricos.
    """)

    look_back_value = st.slider("Escolha o valor de look_back", 1, 10, 1)
    st.write(f"O valor de look_back é: {look_back_value}")

    # Configuração para rodar o código
    tf.random.set_seed(7)

    # Carregar dados
    dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Dividir em treino e teste
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Criar datasets com look_back
    trainX, trainY = create_dataset(train, look_back_value)
    testX, testY = create_dataset(test, look_back_value)

    # Reshape para [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Criar e treinar o modelo LSTM
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back_value)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Fazer previsões
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverter previsões
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calcular RMSE
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    # Plotar previsões
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back_value:len(trainPredict) + look_back_value, :] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back_value * 2) + 1:len(dataset) - 1, :] = testPredict

    # Exibir o gráfico
    st.write("Previsões vs Valores Reais")
    plt.figure()
    plt.plot(scaler.inverse_transform(dataset), label="Real")
    plt.plot(trainPredictPlot, label="Previsão de Treinamento")
    plt.plot(testPredictPlot, label="Previsão de Teste")
    plt.legend()
    st.pyplot()


# Função para carregar o arquivo e realizar as previsões com o modelo LSTM
def new_code():
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


# Interface para selecionar a aba
st.sidebar.title("Escolha uma aba")
option = st.sidebar.radio("Selecione", ["Código Antigo", "Carregar Arquivo e Prever"])

if option == "Código Antigo":
    old_code()
else:
    new_code()
