import random

vetor = [35, 7, 13, 18, 32, 39, 10, 53, 5, 33, 4, 54, 23, 24, 42, 51, 17]


# Adicionando 2 números aleatórios entre 1 e 60, sem repetição
while len(vetor) < 18:
    numero_aleatorio = random.randint(1, 60)
    if numero_aleatorio not in vetor:  # Garante que não haverá repetição
        vetor.append(numero_aleatorio)

# Embaralhando os números aleatoriamente
random.shuffle(vetor)

# Dividindo o vetor em 3 conjuntos de 6 números
conjuntos = [vetor[i:i + 6] for i in range(0, len(vetor), 6)]

# Exibindo os conjuntos
for idx, conjunto in enumerate(conjuntos, 1):
    print(f"Conjunto {idx}: {conjunto}")
