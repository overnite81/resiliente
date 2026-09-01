import sys
sys.path.insert(0, '.')
from GrafoPorAnomaliasAzureSpot import Checkpoint, Options
import java.io.*

# Criar um checkpoint com dados de teste
checkpoint = Checkpoint()
checkpoint.nRolls = 2
checkpoint.nIters = 4
checkpoint.radius = 1
checkpoint.xMultiplier = 1.0
checkpoint.yMultiplier = 1.0
checkpoint.onlyFromMaximalComponent = 0
checkpoint.lessThan = False
checkpoint.percentile = 0.5

# Simular dados de entrada
for i in range(checkpoint.nIters):
    for j in range(i+1, checkpoint.nIters):
        checkpoint.dtws[i][j] = float(i - j)  # Distância simples

print("Checkpoint criado com sucesso")
print(f"nRolls={checkpoint.nRolls}, nIters={checkpoint.nIters}, radius={checkpoint.radius}")
