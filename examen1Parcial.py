import pandas as pd

def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)


datos = importarCSV('https://archive.ics.uci.edu/ml/machine-learning-databases/00314/%5bUCI%5d%20AAAI-13%20Accepted%20Papers%20-%20Papers.csv')
print(datos)
