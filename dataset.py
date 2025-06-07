from datasets import load_dataset
import pandas as pd

# Chargement du dataset
dataset = load_dataset("vekkt/french_CEFR")

# Conversion en DataFrame
df = pd.DataFrame(dataset['train'])

# Renommage des colonnes pour correspondre à ton modèle
df = df.rename(columns={'sentence': 'text', 'difficulty': 'label'})

# Sauvegarde en CSV
df.to_csv('cefr_fr.csv', index=False)
