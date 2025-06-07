import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
from threading import Thread
import time

# ===============================
# Chargement des données
# ===============================
try:
    df = pd.read_csv('cefr_fr.csv')
except FileNotFoundError:
    from datasets import load_dataset
    dataset = load_dataset("vekkt/french_CEFR")
    df = pd.DataFrame(dataset['train'])
    df = df.rename(columns={'sentence': 'text', 'difficulty': 'label'})
    df.to_csv('cefr_fr.csv', index=False)
    print("✔ Dataset téléchargé depuis Hugging Face.")

# Nettoyage
df.dropna(inplace=True)
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# ===============================
# Préparation des données
# ===============================
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = tf.keras.utils.to_categorical(labels_encoded, num_classes=6)

vocab_size = 10000
max_length = 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded, labels_onehot, test_size=0.2, random_state=42)

# ===============================
# Modèle avec 5 perceptrons
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("📚 Entraînement du modèle en cours...")
history = model.fit(X_train, y_train, epochs=15, batch_size=4,
                    validation_data=(X_test, y_test), verbose=1)

# ===============================
# Évaluation
# ===============================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🎯 Accuracy test : {accuracy * 100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
print("Matrice de confusion :")
print(confusion_matrix(y_true_classes, y_pred_classes))

# ===============================
# Fonction de prédiction
# ===============================
def predict_level(text):
    if not text.strip():
        return None
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded_seq, verbose=0)
    level = label_encoder.inverse_transform([np.argmax(pred)])
    return level[0]

# ===============================
# Interface graphique avec Tkinter
# ===============================
class CEFRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détection du niveau CECRL")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f2f5")

        # Style
        self.style_config()

        # Widgets
        self.create_widgets()

        # Prédiction en temps réel
        self.last_text = ""
        self.root.after(500, self.real_time_prediction)

    def style_config(self):
        self.bg_color = "#f0f2f5"
        self.accent_color = "#7f5af0"
        self.text_color = "#16161a"
        self.entry_font = ("Arial", 12)
        self.label_font = ("Arial", 14, "bold")
        self.button_font = ("Arial", 12, "bold")

    def create_widgets(self):
        # Titre
        tk.Label(self.root, text="Détecteur de Niveau CECRL", font=("Arial", 16, "bold"),
                 bg=self.bg_color, fg=self.text_color).pack(pady=10)

        # Champ de saisie
        self.entry = tk.Entry(self.root, width=50, font=self.entry_font, bg="white", fg=self.text_color,
                              borderwidth=2, relief="flat")
        self.entry.pack(pady=10, padx=20)
        self.entry.bind("<KeyRelease>", lambda event: self.update_last_text())

        # Label de résultat
        self.result_label = tk.Label(self.root, text="Niveau prédit : Aucun", font=self.label_font,
                                    bg=self.bg_color, fg=self.accent_color)
        self.result_label.pack(pady=10)

        # Bouton de réinitialisation
        tk.Button(self.root, text="Réinitialiser", command=self.clear_entry,
                  font=self.button_font, bg=self.accent_color, fg="white",
                  relief="flat", activebackground="#6b48d6").pack(pady=5)

        # Historique des prédictions
        tk.Label(self.root, text="Historique des prédictions", font=("Arial", 12, "bold"),
                 bg=self.bg_color, fg=self.text_color).pack(pady=5)
        self.history_text = scrolledtext.ScrolledText(self.root, height=5, width=60, font=("Arial", 10),
                                                     bg="white", fg=self.text_color, state='disabled')
        self.history_text.pack(pady=10, padx=20)

    def update_last_text(self):
        self.last_text = self.entry.get()

    def real_time_prediction(self):
        current_text = self.entry.get()
        if current_text != self.last_text and current_text.strip():
            Thread(target=self.predict_and_update).start()
        self.root.after(500, self.real_time_prediction)

    def predict_and_update(self):
        text = self.entry.get()
        level = predict_level(text)
        if level:
            self.result_label.config(text=f"Niveau prédit : {level}")
            self.update_history(text, level)
        else:
            self.result_label.config(text="Niveau prédit : Aucun")

    def update_history(self, text, level):
        self.history_text.config(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{timestamp}] \"{text}\" -> {level}\n")
        self.history_text.see(tk.END)
        self.history_text.config(state='disabled')

    def clear_entry(self):
        self.entry.delete(0, tk.END)
        self.result_label.config(text="Niveau prédit : Aucun")
        self.last_text = ""

# Lancer l'application
if __name__ == "__main__":
    window = tk.Tk()
    app = CEFRApp(window)
    window.mainloop()