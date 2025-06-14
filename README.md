# Détection du Niveau CECRL en Français - Prédiction en Temps Réel
# ONJATIANA TAHINJANAHARY Desire Fabrice L3 Da2i 022I23 

Ce projet utilise **TensorFlow** et **Tkinter** pour entraîner un modèle de classification du niveau de langue français selon le Cadre Européen Commun de Référence pour les Langues (CECRL) et propose une interface graphique moderne avec prédiction en temps réel.

---

## Fonctionnalités

- Chargement automatique du dataset CEFR français (via Hugging Face si absent localement).
- Prétraitement des données textuelles (tokenization, padding).
- Modèle de réseau de neurones profond (5 couches denses) pour la classification multi-classes (6 niveaux CECRL).
- Prédiction en temps réel dans l'interface utilisateur.
- UI/UX moderne avec Tkinter (design épuré, retour instantané).
- Support des entrées utilisateur avec gestion des erreurs (phrase vide).

---

## Prérequis

- Python 3.8+
- TensorFlow
- pandas
- numpy
- scikit-learn
- datasets (Hugging Face)
- tkinter (généralement inclus avec Python)

---

## Installation

1. Cloner ce dépôt :

```bash
git clone https://github.com/ton-utilisateur/cefr-level-prediction.git
cd cefr-level-prediction
