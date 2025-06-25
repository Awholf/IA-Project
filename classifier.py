import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class NewsTopicClassifier:
    """
    Clase para clasificar noticias utilizando el algoritmo Naive Bayes y el dataset 20 Newsgroups.
    Maneja el preprocesamiento de texto, carga de datos, entrenamiento y predicción.
    """

    def __init__(self):
        """
        Inicializa el clasificador con los componentes necesarios.
        Configura el modelo, vectorizador, pipeline y recursos de preprocesamiento.
        """
        self.model = None  # Modelo Naive Bayes
        self.vectorizer = None  # Vectorizador TF-IDF
        self.pipeline = None  # Pipeline de procesamiento y clasificación
        self.categories = []  # Lista de categorías disponibles
        self.category_mapping = {  # Mapeo de nombres de categorías para mejor legibilidad
            'alt.atheism': 'Atheism',
            'comp.graphics': 'Computer Graphics',
            'comp.os.ms-windows.misc': 'Windows Misc',
            'comp.sys.ibm.pc.hardware': 'IBM PC Hardware',
            'comp.sys.mac.hardware': 'Mac Hardware',
            'comp.windows.x': 'Windows X',
            'misc.forsale': 'For Sale',
            'rec.autos': 'Autos',
            'rec.motorcycles': 'Motorcycles',
            'rec.sport.baseball': 'Baseball',
            'rec.sport.hockey': 'Hockey',
            'sci.crypt': 'Cryptography',
            'sci.electronics': 'Electronics',
            'sci.med': 'Medicine',
            'sci.space': 'Space',
            'soc.religion.christian': 'Christianity',
            'talk.politics.guns': 'Politics - Guns',
            'talk.politics.mideast': 'Politics - Mideast',
            'talk.politics.misc': 'Politics - Misc',
            'talk.religion.misc': 'Religion - Misc'
        }
        self.stemmer = SnowballStemmer('english')  # Stemmer para inglés

        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("⚠️  Usando conjunto básico de stopwords en inglés")
            self.stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
                'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
            }

    def preprocess_text(self, text):
        """
        Preprocesa un texto para la clasificación: limpia, tokeniza y elimina stopwords.
        Maneja headers de emails, URLs, caracteres especiales y aplica stemming.

        Args:
            text: Texto a preprocesar (str)
        Returns:
            Texto preprocesado como una cadena de palabras (str)
        """
        try:
            if not isinstance(text, str):
                return ""

            text = text.lower()
            text = re.sub(r'^.*?subject:.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
            text = re.sub(r'^.*?from:.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
            text = re.sub(r'^.*?organization:.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
            text = re.sub(r'^.*?lines:.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            try:
                tokens = word_tokenize(text, language='english')
            except LookupError:
                tokens = text.split()

            processed_tokens = [
                self.stemmer.stem(token) for token in tokens
                if token not in self.stop_words and len(token) > 2 and not token.isdigit()
            ]

            return ' '.join(processed_tokens)

        except Exception as e:
            print(f"⚠️  Error procesando texto: {e}")
            return text.lower() if isinstance(text, str) else ""

    def load_20newsgroups_dataset(self, subset_categories=None, remove_headers=True):
        """
        Carga el dataset 20 Newsgroups y lo organiza en un DataFrame.

        Args:
            subset_categories: Lista opcional de categorías a cargar (default: None, carga todas)
            remove_headers: Booleano para remover headers, footers y citas (default: True)
        Returns:
            DataFrame con columnas: texto, categoria_original, categoria_num, categoria
        """
        print("📂 Iniciando carga del dataset 20 Newsgroups...")

        try:
            if subset_categories is None:
                subset_categories = [
                    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                    'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                    'talk.politics.misc', 'talk.religion.misc'
                ]

            remove = ('headers', 'footers', 'quotes') if remove_headers else None

            print("⬇️  Descargando datos de entrenamiento...")
            newsgroups_train = fetch_20newsgroups(
                subset='train', categories=subset_categories, shuffle=True, random_state=42, remove=remove
            )

            print("⬇️  Descargando datos de prueba...")
            newsgroups_test = fetch_20newsgroups(
                subset='test', categories=subset_categories, shuffle=True, random_state=42, remove=remove
            )

            all_data = newsgroups_train.data + newsgroups_test.data
            all_targets = list(newsgroups_train.target) + list(newsgroups_test.target)
            target_names = [newsgroups_train.target_names[i] for i in all_targets]

            df = pd.DataFrame({
                'texto': all_data,
                'categoria_original': target_names,
                'categoria_num': all_targets
            })

            df['categoria'] = df['categoria_original'].map(lambda x: self.category_mapping.get(x, x))
            df = df[df['texto'].str.len() > 50].copy()
            df = df.dropna().reset_index(drop=True)

            print(f"✅ Dataset 20 Newsgroups cargado exitosamente!")
            print(f"📊 Total de documentos: {len(df)}")
            print(f"📊 Número de categorías: {len(df['categoria'].unique())}")
            print(f"📊 Categorías disponibles:")
            for cat in sorted(df['categoria'].unique()):
                count = len(df[df['categoria'] == cat])
                print(f"   • {cat}: {count} documentos")

            return df

        except Exception as e:
            print(f"❌ Error cargando 20 Newsgroups: {e}")
            print("💡 Verifica tu conexión a internet para descargar el dataset")
            return None

    def create_balanced_subset(self, df, samples_per_category=100):
        """
        Crea un subconjunto balanceado del dataset.

        Args:
            df: DataFrame con los datos completos
            samples_per_category: Número de muestras por categoría (default: 100)
        Returns:
            DataFrame balanceado
        """
        print(f"⚖️  Creando subconjunto balanceado con {samples_per_category} muestras por categoría...")

        balanced_dfs = []
        for category in df['categoria'].unique():
            category_df = df[df['categoria'] == category]
            sampled_df = category_df.sample(
                n=min(samples_per_category, len(category_df)), random_state=42
            )
            balanced_dfs.append(sampled_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✅ Subconjunto balanceado creado: {len(balanced_df)} documentos")
        return balanced_df

    def train_model(self, df):
        """
        Entrena el modelo Naive Bayes con un DataFrame preprocesado.

        Args:
            df: DataFrame con columnas 'texto' y 'categoria'
        Returns:
            Tupla con: X_test, y_test, y_pred, accuracy
        """
        print("🔄 Iniciando entrenamiento del modelo...")

        print("🔄 Preprocesando textos...")
        df['texto_procesado'] = df['texto'].apply(self.preprocess_text)
        df = df[df['texto_procesado'].str.len() > 10].copy()

        print(f"📊 Documentos después del preprocesamiento: {len(df)}")

        X_train, X_test, y_train, y_test = train_test_split(
            df['texto_procesado'], df['categoria'], test_size=0.25, random_state=42, stratify=df['categoria']
        )

        print(f"📊 Datos de entrenamiento: {len(X_train)} documentos")
        print(f"📊 Datos de prueba: {len(X_test)} documentos")
        print(f"📊 Proporción entrenamiento/prueba: {len(X_train)/(len(X_train)+len(X_test))*100:.0f}%/{len(X_test)/(len(X_train)+len(X_test))*100:.0f}%")

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words='english', strip_accents='ascii'
            )),
            ('nb', MultinomialNB(alpha=0.1))
        ])

        print("🔄 Entrenando modelo Naive Bayes...")
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Modelo entrenado exitosamente!")
        print(f"📊 Exactitud del modelo: {accuracy:.2%}")

        self.categories = sorted(list(df['categoria'].unique()))
        return X_test, y_test, y_pred, accuracy

    def predict_single_text(self, text):
        """
        Clasifica un solo texto y devuelve la categoría predicha con sus probabilidades.

        Args:
            text: Texto a clasificar (str)
        Returns:
            Tupla: (categoría predicha, diccionario de probabilidades LGBTQ

        """
        if self.pipeline is None:
            return "Error: El modelo no ha sido entrenado.", {}

        try:
            processed_text = self.preprocess_text(text)
            if len(processed_text.strip()) < 5:
                return "Error: Texto demasiado corto después del preprocesamiento.", {}

            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            prob_dict = dict(zip(self.pipeline.classes_, probabilities))

            return prediction, prob_dict

        except Exception as e:
            return f"Error durante la clasificación: {str(e)}", {}

    def generate_analysis_plots(self, y_test, y_pred):
        # Calcular la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        categories = sorted(set(y_test))

        # Calcular métricas por categoría
        metrics = {}
        for i, category in enumerate(categories):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # Valor predictivo negativo
            jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            
            metrics[category] = {
                'precision': precision,
                'recall': recall,
                'npv': npv,
                'jaccard': jaccard
            }

        # Calcular exactitud global
        accuracy = accuracy_score(y_test, y_pred)

        # Usar todas las categorías
        all_categories = list(metrics.keys())

        # Crear la figura con 6 subplots (2 filas, 3 columnas)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        # Subplot 1: Matriz de confusión con seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=categories, yticklabels=categories)
        axes[0].set_title('Matriz de Confusión')
        axes[0].set_xlabel('Predicho')
        axes[0].set_ylabel('Real')

        # Subplot 2: Exactitud (Accuracy)
        axes[1].bar(['Exactitud'], [accuracy], color='skyblue')
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Exactitud')
        axes[1].text(0, accuracy + 0.02, f'{accuracy:.2f}', ha='center')

        # Subplot 3: Precisión (Precision, igual a VPP)
        precisions = [metrics[cat]['precision'] for cat in all_categories]
        axes[2].barh(all_categories, precisions, color='lightgreen')
        axes[2].set_xlim(0, 1)
        axes[2].set_title('Precisión (VPP)')

        # Subplot 4: Sensibilidad (Recall)
        recalls = [metrics[cat]['recall'] for cat in all_categories]
        axes[3].barh(all_categories, recalls, color='salmon')
        axes[3].set_xlim(0, 1)
        axes[3].set_title('Sensibilidad')

        # Subplot 5: Valores Predictivos Negativos (VPN)
        npvs = [metrics[cat]['npv'] for cat in all_categories]
        axes[4].barh(all_categories, npvs, color='lightblue')
        axes[4].set_xlim(0, 1)
        axes[4].set_title('VPN')

        # Subplot 6: Medida de Jaccard
        jaccards = [metrics[cat]['jaccard'] for cat in all_categories]
        axes[5].barh(all_categories, jaccards, color='orchid')
        axes[5].set_xlim(0, 1)
        axes[5].set_title('Medida de Jaccard')

        # Ajustar el diseño
        plt.tight_layout()
        return fig

    