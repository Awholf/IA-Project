import nltk

def download_nltk_resources():
    """
    Descarga los recursos necesarios de NLTK para el procesamiento de texto.
    Verifica si los recursos ya están disponibles y los descarga si es necesario.
    """
    print("📥 Iniciando descarga de recursos de NLTK...")
    
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
            print(f"✅ Recurso '{resource}' ya está disponible")
        except LookupError:
            try:
                print(f"⬇️  Descargando '{resource}'...")
                nltk.download(resource, quiet=True)
                print(f"✅ Recurso '{resource}' descargado exitosamente")
            except Exception as e:
                print(f"⚠️  Error al descargar '{resource}': {e}")
    
    print("🎉 Recursos de NLTK configurados correctamente\n")