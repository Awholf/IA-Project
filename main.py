from interface import NewsClassifierInterface
import tkinter.messagebox as messagebox

def main():
    """
    Función principal para iniciar la aplicación.
    """
    print("🚀 Iniciando Clasificador de Noticias con Interfaz Gráfica")
    print("=" * 60)
    try:
        app = NewsClassifierInterface()
        app.run()
    except Exception as e:
        print(f"❌ Error iniciando la aplicación: {e}")
        messagebox.showerror("Error Fatal", f"No se pudo iniciar la aplicación:\n{str(e)}")

if __name__ == "__main__":
    main()