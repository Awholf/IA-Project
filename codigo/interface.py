import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, classification_report
from classifier import NewsTopicClassifier
from utils import download_nltk_resources

class NewsClassifierInterface:
    """
    Interfaz gráfica para el clasificador de noticias usando Tkinter.
    """

    def __init__(self):
        """
        Inicializa la interfaz gráfica y sus componentes principales.
        """
        self.root = tk.Tk()
        self.classifier = NewsTopicClassifier()
        self.model_trained = False
        self.training_thread = None
        self.df_trained = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Listo para cargar modelo")

        self.setup_window()
        self.setup_widgets()
        self.setup_menu()

    def setup_window(self):
        """
        Configura las propiedades básicas de la ventana principal.
        """
        self.root.title("Clasificador de Noticias - 20 Newsgroups con Naive Bayes")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'background': '#F5F5F5',
            'text': '#2D3436'
        }

    def setup_menu(self):
        """
        Configura el menú superior de la aplicación.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Cargar texto desde archivo", command=self.load_text_file)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)

        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modelo", menu=model_menu)
        model_menu.add_command(label="Entrenar modelo", command=self.start_training)
        model_menu.add_command(label="Ver métricas del modelo", command=self.show_model_metrics)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Acerca de", command=self.show_about)

    def setup_widgets(self):
        """
        Configura los widgets principales de la interfaz gráfica.
        """
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        ttk.Label(main_frame, text="🤖 Clasificador de Noticias con IA", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, columnspan=3, pady=(0, 20))

        ttk.Label(main_frame, text="20 Newsgroups Dataset • Algoritmo Naive Bayes", font=('Arial', 10)).grid(
            row=1, column=0, columnspan=3, pady=(0, 20))

        input_frame = ttk.LabelFrame(main_frame, text="📰 Ingrese el texto a clasificar", padding=10)
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=70, height=8, font=('Arial', 10))
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        placeholder_text = (
            "Ingrese aquí el texto que desea clasificar...\n\n"
            "💡 Consejos:\n"
            "• Funciona mejor con textos en inglés\n"
            "• Temas: tecnología, deportes, política, ciencia, religión, etc.\n"
            "• Mínimo 20 caracteres para mejor exactitud"
        )
        self.text_input.insert('1.0', placeholder_text)
        self.text_input.bind('<FocusIn>', self.clear_placeholder)

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.classify_btn = ttk.Button(button_frame, text="🎯 Clasificar Texto", command=self.classify_text, state='disabled')
        self.classify_btn.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="🗑️ Limpiar", command=self.clear_input).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="📁 Cargar archivo", command=self.load_text_file).pack(side=tk.LEFT)

        training_frame = ttk.LabelFrame(main_frame, text="🧠 Estado del Modelo", padding=10)
        training_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        training_frame.columnconfigure(1, weight=1)

        self.train_btn = ttk.Button(training_frame, text="🚀 Entrenar Modelo", command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=(0, 10))

        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.status_label = ttk.Label(training_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=2)

        results_frame = ttk.LabelFrame(main_frame, text="📊 Resultados de Clasificación", padding=10)
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        self.result_label = ttk.Label(results_frame, text="Esperando clasificación...", font=('Arial', 12, 'bold'),
                                     foreground=self.colors['primary'])
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        columns = ('Posición', 'Categoría', 'Probabilidad', 'Barra')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=6)
        self.results_tree.heading('Posición', text='#')
        self.results_tree.heading('Categoría', text='Categoría')
        self.results_tree.heading('Probabilidad', text='Probabilidad')
        self.results_tree.heading('Barra', text='Grafica de carga')
        self.results_tree.column('Posición', width=50, anchor='center')
        self.results_tree.column('Categoría', width=200)
        self.results_tree.column('Probabilidad', width=100, anchor='center')
        self.results_tree.column('Barra', width=200)
        self.results_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        info_text = (
            "💡 Este clasificador utiliza el algoritmo Naive Bayes entrenado con el dataset 20 Newsgroups. "
            "Para mejores resultados, use textos en inglés relacionados con los temas disponibles."
        )
        ttk.Label(info_frame, text=info_text, wraplength=800, font=('Arial', 9)).pack()

    def clear_placeholder(self, event):
        """
        Limpia el texto placeholder en el área de entrada.

        Args:
            event: Evento de Tkinter (foco en el widget).
        """
        current_text = self.text_input.get('1.0', tk.END).strip()
        if "Ingrese aquí el texto que desea clasificar..." in current_text:
            self.text_input.delete('1.0', tk.END)

    def clear_input(self):
        """
        Limpia el área de entrada y los resultados.
        """
        self.text_input.delete('1.0', tk.END)
        self.clear_results()

    def clear_results(self):
        """
        Limpia los resultados de clasificación en la interfaz.
        """
        self.result_label.config(text="Esperando clasificación...")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def load_text_file(self):
        """
        Carga un archivo de texto en el área de entrada.
        """
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de texto",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.text_input.delete('1.0', tk.END)
                self.text_input.insert('1.0', content)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")

    def start_training(self):
        """
        Inicia el entrenamiento del modelo en un hilo separado.
        """
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Advertencia", "El modelo ya se está entrenando...")
            return

        self.train_btn.config(state='disabled')
        self.classify_btn.config(state='disabled')
        self.progress_var.set(0)
        self.status_var.set("Iniciando entrenamiento...")

        self.training_thread = threading.Thread(target=self.train_model_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def train_model_thread(self):
        """
        Ejecuta el entrenamiento del modelo en un hilo separado.
        """
        try:
            steps = [
                ("Descargando recursos NLTK...", 10),
                ("Cargando dataset 20 Newsgroups...", 25),
                ("Creando subconjunto balanceado...", 40),
                ("Preprocesando textos...", 60),
                ("Entrenando modelo Naive Bayes...", 80),
                ("Evaluando rendimiento...", 90),
                ("Finalizando...", 100)
            ]

            for status, progress in steps:
                self.root.after(0, lambda s=status, p=progress: self.update_progress(s, p))
                if progress == 10:
                    download_nltk_resources()
                elif progress == 25:
                    df = self.classifier.load_20newsgroups_dataset()
                    if df is None:
                        self.root.after(0, lambda: self.training_error("No se pudo cargar el dataset"))
                        return
                elif progress == 40:
                    self.df_trained = self.classifier.create_balanced_subset(df, samples_per_category=100)
                elif progress == 80:
                    self.X_test, self.y_test, self.y_pred, self.accuracy = self.classifier.train_model(self.df_trained)
                time.sleep(0.5)

            self.root.after(0, self.training_completed)

        except Exception as e:
            self.root.after(0, lambda: self.training_error(str(e)))

    def update_progress(self, status, progress):
        """
        Actualiza la barra de progreso y el mensaje de estado.

        Args:
            status: Mensaje de estado (str)
            progress: Valor de progreso (0-100)
        """
        self.status_var.set(status)
        self.progress_var.set(progress)

    def training_completed(self):
        """
        Acciones al completar el entrenamiento del modelo.
        """
        self.model_trained = True
        self.status_var.set(f"Modelo entrenado ✅ - Exactitud: {self.accuracy:.1%}")
        self.train_btn.config(state='normal', text="🔄 Re-entrenar Modelo")
        self.classify_btn.config(state='normal')

        messagebox.showinfo("Éxito",
                           f"¡Modelo entrenado exitosamente!\n\n"
                           f"Exactitud: {self.accuracy:.2%}\n"
                           f"Ya puede clasificar textos.")

    def training_error(self, error_msg):
        """
        Maneja errores durante el entrenamiento.

        Args:
            error_msg: Mensaje de error (str)
        """
        self.status_var.set("Error en entrenamiento ❌")
        self.train_btn.config(state='normal')
        self.progress_var.set(0)
        messagebox.showerror("Error de Entrenamiento", f"Error durante el entrenamiento:\n\n{error_msg}")

    def classify_text(self):
        """
        Clasifica el texto ingresado y muestra los resultados.
        """
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo primero.")
            return

        text = self.text_input.get('1.0', tk.END).strip()
        if len(text) < 20:
            messagebox.showwarning("Advertencia", "Por favor ingrese un texto más largo (mínimo 20 caracteres).")
            return

        try:
            prediction, probabilities = self.classifier.predict_single_text(text)
            if isinstance(prediction, str) and prediction.startswith("Error"):
                messagebox.showerror("Error", prediction)
                return

            self.result_label.config(text=f"🎯 Categoría predicha: {prediction}")
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (category, prob) in enumerate(sorted_probs, 1):
                bar_length = int(prob * 20)
                progress_bar = "█" * bar_length + "░" * (20 - bar_length)
                self.results_tree.insert('', 'end', values=(
                    i, category, f"{prob:.1%}", progress_bar + f" {prob:.1%}"
                ))

        except Exception as e:
            messagebox.showerror("Error", f"Error durante la clasificación:\n{str(e)}")

    def show_model_metrics(self):
        """
        Muestra métricas detalladas del modelo con estadísticas y gráficos.
        """
        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo primero.")
            return

        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("📊 Métricas y Análisis del Modelo")
        metrics_window.geometry("1200x800")
        metrics_window.state('zoomed')

        notebook = ttk.Notebook(metrics_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="📈 Métricas del Modelo")

        metrics_text = scrolledtext.ScrolledText(metrics_frame, wrap=tk.WORD, font=('Courier', 10))
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        metrics_info = f"""
MÉTRICAS DEL MODELO CLASIFICADOR
================================

Algoritmo: Multinomial Naive Bayes
Dataset: 20 Newsgroups
Fecha de entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXACTITUD GENERAL: {accuracy:.2%}

REPORTE DETALLADO DE CLASIFICACIÓN:
{report}

INFORMACIÓN ADICIONAL:
- Total de muestras de entrenamiento: {len(self.df_trained):,}
- Total de muestras de prueba: {len(self.y_test):,}
- Número de categorías: {len(set(self.y_test))}
- Vectorización: TF-IDF (max 5000 features)
- N-gramas: 1-2
- Preprocesamiento: Stemming + eliminación de stopwords
"""
        metrics_text.insert('1.0', metrics_info)
        metrics_text.config(state='disabled')

        graphics_frame = ttk.Frame(notebook)
        notebook.add(graphics_frame, text="📊 Análisis Gráfico")

        try:
            fig = self.classifier.generate_analysis_plots(self.y_test, self.y_pred)
            canvas = FigureCanvasTkAgg(fig, master=graphics_frame)
            canvas.draw()
            canvas_frame = ttk.Frame(graphics_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            ttk.Button(graphics_frame, text="💾 Guardar Gráficas", command=lambda: self.save_plots(fig)).pack(pady=5)
        except Exception as e:
            ttk.Label(graphics_frame, text=f"Error generando gráficas: {str(e)}").pack(expand=True)

    def show_about(self):
        """
        Muestra información sobre la aplicación.
        """
        about_text = """
🤖 Clasificador de Noticias con IA

Trabajo Computacional N°1
Curso: Inteligencia Artificial
Grupo 10

📊 Características:
• Algoritmo: Multinomial Naive Bayes
• Dataset: 20 Newsgroups de scikit-learn
• Interfaz gráfica con Tkinter
• Preprocesamiento avanzado de texto
• Vectorización TF-IDF

🎯 Capacidades:
• Clasificación de textos en 20 categorías
• Análisis de probabilidades por categoría
• Carga de archivos de texto
• Métricas detalladas del modelo

💡 Desarrollado con Python y scikit-learn
"""
        messagebox.showinfo("Acerca de", about_text)

    def save_plots(self, fig):
        """
        Guarda los gráficos generados en un archivo.

        Args:
            fig: Figura de matplotlib a guardar
        """
        try:
            file_path = filedialog.asksaveasfilename(
                title="Guardar gráficas",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
            )

            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráficas guardadas en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{str(e)}")

    def run(self):
        """
        Inicia el bucle principal de la interfaz gráfica.
        """
        self.root.mainloop()