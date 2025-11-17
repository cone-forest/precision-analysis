import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision analysis")
        root.geometry('800x600')
        
        self.file1_path = None
        self.file2_path = None
        
        self.create_widgets()
    
    def create_widgets(self):
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10)
        
        self.btn_file1 = ttk.Button(
            file_frame,
            text="Загрузить файл 1",
            command=self.load_file1
        )
        self.btn_file1.pack(side=tk.LEFT, padx=5)
        
        self.btn_file2 = ttk.Button(
            file_frame,
            text="Загрузить файл 2",
            command=self.load_file2
        )
        self.btn_file2.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.label_file1 = ttk.Label(btn_frame, text="Файл 1 не выбран")
        self.label_file1.pack(side=tk.LEFT, padx=5)
        
        self.label_file2 = ttk.Label(btn_frame, text="Файл 2 не выбран")
        self.label_file2.pack(side=tk.RIGHT, padx=5)
    
    def load_file1(self):
        return
    
    def load_file2(self):
        return

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()