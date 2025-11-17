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
        
        self.submit_btn = ttk.Button(
            self.root,
            text="Выполнить",
            command=self.run_methods
        )
        self.submit_btn.pack(pady=10)
    

    def load_file1(self):
        file_path = filedialog.askopenfilename(title="Выберите первый файл")
        if file_path:
            self.file1_path = file_path
            self.label_file1.config(text=f"Файл 1: {file_path.split('/')[-1]}")
    

    def load_file2(self):
        file_path = filedialog.askopenfilename(title="Выберите второй файл")
        if file_path:
            self.file2_path = file_path
            self.label_file2.config(text=f"Файл 2: {file_path.split('/')[-1]}")
    
    def run_methods(self):
        data = {
            "file1": self.file1_path,
            "file2": self.file2_path,
        }
        # parser(data['file1'])
        # parser(data['file2'])
        # method_1(file1, file2)
        # method_2(file1, file2)
        # method_3(file1, file2)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()