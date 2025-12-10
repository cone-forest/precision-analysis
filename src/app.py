import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from functions_call import get_error_data
import sys
import tempfile
import os

try:
    from perlin_noise import process_perlin_file
    HAS_PERLIN = True
except ImportError:
    print("Файл perlin_noise.py не найден")
    HAS_PERLIN = False

try:
    from gause_noise import process_gaussian_file
    HAS_GAUSSIAN = True
except ImportError:
    print("Файл gause_noise.py не найден")
    HAS_GAUSSIAN = False


class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision analysis")
        # root.state('zoomed') # comment on linux
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = tk.Frame(root)
        main_frame.pack(fill='both', expand=True)

        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        main_canvas = tk.Canvas(main_frame, yscrollcommand=scrollbar.set)
        main_canvas.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=main_canvas.yview)
        self.content = tk.Frame(main_canvas)
        self.content.pack(fill=tk.BOTH, expand=True)
        self.canvas_window = main_canvas.create_window((0, 0), window=self.content, anchor="nw")
        
        def on_content_configure(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        self.content.bind("<Configure>", on_content_configure)

        def on_canvas_configure(event):
            main_canvas.itemconfig(self.canvas_window, width=event.width)
        main_canvas.bind("<Configure>", on_canvas_configure)


        self.file1_path = None
        self.file2_path = None
        
        self.tsai_lenz_var = tk.IntVar()
        self.park_martin_var = tk.IntVar()
        self.daniilidis_var = tk.IntVar()
        self.li_wang_wu_var = tk.IntVar()
        self.shah_var = tk.IntVar()

        self.first_input = True

        self.add_noise_var = tk.BooleanVar(value=False)
        self.noise_type_var = tk.StringVar()
        self.noise_level_var = tk.DoubleVar(value=0.5)
        
        self.noise_settings_frame = None
        
        self.available_noise_types = []
        if HAS_PERLIN:
            self.available_noise_types.append("Perlin Noise")
        if HAS_GAUSSIAN:
            self.available_noise_types.append("Gaussian Noise")
        
        if self.available_noise_types:
            self.noise_type_var.set(self.available_noise_types[0])
        else:
            self.add_noise_var.set(False)
        
        self.create_widgets()
    

    def on_closing(self):
        plt.close('all')
        self.root.destroy()    
        sys.exit(0)
    

    def create_widgets(self):
        file_frame = ttk.Frame(self.content)
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

        btn_frame = ttk.Frame(self.content)
        btn_frame.pack(pady=10)
        self.label_file1 = ttk.Label(btn_frame, text="Файл 1 не выбран")
        self.label_file1.pack(side=tk.LEFT, padx=5)
        
        self.label_file2 = ttk.Label(btn_frame, text="Файл 2 не выбран")
        self.label_file2.pack(side=tk.RIGHT, padx=5)

        tsai_lenz_check = tk.Checkbutton(self.content, text="tsai-lenz", variable=self.tsai_lenz_var)
        tsai_lenz_check.pack()

        park_martin_check = tk.Checkbutton(self.content, text="park-martin", variable=self.park_martin_var)
        park_martin_check.pack()

        daniilidis_check = tk.Checkbutton(self.content, text="daniilidis", variable=self.daniilidis_var)
        daniilidis_check.pack()

        li_wang_wu_check = tk.Checkbutton(self.content, text="li-wang-wu", variable=self.li_wang_wu_var)
        li_wang_wu_check.pack()

        shah_check = tk.Checkbutton(self.content, text="shah", variable=self.shah_var)
        shah_check.pack()

        noise_check = tk.Checkbutton(
            self.content, 
            text="Добавить шум к данным", 
            variable=self.add_noise_var,
            command=self.toggle_noise_settings
        )
        noise_check.pack(pady=(10, 5))
        
        self.noise_settings_frame = ttk.Frame(self.content)
        
        noise_type_label = ttk.Label(self.noise_settings_frame, text="Тип шума:")
        noise_type_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        
        noise_type_combo = ttk.Combobox(
            self.noise_settings_frame,
            textvariable=self.noise_type_var,
            values=self.available_noise_types,
            state="readonly",
            width=20
        )
        noise_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        if not self.available_noise_types:
            noise_type_combo.config(state="disabled")
            ttk.Label(self.noise_settings_frame, 
                     text="Файлы шума не найдены", 
                     foreground="red").grid(row=0, column=2, padx=10, pady=5)
        
        noise_level_label = ttk.Label(self.noise_settings_frame, text="Уровень шума:")
        noise_level_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        
        noise_level_scale = ttk.Scale(
            self.noise_settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.noise_level_var,
            orient="horizontal",
            length=200
        )
        noise_level_scale.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.noise_level_value_label = ttk.Label(self.noise_settings_frame, text="0.50")
        self.noise_level_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        def update_noise_label(*args):
            self.noise_level_value_label.config(text=f"{self.noise_level_var.get():.2f}")
        
        self.noise_level_var.trace_add("write", update_noise_label)
        
        self.toggle_noise_settings()

        self.submit_btn = ttk.Button(
            self.content,
            text="Выполнить",
            command=self.run_methods
        )
        self.submit_btn.pack(pady=10)
    

    def toggle_noise_settings(self):
        if self.add_noise_var.get() and self.available_noise_types:
            self.noise_settings_frame.pack(pady=10, padx=10, fill=tk.X)
        else:
            self.noise_settings_frame.pack_forget()

    def create_plot_areas(self):
        translation_label = ttk.Label(self.content, text="Translation Errors", font=('Arial', 12, 'bold'))
        translation_label.pack(pady=(10, 5))
        
        self.create_scrollable_plot_area("translation")
        
        rotation_label = ttk.Label(self.content, text="Rotation Errors", font=('Arial', 12, 'bold'))
        rotation_label.pack(pady=(20, 5))
        
        self.create_scrollable_plot_area("rotation")


    def create_scrollable_plot_area(self, area_type):
        container = ttk.Frame(self.content)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container, height=300)
        h_scrollbar = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(xscrollcommand=h_scrollbar.set)
        
        plots_frame = ttk.Frame(canvas)
        
        canvas.create_window((0, 0), window=plots_frame, anchor="nw")
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        def update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        plots_frame.bind("<Configure>", update_scrollregion)
        
        if area_type == "translation":
            self.translation_canvas = canvas
            self.translation_plots_frame = plots_frame
            self.translation_update_scroll = update_scrollregion
        else:
            self.rotation_canvas = canvas
            self.rotation_plots_frame = plots_frame
            self.rotation_update_scroll = update_scrollregion
    

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
    
    
    def create_plots(self, data, flag):
        plots = []
        if flag == 't':
            for name in data:
                metrics = ["mean", "median", "rmse", "p95", "max"]
                translation = [data[name][m] for m in metrics]
                x = np.arange(len(metrics))
                width = 0.35
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(x - width/2, translation, width, label='translation')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.set_xlabel('Метрики')
                ax.set_ylabel('Ошибка')
                ax.set_title(f'Метод {name} - Translation')
                plots.append(fig)
        else:
            for name in data:
                metrics = ["mean", "median", "rmse", "p95", "max"]
                rotation = [data[name][m] for m in metrics]
                x = np.arange(len(metrics))
                width = 0.35
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(x + width/2, rotation, width, label='rotation') 
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.set_xlabel('Метрики')
                ax.set_ylabel('Ошибка')
                ax.set_title(f'Метод {name} - Rotation')
                plots.append(fig)
        return plots
    

    def display_plots(self, figures, area_type):
        if area_type == "translation":
            plots_frame = self.translation_plots_frame
            canvas = self.translation_canvas
            update_scroll = self.translation_update_scroll
        else:
            plots_frame = self.rotation_plots_frame
            canvas = self.rotation_canvas
            update_scroll = self.rotation_update_scroll
        
        for widget in plots_frame.winfo_children():
            widget.destroy()
        
        for i, fig in enumerate(figures):
            plot_frame = ttk.Frame(plots_frame, relief='solid', borderwidth=1)
            plot_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=False)
            
            title_label = ttk.Label(plot_frame, text=f"График {i+1}", font=('Arial', 10, 'bold'))
            title_label.pack(pady=5)
            
            canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_plot.draw()
            canvas_widget = canvas_plot.get_tk_widget()
            canvas_widget.config(width=500, height=200)
            canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            save_btn = ttk.Button(
                plot_frame,
                text="Сохранить",
                command=lambda f=fig, n=i: self.save_plot(f, n)
            )
            save_btn.pack(pady=5)
        
        plots_frame.update_idletasks()
        update_scroll()
        canvas.xview_moveto(0)
    

    def save_plot(self, fig, plot_number):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title=f"Сохранить график {plot_number + 1}"
        )
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Успех", f"График сохранен как {filename}")
    


    def run_methods(self):
        methods = [
            name for name, var in [
                ("tsai-lenz", self.tsai_lenz_var),
                ("park-martin", self.park_martin_var),
                ("daniilidis", self.daniilidis_var),
                ("li-wang-wu", self.li_wang_wu_var),
                ("shah", self.shah_var)
            ] if var.get()
        ]
        
        if not self.file1_path or not self.file2_path:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите оба файла!")
            return
        
        if not methods:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите хотя бы один метод!")
            return
        
        file_to_compare = self.file2_path
        temp_file_path = None
        
        if self.add_noise_var.get() and self.available_noise_types:
            try:
                # Файлы с шумом будут сохраняться в другой файл
                # То есть, не получится сделать так, чтобы шум был применен
                # к уже якобы измененному файлу
                # Изначальный файл не меняется, для шума создается отдельный
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_noisy.txt', delete=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                noise_level = self.noise_level_var.get()
                noise_type = self.noise_type_var.get()
                
                if noise_type == "Perlin Noise" and HAS_PERLIN:
                    print(noise_level)
                    pos_scale = 20 * noise_level
                    rot_scale = 0.5 * noise_level
                    
                    process_perlin_file(
                        input_file=self.file2_path,
                        output_file=temp_file_path,
                        pos_scale=pos_scale,
                        rot_scale=rot_scale
                    )
                    messagebox.showinfo("Информация", 
                                      f"Шум Перлина применён (уровень: {noise_level:.2f})")
                    
                elif noise_type == "Gaussian Noise" and HAS_GAUSSIAN:
                    pos_std = 20 * noise_level
                    rot_std = 0.5 * noise_level
                    
                    process_gaussian_file(
                        input_file=self.file2_path,
                        output_file=temp_file_path,
                        pos_std=pos_std,
                        rot_std=rot_std
                    )
                    messagebox.showinfo("Информация", 
                                      f"Гауссовский шум применён (уровень: {noise_level:.2f})")
                else:
                    messagebox.showerror("Ошибка", "Выбранный тип шума недоступен")
                    return
                
                file_to_compare = temp_file_path
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось применить шум: {str(e)}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return
        try:
            t_data, r_data = get_error_data(methods, self.file1_path, file_to_compare)
            
            messagebox.showinfo("Информация", "Файлы загружены! Генерация графиков...")
            
            sample_plot_t = self.create_plots(t_data, 't')
            sample_plot_r = self.create_plots(r_data, 'r')
            if self.first_input:
                self.create_plot_areas()
                self.first_input = False

            self.display_plots(sample_plot_t, "translation")
            self.display_plots(sample_plot_r, "rotation")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе данных: {str(e)}")
        finally:
            # Удаление временного файла, если он был создан
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()