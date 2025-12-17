import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal

def gaussian_noise_generator(input_file, pos_std=0.5, rot_std=0.05, 
                           correlation=0.1, seed=42):
    """
    Гауссовский шум
    """
    np.random.seed(seed)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                print(f"Пропуск некорректной строки {line_num+1}")
                continue
                
            try:
                frame_id = parts[0]
                x, y, z, rx, ry, rz = map(float, parts[1:7])
                
                # Ковариационные матрицы SciPy
                pos_cov = pos_std**2 * np.array([
                    [1.0, correlation, correlation],
                    [correlation, 1.0, correlation],
                    [correlation, correlation, 1.0]
                ])
                rot_cov = rot_std**2 * np.array([
                    [1.0, correlation, correlation],
                    [correlation, 1.0, correlation],
                    [correlation, correlation, 1.0]
                ])
                
                # SciPy multivariate_normal — ЧИСТЫЙ ГАУСС
                pos_noise = multivariate_normal(mean=[0,0,0], cov=pos_cov).rvs()
                rot_noise = multivariate_normal(mean=[0,0,0], cov=rot_cov).rvs()
                
                noisy_x = x + pos_noise[0]
                noisy_y = y + pos_noise[1]
                noisy_z = z + pos_noise[2]
                noisy_rx = rx + rot_noise[0]
                noisy_ry = ry + rot_noise[1]
                noisy_rz = rot_noise[2]
                
                yield f"{frame_id} {noisy_x:.6f} {noisy_y:.6f} {noisy_z:.6f} " \
                      f"{noisy_rx:.6f} {noisy_ry:.6f} {noisy_rz:.6f}"
                      
            except (ValueError, IndexError) as e:
                print(f"Ошибка в строке {line_num+1}: {e}")
                continue


def process_gaussian_file(input_file, output_file, pos_std=0.0, rot_std=0.0, 
                         correlation=0.1, seed=42):
    """Обрабатывает файл построчно и сохраняет результат"""
    with open(output_file, 'w') as out_f:
        for noisy_line in gaussian_noise_generator(input_file, pos_std, rot_std, 
                                                 correlation, seed):
            out_f.write(noisy_line + '\n')

"""
Как использовать 
Вызываешь команду process_gaussian_file
каждое поле отвечает за:
input_file - входной фаил
output_file - выходной фаил
pos_std - задает стандарное отклонение положения (при нуле не влияет на исходные данные) [0.0 - 20]
rot_std - задает станлартное отклонение углов (при нуле не влияет на исходные данные) [0.0 - 0.5]
corrlation -  корреляция между осями (0=независимые, 1=идентичный шум) [0.0 - 1.0]
seed - сид как в майне для регерации псевдослучайных чисел 
"""