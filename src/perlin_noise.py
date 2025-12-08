import numpy as np
import math
from pathlib import Path

class PerlinNoise:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.p = np.arange(512)
        np.random.shuffle(self.p)
        self.p = np.concatenate((self.p, self.p))

    def fade(self, t): return 6 * t**5 - 15 * t**4 + 10 * t**3 # функция сглаживания
    def lerp(self, a, b, t): return a + t * (b - a) # линейная интерполяция 
    
    def grad(self, hash_val, x, y, z=0): # градиентный вектор
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in [12, 14] else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y=0, z=0):
        X = int(math.floor(x)) & 255; Y = int(math.floor(y)) & 255; Z = int(math.floor(z)) & 255
        x -= math.floor(x); y -= math.floor(y); z -= math.floor(z)
        u, v, w = self.fade(x), self.fade(y), self.fade(z)
        A = self.p[X] + Y; AA = self.p[A] + Z; AB = self.p[A+1] + Z
        B = self.p[X+1] + Y; BA = self.p[B] + Z; BB = self.p[B+1] + Z
        return self.lerp(
            self.lerp(self.lerp(self.grad(self.p[AA], x, y, z), self.grad(self.p[BA], x-1, y, z), u),
            self.lerp(self.grad(self.p[AB], x, y-1, z), self.grad(self.p[BB], x-1, y-1, z), u), v),
            self.lerp(self.lerp(self.grad(self.p[AA+1], x, y, z-1), self.grad(self.p[BA+1], x-1, y, z-1), u),
                     self.lerp(self.grad(self.p[AB+1], x, y-1, z-1), self.grad(self.p[BB+1], x-1, y-1, z-1), u), v), w
        )

    def fbm(self, x, y=0, z=0, octaves=4, persistence=0.5, frequency=1.0, scale=0.1):
        total, amplitude, max_amp, freq = 0.0, 1.0, 0.0, frequency
        for _ in range(octaves):
            total += self.noise(x * freq, y * freq, z * freq) * amplitude
            max_amp += amplitude; amplitude *= persistence; freq *= 2
        return (total / max_amp - 0.5) * 2 * scale

def noisy_robot_generator(input_file, pos_scale=5.0, rot_scale=0.05, octaves=4, persistence=0.5, seed=42):
    """
    ГЕНЕРАТОР: построчно читает файл и зашумляет НА ЛЕТУ
    yield возвращает каждую зашумленную строку сразу
    """
    noise = PerlinNoise(seed=seed)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                print(f"Пропуск некорректной строки {line_num+1}")
                continue
                
            try:
                # Парсим: номер X Y Z RX RY RZ
                frame_id = parts[0]
                x, y, z, rx, ry, rz = map(float, parts[1:7])
                
                # вычисляем шум для этой строки
                t = line_num * 0.1  # время = номер строки * коэффициент
                noise_x = noise.fbm(t, 0, 0, octaves, persistence, scale=pos_scale)
                noise_y = noise.fbm(t, 100, 0, octaves, persistence, scale=pos_scale)
                noise_z = noise.fbm(t, 200, 0, octaves, persistence, scale=pos_scale)
                noise_rx = noise.fbm(t, 300, 0, octaves, persistence, scale=rot_scale)
                noise_ry = noise.fbm(t, 400, 0, octaves, persistence, scale=rot_scale)
                noise_rz = noise.fbm(t, 500, 0, octaves, persistence, scale=rot_scale)
                
                # зашумленные значения
                noisy_x = x + noise_x
                noisy_y = y + noise_y
                noisy_z = z + noise_z
                noisy_rx = rx + noise_rx
                noisy_ry = ry + noise_ry
                noisy_rz = rz + noise_rz
                
                # возвращаем готовую строку
                yield f"{frame_id} {noisy_x:.6f} {noisy_y:.6f} {noisy_z:.6f} " \
                      f"{noisy_rx:.6f} {noisy_ry:.6f} {noisy_rz:.6f}"
                      
            except (ValueError, IndexError) as e:
                print(f"Ошибка в строке {line_num+1}: {e}")
                continue

def process_robot_file(input_file, output_file, pos_scale=0, rot_scale=0, octaves=4, persistence=0.5, seed=42):
    """Обрабатывает файл построчно и сохраняет результат"""
    count = 0
    with open(output_file, 'w') as out_f:
        for noisy_line in noisy_robot_generator(input_file, pos_scale, rot_scale, octaves, persistence, seed):
            out_f.write(noisy_line + '\n')
            count += 1

"""
Как использовать 
Вызываешь команду process_robot_file
каждое поле отвечает за:
input_file - входной фаил
output_file - выходной фаил
pos_scale - задает отклонение положении (при нуле не влияет на исходные данные) [0.0 - 20]
rot_scale - задает отклонение в углах (при нуле не влияет на исходные данные) [0.0 - 0.5]
octaves - детализация [2 - 8] 
persistence - затухание  [0.2 - 0.8], где 0.4-0.5 — натуральное движение
seed - сид как в майне для регерации псевдослучайных чисел 
"""
