import math
import matplotlib.pyplot as plt
import numpy as np

class Function:
    def __init__(self):
        self.cash = {}
        self.cash_g = {}
    
    def f(self, x, t, lvl=0):
        h = 1e-6
        if lvl == 0:
            if t not in self.cash:
                self.cash[t] = {}
            if x not in self.cash[t]:
                try:
                    result = math.tan(x) ** 2 - t * x
                    # Ограничиваем очень большие значения
                    if abs(result) > 100:
                        result = math.copysign(100, result)
                    self.cash[t][x] = result
                except (ValueError, OverflowError):
                    self.cash[t][x] = math.copysign(100, 1)
            return self.cash[t][x]
        elif lvl == 1:
            # Более безопасное вычисление производной
            try:
                derivative = (self.f(x + h, t) - self.f(x - h, t)) / (2.0 * h)
                # Если производная слишком мала, возвращаем минимальное значение
                if abs(derivative) < 1e-10:
                    return 1e-10 if derivative >= 0 else -1e-10
                return derivative
            except:
                return 1e-10
        elif lvl == 2:
            try:
                return (self.f(x + h, t, 1) - self.f(x - h, t, 1)) / (2.0 * h)
            except:
                return 0
        else:
            return 0
    
    def g(self, x, t, lvl=0):
        h = 1e-6
        if lvl == 0:
            if t not in self.cash_g:
                self.cash_g[t] = {}
            if x not in self.cash_g[t]:
                try:
                    self.cash_g[t][x] = math.tan(x) ** 2 / t
                except:
                    self.cash_g[t][x] = 0
            return self.cash_g[t][x]
        elif lvl == 1:
            return (self.g(x + h, t) - self.g(x - h, t)) / (2.0 * h)
        elif lvl == 2:
            return (self.g(x + h, t, 1) - self.g(x - h, t, 1)) / (2.0 * h)
        else:
            return 0
    
    def tangent(self, x, x0, t):
        return self.f(x0, t) + self.f(x0, t, 1) * (x - x0)
    
    def tangent_zero(self, x0, t):
        try:
            f_val = self.f(x0, t)
            f_derivative = self.f(x0, t, 1)
            
            # Проверяем, что производная не слишком мала
            if abs(f_derivative) < 1e-10:
                # Если производная слишком мала, используем небольшой шаг
                return x0 - 0.1 * math.copysign(1, f_val)
            
            new_x = x0 - f_val / f_derivative
            
            # Проверяем, что новое значение не слишком далеко от старого
            if abs(new_x - x0) > 2.0:
                return x0 - 0.1 * math.copysign(1, f_val)
                
            return new_x
            
        except ZeroDivisionError:
            # Если все же произошло деление на ноль
            return x0 - 0.1 * math.copysign(1, self.f(x0, t))


def safe_f(x, t):
    """Безопасное вычисление функции с обработкой особых точек"""
    try:
        # Избегаем точек разрыва tan(x)
        if abs(x - math.pi/2) % math.pi < 0.1:
            return math.copysign(10, math.tan(x) if abs(math.tan(x)) < 100 else 100)
        
        result = math.tan(x) ** 2 - t * x
        
        # Ограничиваем очень большие значения для лучшего отображения
        if abs(result) > 10:
            return math.copysign(10, result)
        return result
    except (ValueError, OverflowError):
        return math.copysign(10, 1)


def main():
    f = Function()
    error = 0.0001
    
    # Более безопасные начальные приближения
    a = -1.4  # Вместо -1.57075 (слишком близко к -π/2)
    b = 1.4   # Вместо 1.57075 (слишком близко к π/2)
    
    t = 1.0
    
    # Создаем фигуру
    plt.figure(figsize=(12, 8))
    
    # Генерируем точки для графика, избегая разрывов
    x_range = np.linspace(-2, 2, 1000)
    y_values = [safe_f(x, t) for x in x_range]
    
    plt.plot(x_range, y_values, 'b-', linewidth=2, label='f(x) = tan²(x) - x')
    
    # Рисуем координатные оси
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    
    # Сетка
    plt.grid(True, alpha=0.3)
    
    # Метод касательных (Ньютона)
    # Выбираем начальное приближение в зависимости от знака t
    answer = b if t > 0 else a
    iterations = 0
    max_iterations = 50
    
    print("Метод касательных:")
    print(f"Начальное приближение: x₀ = {answer}")
    print(f"f(x₀) = {f.f(answer, t)}")
    print(f"f'(x₀) = {f.f(answer, t, 1)}")
    
    # Сохраняем точки итераций для отображения
    iteration_points = [(answer, f.f(answer, t))]
    
    success = False
    
    while iterations < max_iterations:
        f_val = f.f(answer, t)
        
        print(f"Итерация {iterations}: x = {answer:.6f}, f(x) = {f_val:.6f}")
        
        # Проверяем достигли ли мы нужной точности
        if abs(f_val) <= error:
            success = True
            break
        
        # Сохраняем предыдущее значение
        prev_answer = answer
        
        try:
            # Вычисляем новое приближение
            answer = f.tangent_zero(answer, t)
            
            # Проверяем на расхождение
            if abs(answer - prev_answer) > 2.0 or not math.isfinite(answer):
                print("Метод расходится, пробуем другое начальное приближение")
                answer = prev_answer + 0.1 * math.copysign(1, -f_val)
            
            # Сохраняем точку итерации
            iteration_points.append((answer, f.f(answer, t)))
            
            # Рисуем касательную в текущей точке
            tangent_range = 0.3
            tangent_x = np.linspace(prev_answer - tangent_range, prev_answer + tangent_range, 50)
            tangent_y = [f.tangent(x, prev_answer, t) for x in tangent_x]
            plt.plot(tangent_x, tangent_y, 'r-', alpha=0.6, linewidth=1)
            
            # Рисуем линию к новой точке
            plt.plot([prev_answer, answer], [f.f(prev_answer, t), f.f(answer, t)], 'g--', alpha=0.6, linewidth=1)
            
        except Exception as e:
            print(f"Ошибка на итерации {iterations}: {e}")
            break
        
        iterations += 1
    
    # Отображаем все точки итераций
    if iteration_points:
        iter_x, iter_y = zip(*iteration_points)
        plt.plot(iter_x, iter_y, 'go-', markersize=6, linewidth=2, alpha=0.8, label='Итерации метода')
        
        # Отмечаем начальную точку и конечный корень
        plt.plot(iter_x[0], iter_y[0], 'go', markersize=8, label='Начальная точка')
        plt.plot(iter_x[-1], iter_y[-1], 'ro', markersize=10, label=f'Корень: x = {iter_x[-1]:.6f}')
    
    plt.title(f"Метод касательных (Ньютона) для f(x) = tan²(x) - x")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    # Устанавливаем разумные пределы по осям
    plt.xlim(-2, 2)
    plt.ylim(-5, 10)
    
    # Добавляем информацию о результате
    if success and iteration_points:
        textstr = f'Успешно!\nКорень: x = {iter_x[-1]:.8f}\nf(x) = {iter_y[-1]:.2e}\nИтераций: {iterations}'
        color = 'lightgreen'
    else:
        textstr = f'Не достигнута точность\nЛучшее x = {iter_x[-1]:.8f}\nf(x) = {iter_y[-1]:.2e}\nИтераций: {iterations}'
        color = 'lightcoral'
    
    props = dict(boxstyle='round', facecolor=color, alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nРезультат:")
    if success:
        print(f"answer: f({iter_x[-1]}) = {iter_y[-1]} ~ 0 ; when t = {t}")
    else:
        print(f"Лучшее приближение: f({iter_x[-1]}) = {iter_y[-1]} ; when t = {t}")
    print(f"Количество итераций: {iterations}")


if __name__ == "__main__":
    main()