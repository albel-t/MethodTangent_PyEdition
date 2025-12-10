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
                # f(x, t) = tan(x)^2 - t*x
                result = math.tan(x) ** 2 - t * x
                self.cash[t][x] = result
            return self.cash[t][x]
        elif lvl == 1:
            # Производная f(x)
            try:
                derivative = (self.f(x + h, t) - self.f(x - h, t)) / (2.0 * h)
                if abs(derivative) < 1e-10:
                    return 1e-10 if derivative >= 0 else -1e-10
                return derivative
            except:
                return 1e-10
        elif lvl == 2:
            # Вторая производная f(x)
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
                # g(x, t) = tan(x)^2 / t  (для метода простых итераций)
                result = math.tan(x) ** 2 / t
                self.cash_g[t][x] = result
            return self.cash_g[t][x]
        elif lvl == 1:
            # Производная g(x)
            try:
                derivative = (self.g(x + h, t) - self.g(x - h, t)) / (2.0 * h)
                if abs(derivative) < 1e-10:
                    return 1e-10 if derivative >= 0 else -1e-10
                return derivative
            except:
                return 1e-10
        elif lvl == 2:
            # Вторая производная g(x)
            try:
                return (self.g(x + h, t, 1) - self.g(x - h, t, 1)) / (2.0 * h)
            except:
                return 0
        else:
            return 0
    
    def tangent(self, x, x0, t):
        # Уравнение касательной: f(x0) + f'(x0)*(x - x0)
        return self.f(x0, t) + self.f(x0, t, 1) * (x - x0)
    
    def tangent_zero(self, x0, t):
        # Метод Ньютона: x_new = x0 - f(x0)/f'(x0)
        try:
            f_val = self.f(x0, t)
            f_derivative = self.f(x0, t, 1)
            
            if abs(f_derivative) < 1e-10:
                return x0 - 0.1 * math.copysign(1, f_val)
            
            new_x = x0 - f_val / f_derivative
            
            if abs(new_x - x0) > 2.0:
                return x0 - 0.1 * math.copysign(1, f_val)
                
            return new_x
            
        except ZeroDivisionError:
            return x0 - 0.1 * math.copysign(1, self.f(x0, t))


def safe_f(x, t):
    """Безопасное вычисление f(x, t) = tan(x)^2 - t*x"""
    try:
        # Избегаем точек разрыва tan(x) в π/2 + kπ
        if abs(abs(x) - math.pi/2) % math.pi < 0.1:
            return math.copysign(10, math.tan(x))
        
        result = math.tan(x) ** 2 - t * x
        
        if abs(result) > 10:
            return math.copysign(10, result)
        return result
    except (ValueError, OverflowError):
        return math.copysign(10, 1)


def safe_g(x, t):
    """Безопасное вычисление g(x, t) = tan(x)^2 / t"""
    try:
        if abs(abs(x) - math.pi/2) % math.pi < 0.1:
            return math.copysign(10, math.tan(x))
        
        result = math.tan(x) ** 2 / t
        
        if abs(result) > 10:
            return math.copysign(10, result)
        return result
    except (ValueError, OverflowError):
        return math.copysign(10, 1)


def check_iteration_convergence(x, t, func):
    """Проверка условия сходимости метода простых итераций"""
    try:
        # Условие сходимости: |g'(x)| < 1
        g_derivative = func.g(x, t, 1)
        return abs(g_derivative) < 1
    except:
        return False


def main():
    CELLSSCALE = 10
    f = Function()
    error = 0.0001
    
    a = -1.57075
    b = 1.57075
    a += error
    b -= error
    
    t = 1.0
    
    print("=" * 60)
    print("МЕТОД КАСАТЕЛЬНЫХ (НЬЮТОНА) И МЕТОД ПРОСТЫХ ИТЕРАЦИЙ")
    print("=" * 60)
    print(f"f(x, t) = tan²(x) - {t}·x")
    print(f"g(x, t) = tan²(x) / {t}")
    print(f"Точность: {error}")
    print()
    
    # Ввод t
    try:
        t_input = input("Введите значение t (или Enter для t=1): ")
        if t_input.strip():
            t = float(t_input)
    except ValueError:
        print(f"Некорректный ввод, используется t = {t}")
    
    print(f"\nИспользуется t = {t}")
    print()
    
    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # =========== МЕТОД КАСАТЕЛЬНЫХ (ЛЕВЫЙ ГРАФИК) ===========
    answer_tangent = b if t > 0 else a
    lastx = answer_tangent
    lasty = f.f(answer_tangent, t)
    
    # Генерируем точки для графика f(x)
    x_range_f = np.linspace(-math.pi/2 + 0.1, math.pi/2 - 0.1, 500)
    y_values_f = [safe_f(x, t) for x in x_range_f]
    
    ax1.plot(x_range_f, y_values_f, 'b-', linewidth=2, label=f'f(x) = tan²(x) - {t}·x')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    print("МЕТОД КАСАТЕЛЬНЫХ:")
    print("-" * 40)
    print(f"Начальное приближение: x₀ = {answer_tangent}")
    print(f"f(x₀) = {f.f(answer_tangent, t):.6f}")
    print(f"f'(x₀) = {f.f(answer_tangent, t, 1):.6f}")
    print()
    
    # Сохраняем точки итераций для метода касательных
    tangent_points = [(answer_tangent, f.f(answer_tangent, t))]
    tangent_iterations = 0
    tangent_success = False
    max_iterations = 50
    
    while tangent_iterations < max_iterations:
        f_val = f.f(answer_tangent, t)
        
        print(f"Итерация {tangent_iterations}: x = {answer_tangent:.8f}, f(x) = {f_val:.8f}")
        
        if abs(f_val) <= error:
            tangent_success = True
            break
        
        prev_answer = answer_tangent
        
        try:
            answer_tangent = f.tangent_zero(answer_tangent, t)
            
            if abs(answer_tangent - prev_answer) > 2.0 or not math.isfinite(answer_tangent):
                print("  Метод расходится, корректируем шаг...")
                answer_tangent = prev_answer + 0.1 * math.copysign(1, -f_val)
            
            tangent_points.append((answer_tangent, f.f(answer_tangent, t)))
            
            # Рисуем касательную
            tangent_range = 0.3
            tangent_x = np.linspace(prev_answer - tangent_range, prev_answer + tangent_range, 50)
            tangent_y = [f.tangent(x, prev_answer, t) for x in tangent_x]
            ax1.plot(tangent_x, tangent_y, 'r-', alpha=0.6, linewidth=1)
            
            # Рисуем линию к новой точке
            ax1.plot([prev_answer, answer_tangent], 
                    [f.f(prev_answer, t), f.f(answer_tangent, t)], 
                    'g--', alpha=0.6, linewidth=1)
            
        except Exception as e:
            print(f"  Ошибка: {e}")
            break
        
        tangent_iterations += 1
    
    # Отображаем точки итераций метода касательных
    if tangent_points:
        iter_x, iter_y = zip(*tangent_points)
        ax1.plot(iter_x, iter_y, 'go-', markersize=6, linewidth=2, alpha=0.8, label='Итерации метода')
        ax1.plot(iter_x[0], iter_y[0], 'go', markersize=8, label='Начальная точка')
        ax1.plot(iter_x[-1], iter_y[-1], 'ro', markersize=10, label=f'Корень: x = {iter_x[-1]:.6f}')
    
    ax1.set_title(f"Метод касательных (Ньютона)\nf(x) = tan²(x) - {t}·x")
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-5, 10)
    
    # =========== МЕТОД ПРОСТЫХ ИТЕРАЦИЙ (ПРАВЫЙ ГРАФИК) ===========
    print("\nМЕТОД ПРОСТЫХ ИТЕРАЦИЙ:")
    print("-" * 40)
    
    # Выбираем начальное приближение для метода простых итераций
    # Согласно C++ коду: answer = (t > 0 ? b+0.1 : a-0.1)
    answer_iter = b + 0.1 if t > 0 else a - 0.1
    
    # Проверяем условие сходимости в начальной точке
    convergence_checked = check_iteration_convergence(answer_iter, t, f)
    print(f"Начальное приближение: x₀ = {answer_iter}")
    print(f"g(x₀) = {f.g(answer_iter, t):.6f}")
    print(f"g'(x₀) = {f.g(answer_iter, t, 1):.6f}")
    print(f"Условие сходимости |g'(x₀)| < 1: {abs(f.g(answer_iter, t, 1)):.6f} {'✓' if convergence_checked else '✗'}")
    print()
    
    # Сохраняем точки для метода простых итераций
    # Правильный алгоритм метода простых итераций:
    # 1. Проверяем разницу между текущим x и g(x)
    # 2. Обновляем x = g(x)
    # 3. Останавливаемся когда |x - g(x)| < error
    
    lastx_iter = answer_iter
    lasty_iter = f.g(answer_iter, t)
    
    iter_points = [(answer_iter, f.g(answer_iter, t))]
    iter_iterations = 0
    iter_success = False
    
    # Генерируем точки для графика g(x)
    x_range_g = np.linspace(-math.pi/2 + 0.1, math.pi/2 - 0.1, 500)
    y_values_g = [safe_g(x, t) for x in x_range_g]
    
    ax2.plot(x_range_g, y_values_g, 'b-', linewidth=2, label=f'g(x) = tan²(x) / {t}')
    # Добавляем линию y = x для метода простых итераций
    ax2.plot([-2, 2], [-2, 2], 'k--', alpha=0.5, linewidth=1, label='y = x')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    # Основной цикл метода простых итераций
    while iter_iterations < max_iterations:
        current_x = answer_iter
        g_value = f.g(current_x, t)
        
        print(f"Итерация {iter_iterations}: x = {current_x:.8f}, g(x) = {g_value:.8f}, |x - g(x)| = {abs(current_x - g_value):.8f}")
        
        # Критерий остановки: |x - g(x)| < error
        if abs(current_x - g_value) <= error:
            iter_success = True
            break
        
        # Проверяем условие сходимости на текущей итерации
        if iter_iterations > 0:
            convergence_current = check_iteration_convergence(current_x, t, f)
            if not convergence_current:
                print(f"  Внимание: условие сходимости нарушено |g'(x)| = {abs(f.g(current_x, t, 1)):.6f}")
        
        # Сохраняем текущую точку (x, x) - на диагонали
        iter_points.append((current_x, current_x))
        
        # Рисуем вертикальную линию от (x, x) к (x, g(x))
        ax2.plot([current_x, current_x], [current_x, g_value], 
                'r-', alpha=0.6, linewidth=1)
        
        # Обновляем x для следующей итерации
        answer_iter = g_value
        
        # Сохраняем точку (x, g(x))
        iter_points.append((current_x, g_value))
        
        iter_iterations += 1
    
    # Если метод сошелся, добавляем финальную точку на диагонали
    if iter_success:
        iter_points.append((answer_iter, answer_iter))
    
    # Отображаем точки итераций метода простых итераций
    if iter_points:
        iter_x2, iter_y2 = zip(*iter_points)
        ax2.plot(iter_x2, iter_y2, 'mo-', markersize=4, linewidth=1, alpha=0.8, label='Итерации метода')
        ax2.plot(iter_x2[0], iter_y2[0], 'mo', markersize=8, label='Начальная точка')
        
        if iter_success:
            ax2.plot(answer_iter, answer_iter, 'co', markersize=10, label=f'Корень: x = {answer_iter:.6f}')
    
    ax2.set_title(f"Метод простых итераций\ng(x) = tan²(x) / {t}")
    ax2.set_xlabel('x')
    ax2.set_ylabel('g(x)')
    ax2.legend()
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    
    if tangent_success:
        print(f"Метод касательных: УСПЕШНО")
        print(f"  Корень: x = {tangent_points[-1][0]:.8f}")
        print(f"  f(x) = {tangent_points[-1][1]:.2e}")
        print(f"  Итераций: {tangent_iterations}")
    else:
        print(f"Метод касательных: НЕ ДОСТИГНУТА ТОЧНОСТЬ")
        print(f"  Лучшее приближение: x = {tangent_points[-1][0]:.8f}")
        print(f"  f(x) = {tangent_points[-1][1]:.2e}")
        print(f"  Итераций: {tangent_iterations}")
    
    print()
    
    if iter_success:
        print(f"Метод простых итераций: УСПЕШНО")
        print(f"  Корень: x = {answer_iter:.8f}")
        print(f"  |x - g(x)| = {abs(answer_iter - f.g(answer_iter, t)):.2e}")
        print(f"  Итераций: {iter_iterations}")
    else:
        print(f"Метод простых итераций: НЕ ДОСТИГНУТА ТОЧНОСТЬ")
        print(f"  Лучшее приближение: x = {answer_iter:.8f}")
        print(f"  |x - g(x)| = {abs(answer_iter - f.g(answer_iter, t)):.2e}")
        print(f"  Итераций: {iter_iterations}")
    
    # Проверяем оба корня в исходном уравнении
    if tangent_points:
        root_tangent = tangent_points[-1][0]
        print(f"\nПроверка в исходном уравнении f(x) = tan²(x) - {t}·x:")
        print(f"  Для метода касательных: f({root_tangent:.8f}) = {f.f(root_tangent, t):.2e}")
    
    if iter_success:
        print(f"  Для метода простых итераций: f({answer_iter:.8f}) = {f.f(answer_iter, t):.2e}")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
