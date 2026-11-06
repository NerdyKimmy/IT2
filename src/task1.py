import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def target_function(x):
    """Негладка мультимодальна цільова функція."""
    return np.abs(x[0]) + np.abs(x[1]) + 0.5 * (np.sin(5 * np.pi * x[0]) + np.cos(5 * np.pi * x[1]))

def numerical_gradient(x, epsilon=1e-5):
    """Обчислення градієнта чисельним методом."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (target_function(x_plus) - target_function(x_minus)) / (2 * epsilon)
    return grad

def gradient_descent(start_point, learning_rate=0.01, iterations=150):
    """Градієнтний спуск з фіксацією історії значень."""
    current_point = np.array(start_point, dtype=float)
    history_vals = [target_function(current_point)]
    
    for _ in range(iterations):
        grad = numerical_gradient(current_point)
        current_point -= learning_rate * grad
        history_vals.append(target_function(current_point))
        
    return current_point.copy(), target_function(current_point), history_vals

class GeneticAlgorithm:
    def __init__(self, pop_size=50, dimensions=2, bounds=(-3.0, 3.0), mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))

    def fitness(self):
        values = np.array([target_function(ind) for ind in self.population])
        fit_scores = 1.0 / (1e-6 + values - np.min(values))
        return fit_scores, values

    def select(self, fit_scores):
        selected = np.zeros_like(self.population)
        for i in range(self.pop_size):
            competitors = np.random.choice(self.pop_size, 3, replace=False)
            best_idx = competitors[np.argmax(fit_scores[competitors])]
            selected[i] = self.population[best_idx]
        return selected

    def crossover(self, parents):
        offspring = np.zeros_like(parents)
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < self.pop_size else parents[0]
            if np.random.rand() < self.crossover_rate:
                alpha = np.random.rand(self.dimensions)
                offspring[i] = alpha * p1 + (1 - alpha) * p2
                offspring[i+1] = alpha * p2 + (1 - alpha) * p1
            else:
                offspring[i], offspring[i+1] = p1, p2
        return offspring

    def mutate(self, offspring):
        for i in range(self.pop_size):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, 0.2, self.dimensions)
                offspring[i] += mutation
                offspring[i] = np.clip(offspring[i], self.bounds[0], self.bounds[1])
        return offspring

    def run(self, generations=150):
        global_best_val = float('inf')
        global_best_point = None
        history_vals = []
        
        for _ in range(generations):
            fit_scores, raw_values = self.fitness()
            current_best_idx = np.argmin(raw_values)
            
            if raw_values[current_best_idx] < global_best_val:
                global_best_val = raw_values[current_best_idx]
                global_best_point = self.population[current_best_idx].copy()
                
            history_vals.append(global_best_val)
            parents = self.select(fit_scores)
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring)
            
        return global_best_point, global_best_val, history_vals

# Основний блок виконання
if __name__ == "__main__":
    generations = 150

    # 1. Градієнтний спуск із точки (2.5, 2.5)
    gd_point, gd_final, gd_history = gradient_descent(start_point=[2.5, 2.5], iterations=generations)

    # 2. Генетичний алгоритм (популяція 50)
    ga = GeneticAlgorithm(pop_size=50, mutation_rate=0.1)
    ga_point, ga_final, ga_history = ga.run(generations=generations)

    print(f"Градієнтний спуск: x={gd_point[0]:.4f}, y={gd_point[1]:.4f}, f={gd_final:.4f}")
    print(f"Генетичний алгоритм: x={ga_point[0]:.4f}, y={ga_point[1]:.4f}, f={ga_final:.4f}")

    # Побудова графіка
    plt.figure(figsize=(9, 5))
    plt.plot(gd_history, label=f'Градієнтний спуск (застряг на {gd_final:.2f})', color='red', linestyle='--', linewidth=2)
    plt.plot(ga_history, label=f'Генетичний алгоритм (знайшов {ga_final:.2f})', color='green', linewidth=2)

    plt.title('Порівняння збіжності на негладкій функції')
    plt.xlabel('Покоління / Ітерація')
    plt.ylabel('Значення функції f(x, y)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task1.png", dpi=200, bbox_inches="tight")
    plt.close()
