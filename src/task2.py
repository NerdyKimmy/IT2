import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ScheduleProblem:
    def __init__(self, num_groups, num_teachers, num_rooms, num_slots_per_day, num_days, num_lessons):
        self.num_groups = num_groups
        self.num_teachers = num_teachers
        self.num_rooms = num_rooms
        self.num_slots_per_day = num_slots_per_day
        self.num_days = num_days
        self.total_slots = num_slots_per_day * num_days
        self.num_lessons = num_lessons
        
        np.random.seed(42)
        random.seed(42)
        self.lessons = []
        
        group_counts = {i: 0 for i in range(num_groups)}
        teacher_counts = {i: 0 for i in range(num_teachers)}
        
        while len(self.lessons) < num_lessons:
            g = np.random.randint(0, num_groups)
            t = np.random.randint(0, num_teachers)
            if group_counts[g] < self.total_slots and teacher_counts[t] < self.total_slots:
                self.lessons.append((g, t))
                group_counts[g] += 1
                teacher_counts[t] += 1

def calculate_penalties(chromosome, problem):
    hard_penalty = 0
    soft_penalty = 0
    
    slot_room = set()
    slot_teacher = set()
    slot_group = set()
    group_daily_slots = {g: {d: [] for d in range(problem.num_days)} for g in range(problem.num_groups)}
    
    for lesson_idx, gene in enumerate(chromosome):
        t_slot = gene // problem.num_rooms
        room = gene % problem.num_rooms
        group, teacher = problem.lessons[lesson_idx]
        
        day = t_slot // problem.num_slots_per_day
        slot_in_day = t_slot % problem.num_slots_per_day
        
        if (t_slot, room) in slot_room: hard_penalty += 1
        else: slot_room.add((t_slot, room))
            
        if (t_slot, teacher) in slot_teacher: hard_penalty += 1
        else: slot_teacher.add((t_slot, teacher))
            
        if (t_slot, group) in slot_group: hard_penalty += 1
        else: slot_group.add((t_slot, group))
            
        group_daily_slots[group][day].append(slot_in_day)
        
    for g, days in group_daily_slots.items():
        for d, slots in days.items():
            if len(slots) > 0:
                slots.sort()
                windows = (slots[-1] - slots[0] + 1) - len(slots)
                soft_penalty += windows
                if len(slots) > 3:
                    soft_penalty += (len(slots) - 3)
                    
    return hard_penalty, soft_penalty

def greedy_schedule(problem):
    chromosome = []
    slot_room = set()
    slot_teacher = set()
    slot_group = set()
    
    for group, teacher in problem.lessons:
        assigned = False
        for t_slot in range(problem.total_slots):
            for room in range(problem.num_rooms):
                if (t_slot, room) not in slot_room and \
                   (t_slot, teacher) not in slot_teacher and \
                   (t_slot, group) not in slot_group:
                    chromosome.append(t_slot * problem.num_rooms + room)
                    slot_room.add((t_slot, room))
                    slot_teacher.add((t_slot, teacher))
                    slot_group.add((t_slot, group))
                    assigned = True
                    break
            if assigned: break
        if not assigned:
            chromosome.append(random.randint(0, problem.total_slots * problem.num_rooms - 1))
            
    h_pen, s_pen = calculate_penalties(chromosome, problem)
    return chromosome, h_pen, s_pen

class ScheduleGA:
    def __init__(self, problem, pop_size, mut_rate, cross_rate):
        self.problem = problem
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.max_gene = problem.total_slots * problem.num_rooms - 1
        self.population = [[random.randint(0, self.max_gene) for _ in range(problem.num_lessons)] for _ in range(pop_size)]
        
    def evaluate(self):
        fitnesses = []
        valid_count = 0
        best_ind, best_h, best_s, max_fit = None, float('inf'), float('inf'), -1
        
        for chrom in self.population:
            h_pen, s_pen = calculate_penalties(chrom, self.problem)
            if h_pen == 0: valid_count += 1
            
            fit = 1.0 / (1.0 + 100 * h_pen + s_pen)
            fitnesses.append(fit)
            
            if fit > max_fit:
                max_fit = fit
                best_ind, best_h, best_s = chrom, h_pen, s_pen
                
        return fitnesses, valid_count / self.pop_size, best_ind, best_h, best_s

    def select(self, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            competitors = random.sample(range(self.pop_size), 3)
            best_idx = max(competitors, key=lambda idx: fitnesses[idx])
            selected.append(self.population[best_idx].copy())
        return selected

    def crossover(self, parents):
        offspring = []
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < self.pop_size else parents[0]
            if random.random() < self.cross_rate:
                mask = [random.choice([True, False]) for _ in range(self.problem.num_lessons)]
                c1 = [p1[j] if mask[j] else p2[j] for j in range(self.problem.num_lessons)]
                c2 = [p2[j] if mask[j] else p1[j] for j in range(self.problem.num_lessons)]
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.extend([c1, c2])
        return offspring[:self.pop_size]

    def mutate(self, offspring):
        for ind in offspring:
            if random.random() < self.mut_rate:
                idx = random.randint(0, self.problem.num_lessons - 1)
                ind[idx] = random.randint(0, self.max_gene)
        return offspring

    def run(self, generations=100):
        history_valid = []
        history_h = []
        history_s = []
        best_overall, best_h, best_s = None, float('inf'), float('inf')
        
        for _ in range(generations):
            fitnesses, valid_ratio, best_ind, h_pen, s_pen = self.evaluate()
            
            if h_pen < best_h or (h_pen == best_h and s_pen < best_s):
                best_h, best_s = h_pen, s_pen
                best_overall = best_ind.copy()
                
            history_valid.append(valid_ratio * 100)
            history_h.append(best_h)
            history_s.append(best_s)
                
            parents = self.select(fitnesses)
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring)
            
        return best_overall, best_h, best_s, history_valid, history_h, history_s

def decode_and_print(chromosome, problem):
    schedule_details = []
    
    for idx, gene in enumerate(chromosome):
        t_slot = gene // problem.num_rooms
        room = gene % problem.num_rooms
        group, teacher = problem.lessons[idx]
        day = t_slot // problem.num_slots_per_day
        slot = t_slot % problem.num_slots_per_day
        schedule_details.append((day + 1, slot + 1, room + 1, group + 1, teacher + 1))
        
    schedule_details.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\nПовний розклад найкращої особини:")
    print(f"| {'День':<4} | {'Пара':<4} | {'Аудиторія':<10} | {'Група':<6} | {'Викладач':<9} |")
    print(f"|{'-'*6}|{'-'*6}|{'-'*12}|{'-'*8}|{'-'*11}|")
    for day, slot, room, group, teacher in schedule_details:
        print(f"| {day:<4} | {slot:<4} | {room:<10} | {group:<6} | {teacher:<9} |")

def plot_report_graphs(greedy_res, ga_results, dataset_name, output_file):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    best_ga = ga_results[-1]
    axes[0].plot(best_ga['hist_h'], label='Жорсткі штрафи', color='red', linewidth=2)
    axes[0].plot(best_ga['hist_s'], label="М'які штрафи", color='orange', linewidth=2)
    axes[0].set_title(f'Динаміка штрафів ({dataset_name})')
    axes[0].set_xlabel('Покоління')
    axes[0].set_ylabel('Кількість порушень')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    colors = ['blue', 'green', 'purple']
    labels = ['Швидка', 'Збалансована', 'Дослідницька']
    for idx, res in enumerate(ga_results):
        axes[1].plot(res['hist_v'], label=labels[idx], color=colors[idx], linewidth=2)
    axes[1].set_title('Особини без жорстких конфліктів')
    axes[1].set_xlabel('Покоління')
    axes[1].set_ylabel('% популяції')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    bar_labels = ['Жадібний', 'Швидка', 'Збалансов.', 'Дослідн.']
    hard_scores = [greedy_res[0]] + [res['best_h'] for res in ga_results]
    soft_scores = [greedy_res[1]] + [res['best_s'] for res in ga_results]
    
    x = np.arange(len(bar_labels))
    width = 0.35
    axes[2].bar(x - width/2, hard_scores, width, label='Жорсткі', color='red')
    axes[2].bar(x + width/2, soft_scores, width, label="М'які", color='orange')
    axes[2].set_title(f'Фінальний результат ({dataset_name})')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(bar_labels)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Графік збережено: {output_file}")

if __name__ == "__main__":
    datasets = [
        ("Малий дефіцит", ScheduleProblem(3, 3, 2, 4, 3, 18)),
        ("Середній дефіцит", ScheduleProblem(5, 5, 3, 4, 5, 45)),
        ("Екстремальний дефіцит", ScheduleProblem(4, 4, 2, 4, 5, 38))
    ]
    
    configs = [
        {"pop": 50, "mut": 0.10, "cross": 0.7, "name": "Швидка"},
        {"pop": 150, "mut": 0.25, "cross": 0.8, "name": "Збалансована"},
        {"pop": 300, "mut": 0.40, "cross": 0.9, "name": "Дослідницька"}
    ]
    
    generations = 150
    summary_results = {}
    
    for dataset_name, problem in datasets:
        print(f"\n{'='*60}\nТестування: {dataset_name}\n{'='*60}")
        
        _, greedy_h, greedy_s = greedy_schedule(problem)
        print(f"Жадібний: Жорсткі {greedy_h}, М'які {greedy_s}")
        
        dataset_results = {"greedy": (greedy_h, greedy_s), "ga": {}}
        ga_full_results = []
        best_overall_chromosome = None
        best_overall_h = float('inf')
        best_overall_s = float('inf')
        
        for cfg in configs:
            ga = ScheduleGA(problem, cfg["pop"], cfg["mut"], cfg["cross"])
            best_ind, best_h, best_s, hist_v, hist_h, hist_s = ga.run(generations)
            
            ga_full_results.append({
                'best_h': best_h, 'best_s': best_s, 
                'hist_v': hist_v, 'hist_h': hist_h, 'hist_s': hist_s
            })
            dataset_results["ga"][cfg["name"]] = (best_h, best_s)
            
            print(f"ГА ({cfg['name']}): Жорсткі {best_h}, М'які {best_s}")
            
            if (best_h < best_overall_h or
                    (best_h == best_overall_h and best_s < best_overall_s)):
                best_overall_h = best_h
                best_overall_s = best_s
                best_overall_chromosome = best_ind.copy()
                
        summary_results[dataset_name] = dataset_results
        
        decode_and_print(best_overall_chromosome, problem)
        
        output_file = f"task2_dataset_{datasets.index((dataset_name, problem)) + 1}.png"
        plot_report_graphs((greedy_h, greedy_s), ga_full_results, dataset_name, output_file)

    print("\n\n" + "="*80)
    print("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ (Жорсткі / М'які штрафи)")
    print("="*80)
    
    header = f"{'Набір даних':<25} | {'Жадібний':<10} | " + " | ".join([f"{c['name']:<12}" for c in configs])
    print(header)
    print("-" * 80)
    
    for d_name, _ in datasets:
        res = summary_results[d_name]
        gr_h, gr_s = res['greedy']
        row = f"{d_name:<25} | {gr_h:>3} / {gr_s:<4} | "
        
        ga_parts = []
        for c in configs:
            h, s = res['ga'][c['name']]
            ga_parts.append(f"{h:>4} / {s:<5}")
            
        row += " | ".join(ga_parts)
        print(row)
    print("="*80)