import numpy as np
import random
import csv
from tqdm import tqdm
import openpyxl

from person import Person

class Population:
    def __init__(self, npop, p_cross, p_mute, stock_generator, sizeof_tourn, num_elites = 0, buy_rate = 0.5, sell_rate = 0.5, start_gold = 100, t_ga = 10, stats_time = 1):
        assert not (npop % 2)
        self.npop = npop
        self.p_cross = p_cross
        self.p_mute = p_mute
        self.stock_generator = stock_generator
        self.num_elites = num_elites
        self.sizeof_tourn = sizeof_tourn
        self.t_ga = t_ga
        self.buy_rate = buy_rate
        self.sell_rate = sell_rate
        self.start_gold = start_gold
        self.wealthiest_history, self.mean_wealth_history, self.prices, self.best_so_far = [], [], [], 0
        self.init_pop()

    def init_pop(self):
        chromosoms = np.random.randint(2, size=(self.npop, 16) )
        # chromosoms = [[0, -1, 1 ,0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]] * self.npop
        self.population = np.empty(self.npop, dtype=Person)
        for i, chrom in enumerate(chromosoms):
            self.population[i] = Person(gold=self.start_gold, buy_rate=self.buy_rate, sell_rate=self.sell_rate, chrom=chrom)

        # self.stock_generator = self.alternate_sequence_generator(self.T_l, self.T_g)

    def run_population(self, num_days:int):
        last_changes = [next(self.stock_generator)[0] for _ in range(3)]
        for day in tqdm(range(num_days)):
            change, price = next(self.stock_generator)
            last_changes.append(change)
            self.next_day(last_changes = last_changes, today_price = price)
            if not (day % self.t_ga):
                best_wealth = self.get_best_wealth(price)
                # if best_wealth > self.best_so_far:
                #     self.best_so_far = best_wealth
                self.wealthiest_history.append(best_wealth)
                self.mean_wealth_history.append(self.get_mean_wealth(price))
                self.prices.append(price)
            if not (day % self.t_ga) and day != 0:
                self.make_new_pop(price, best_now=best_wealth, num_elites=self.num_elites)

            last_changes.pop(0)

    def next_day(self, last_changes, today_price):
        for person in self.population:
            decision = person.get_decision(last_changes)
            if decision == 1:
                person.buy_oil(today_price)
            elif decision == 0:
                person.sell_oil(today_price)

    def make_new_pop(self, price, best_now, num_elites):
        fits = [person.get_wealth(price) for person in self.population]
        parent_list = self.choose_parent_pop(sizeof_tourn=self.sizeof_tourn, fits=fits, num_elites=num_elites)

        # Select the indices of the best individuals for elitism
        elite_indices = parent_list[:num_elites]

        # Create a new population with the elite individuals preserved
        new_pop_list = []

        # Check if there are elites before attempting the assignment
        if elite_indices:
            elites = self.population[elite_indices]
            for elite in elites:
                elite.reset_gold(self.start_gold)  # Reset gold to start_gold
                elite.reset_oil(0)
                new_pop_list.append(elite)

        # Perform crossover and mutation for the rest of the population
        for i in range(num_elites, self.npop, 2):
            child1, child2 = self.crossover(parent_list[i], parent_list[i+1])
            new_pop_list.append(child1)
            new_pop_list.append(child2)

        # Apply mutation to the non-elite individuals
        p_mute = self.p_mute
        self.apply_mutation(new_pop_list[num_elites:], p_mute)
        # Convert the Python list to a NumPy array with dtype=object
        self.population = np.array(new_pop_list, dtype=Person)

    def crossover(self, p1_idx, p2_idx):
        parent1, parent2 = self.population[[p1_idx, p2_idx]]
        child1 = Person(chrom=parent1.chrom, gold=self.start_gold, oil=0, sell_rate=self.sell_rate, buy_rate=self.buy_rate)
        child2 = Person(chrom=parent2.chrom, gold=self.start_gold, oil=0, sell_rate=self.sell_rate, buy_rate=self.buy_rate)

        if random.random() < self.p_cross:
            cross_point = random.randint(1, len(parent1) - 1)
            temp = child1.chrom[:cross_point].copy()
            child1.chrom[:cross_point] = child2.chrom[:cross_point]
            child2.chrom[:cross_point] = temp

        return child1, child2

    def apply_mutation(self, population, p_mute):
        for person in population:
            for i in range(len(person)):
                if random.random() < p_mute:
                    person.chrom[i] = 1 - person.chrom[i]


    def choose_parent_pop(self, sizeof_tourn, fits, num_elites) -> list:
        fits = np.maximum(fits, 0)
        total_fitness = np.sum(fits)
        elite_indices = np.argsort(fits)[-num_elites:]

        if total_fitness > 0:
            # Exclude elite indices from random selection
            non_elite_indexes = np.setdiff1d(np.arange(len(fits)), elite_indices)

            # Calculate probabilities only for non-elite individuals
            non_elite_fit_sum = np.sum(fits[non_elite_indexes])

            if non_elite_fit_sum > 0:
                non_elite_probabilities = fits[non_elite_indexes] / non_elite_fit_sum
            else:
                # If the sum of non-elite fitness values is 0, set probabilities to a uniform distribution
                non_elite_probabilities = np.ones_like(fits[non_elite_indexes]) / len(non_elite_indexes)

            # Randomly select individuals using normalized probabilities
            selected_indexes = np.random.choice(non_elite_indexes, size=self.npop - num_elites, p=non_elite_probabilities)

            # Use elite indices for the first num_elites positions
            winners = elite_indices.tolist() + selected_indexes.tolist()
        else:
            # If total fitness is 0, select individuals randomly without replacement
            winners = np.random.choice(np.arange(len(fits)), size=self.npop, replace=False).tolist()

        return winners


    def get_mean_wealth(self, curr_price):
        wealths = [person.get_wealth(curr_price) for person in self.population]
        return np.average(wealths)
    
    def get_best_wealth(self, curr_price):
        wealths = [person.get_wealth(curr_price) for person in self.population]
        wealthiest_idx = np.argmax(wealths)
        best_wealth = wealths[wealthiest_idx]
        return best_wealth

    @staticmethod
    def excel_column_generator(path = 'Crude Oil Prices Daily.xlsx'):
        dataframe = openpyxl.load_workbook(path, data_only=True)
        dataframe1 = dataframe.active
        
        # Iterate the loop to read the cell values
        for row in range(1, dataframe1.max_row):
            col1 = dataframe1['C']
            col2 = dataframe1['B']
            yield float(col1[row].value), float(col2[row].value)

    @staticmethod
    def csv_column_generator(file_path = 'crude-oil-price.csv', col1 = 'percentChange', col2 = 'price'):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yield float(row[col1]), float(row[col2]) 

    @staticmethod
    def alternate_sequence_generator(seq1, seq2, T_global):
        # seq1 = [21, 24, 30, 16, 11, 15, 18, 13, 16, 17, 19, 20, 30, 12]
        # seq2 = [18, 11, 15, 20, 21, 24, 30, 30, 16, 17, 19,  12, 18, 13] 
        n1, n2 = len(seq1), len(seq2)
        current_index = 0
        previous_value = 0

        while True:
            current_value = seq1[current_index % n1] if current_index % (2 * T_global) < T_global else seq2[current_index % n2]

            percent_change = 0 if previous_value == 0 else ((current_value - previous_value) / previous_value)

            yield percent_change, current_value

            previous_value = current_value
            current_index += 1