

class Person:
    pos_dict = {
        (0, 0, 0, 0) : 0,
        (0, 0, 0, 1) : 1,
        (0, 0, 1, 0) : 2,
        (0, 0, 1, 1) : 3,
        (0, 1, 0, 0) : 4,
        (0, 1, 0, 1) : 5,
        (0, 1, 1, 0) : 6,
        (0, 1, 1, 1) : 7,
        (1, 0, 0, 0) : 8,
        (1, 0, 0, 1) : 9,
        (1, 0, 1, 0) : 10,
        (1, 0, 1, 1) : 11,
        (1, 1, 0, 0) : 12,
        (1, 1, 0, 1) : 13,
        (1, 1, 1, 0) : 14,
        (1, 1, 1, 1) : 15,
    }

    def __init__(self, chrom, gold = 10, oil = 0, buy_rate = 0.5, sell_rate = 0.5):
        self.gold = gold
        self.oil = oil
        self.buy_rate = buy_rate
        self.sell_rate = sell_rate
        self.chrom = chrom

    def get_wealth(self, curr_price) -> float:
        return self.gold + curr_price * self.oil
    
    def get_income(self, initial_gold, curr_price):
        return (self.gold + curr_price * self.oil) - initial_gold
    
    def buy_oil(self, price, rate = None):
        if rate is None:
            rate = self.buy_rate
        self.oil += rate * self.gold / price
        self.gold *= (1 - rate)


    def sell_oil(self, price, rate = None):
        if rate is None:
            rate = self.sell_rate
        self.gold += rate * self.oil * price
        self.oil *= (1 - rate)

    def reset_gold(self, amount):
        self.gold = amount

    def reset_oil(self, amount):
        self.oil = amount
    
    def get_decision(self, last_changes):
        """
        returns: 
        1 - buy
        0 - do nothing
        -1 - sell
        """
        dict_key = tuple([0 if change < 0 else 1 for change in last_changes])
        position = self.pos_dict[dict_key]
        decision = self.chrom[position]
        return decision
    
    # def __getitem__(self, idx):
    #     return self.chrom[idx]
    
    def __setitem__(self, idx, val):
        self.chrom[idx] = val

    def __len__(self):
        return len(self.chrom)
    
    def __repr__(self):
        return f"Person(chrom={self.chrom}, gold={self.gold}, oil={self.oil}, " \
               f"buy_rate={self.buy_rate}, sell_rate={self.sell_rate})"