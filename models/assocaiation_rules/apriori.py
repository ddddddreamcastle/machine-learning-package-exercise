from models.model import model

class Apriori(model):

    def __init__(self, support_threshold):
        super(Apriori, self).__init__()
        self.frequent_items = None
        self.support_threshold = support_threshold

    def support(self, item):
        """ calc support value """
        item_count = 0
        for it in self.items:
            if item.issubset(it):
                item_count += 1
        return item_count

    def remove_infrequent_items(self):
        """ remove infreuquent items from frequent items """
        removed = []
        for k, v in self.frequent_items.items():
            if v < self.support_threshold:
                removed.append(k)
        for r in removed:
            self.frequent_items.pop(r)

    def get_next_level_frequent_items(self):
        """ calc k-size frequent items """
        k = len(list(self.frequent_items)[0].split('-')) + 1
        key_set = list(self.frequent_items)
        self.frequent_items = {}
        for idx, key in enumerate(key_set):
            for other_idx in range(idx+1, len(key_set)):
                current = set(key.split('-'))
                other = set(key_set[other_idx].split('-'))
                if len(current | other) == k:
                    self.frequent_items["-".join(current | other)] = self.support(current | other)
        self.remove_infrequent_items()

    def train(self, items):
        self.items = items
        self.frequent_items = {}
        result = []
        for i in range(len(self.items)):
            self.items[i] = set(self.items[i])
        for item in self.items:
            for it in item:
                if it not in self.frequent_items:
                    self.frequent_items[it] = 1
                else:
                    self.frequent_items[it] += 1
        self.remove_infrequent_items()
        while len(self.frequent_items) != 0:
            result += list(self.frequent_items)
            self.get_next_level_frequent_items()
        return result

    def predict(self, x):
        pass

if __name__ == '__main__':
    """ test code """
    a = Apriori(2)
    print(a.train([['1','3','4'],['2','3','5'],['1','2','3','5'],['2','5']]))


