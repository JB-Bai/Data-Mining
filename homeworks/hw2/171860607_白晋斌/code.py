import pandas as pd
import itertools
import time
import copy
import os

def groceries_data():
    groceries = pd.read_csv("GroceryStore/Groceries.csv", index_col=0).values.tolist()
    newgroceries = []
    for grocery in groceries:
        newgrocery = set(grocery[0][1:-1].split(","))  # remove {}
        newgroceries.append(newgrocery)
    return newgroceries

def unix_data():
    unix = []
    for i in range(9):
        with open("UNIX_usage/USER" + str(i) + "/sanitized_all.981115184025", 'r') as f:
            item=set()
            for line in f:

                if line.strip() == '**SOF**':
                    item=set()
                elif line.strip() == '**EOF**':
                    unix.append(item)
                else:
                    l = line.strip()
                    item.add(l)
    return unix # 2357

def dummy(data,minimum_support=0.01,minimum_confidence=0.5,itemset_size=10,association_rules_open=True,save_in_files=False):
    frequency_threshold = int(len(data) * minimum_support)
    print("minimum_support = %f, minimum_items = %d" % (minimum_support, frequency_threshold))
    start_time=time.time()
    freq = {}
    rule_freq = {}
    for items in data:
        for item in items:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
    freqs = []
    for k, v in freq.items():
        if v >= frequency_threshold:
            freqs.append([k])
            rule_freq[k] = v
    print('generate', len(freqs), 'Frequent 1-Item Set waste time', time.time() - start_time, 's.')
    pre_freqs = sorted(freqs)
    if save_in_files:
        with open('dummy-freq-1-itemsets.txt', 'w') as outfile:
            for key in pre_freqs:
                keystr = ','.join(key)
                if freq[keystr] >= frequency_threshold:
                    outfile.write(keystr + ' : ' + str(freq[keystr]) + '\n')
    del freq
    for k in range(2, itemset_size + 1):
        candidates = []
        if k == 2:
            for i in range(len(pre_freqs)):
                for j in range(i + 1, len(pre_freqs)):
                    candidates.append([pre_freqs[i][0], pre_freqs[j][0]])
        else:
            i = 0
            while i < len(pre_freqs) - 1:
                tails = []
                while i < len(pre_freqs) - 1 and pre_freqs[i][:-1] == pre_freqs[i + 1][:-1]:
                    tails.append(pre_freqs[i][-1])
                    i += 1
                if tails:
                    tails.append(pre_freqs[i][-1])
                    prefix = copy.deepcopy(pre_freqs[i][0:-1])
                    for a in range(len(tails)):
                        for b in range(a + 1, len(tails)):
                            items = copy.deepcopy(prefix)
                            items.append(tails[a])
                            items.append(tails[b])
                            candidates.append(items)
                i += 1
        # count freqset
        k_item_freq = {}
        for candidate in candidates:
            canset = set(candidate)
            canstr = ','.join(candidate)
            k_item_freq[canstr] = 0
            for i in range(len(data)):
                if canset <= data[i]:
                    k_item_freq[canstr] += 1

        pre_freqs = []
        new_k_item_freq = {}
        for keyt, v in k_item_freq.items():
            if v >= frequency_threshold:
                pre_freqs.append(keyt.split(','))
                new_k_item_freq[','.join(pre_freqs[-1])] = k_item_freq[keyt]
                rule_freq[','.join(pre_freqs[-1])] = k_item_freq[keyt]
        k_item_freq = new_k_item_freq
        if len(pre_freqs) == 0:
            break
        pre_freqs = sorted(pre_freqs)
        print('generate', len(pre_freqs), 'Frequent', str(k) + '-Item Set waste time', time.time() - start_time, 's.')
        if save_in_files:
            with open('dummy-freq-' + str(k) + '-itemsets.txt', 'w') as outfile:
                for key in pre_freqs:
                    keystr = ','.join(key)
                    if k_item_freq[keystr] >= frequency_threshold:
                        outfile.write(keystr + ' : ' + str(k_item_freq[keystr]) + '\n')
        del k_item_freq
        # generate association rules
        association_rules = []
        if association_rules_open:
            for i in range(len(pre_freqs)):
                n = len(pre_freqs[i])
                all_subsets = []
                for out in range(1, 2 ** (n - 1)):
                    subset = set()
                    for j in range(n):
                        if (out >> j) % 2 == 1:
                            subset.add(pre_freqs[i][j])
                    all_subsets.append(subset)
                par_set = set(pre_freqs[i])
                par_cnt = rule_freq[','.join(pre_freqs[i])]
                for subset in all_subsets:
                    subset_str = ','.join(sorted(list(subset)))
                    diffset_str = ','.join(sorted(list(par_set.difference(subset))))
                    subset_cnt = rule_freq[subset_str]
                    diffset_cnt = rule_freq[diffset_str]
                    if 1.0 * par_cnt / subset_cnt >= minimum_confidence:
                        association_rules.append([subset_str, diffset_str, 1.0 * par_cnt / subset_cnt])
                    elif 1.0 * par_cnt / diffset_cnt >= minimum_confidence:
                        association_rules.append([diffset_str, subset_str, 1.0 * par_cnt / diffset_cnt])
            if save_in_files:
                with open('dummy-freq-' + str(k) + '-itemsets.txt', 'a') as outfile2:
                    outfile2.write('\n')
                    for itemk in association_rules:
                        outfile2.write('(' + itemk[0] + ') -> (' + itemk[1] + ')' + '\t confidence:' + str(itemk[2]) + '\n')


    return pre_freqs

def apriori(data,minimum_support=0.01,minimum_confidence=0.5,itemset_size=10,association_rules_open=True,save_in_files=False):
    frequency_threshold = int(len(data) * minimum_support)
    print("minimum_support = %f, minimum_items = %d"%(minimum_support, frequency_threshold))
    start_time=time.time()
    freq = {}
    rule_freq={}
    for items in data:
        for item in items:
            if item in freq:
                freq[item]+=1
            else:
                freq[item]=1
    freqs=[]
    for k,v in freq.items():
        if v>=frequency_threshold:
            freqs.append([k])
            rule_freq[k]=v
    print('generate', len(freqs), 'Frequent 1-Item Set waste time', time.time() - start_time, 's.')
    pre_freqs=sorted(freqs)
    if save_in_files:
        with open('apriori-freq-1-itemsets.txt', 'w') as outfile:
            for key in pre_freqs:
                keystr = ','.join(key)
                if freq[keystr] >= frequency_threshold:
                    outfile.write(keystr + ' : ' + str(freq[keystr]) + '\n')
    del freq
    for k in range(2,itemset_size+1):
        candidates = []
        if k == 2:
            for i in range(len(pre_freqs)):
                for j in range(i + 1, len(pre_freqs)):
                    candidates.append([pre_freqs[i][0], pre_freqs[j][0]])
        else:
            i = 0
            while i < len(pre_freqs) - 1:
                tails = []
                while i < len(pre_freqs) - 1 and pre_freqs[i][:-1] == pre_freqs[i + 1][:-1]:
                    tails.append(pre_freqs[i][-1])
                    i += 1
                if tails:
                    tails.append(pre_freqs[i][-1])
                    prefix = copy.deepcopy(pre_freqs[i][0:-1])
                    for a in range(len(tails)):
                        for b in range(a + 1, len(tails)):
                            items = copy.deepcopy(prefix)
                            items.append(tails[a])
                            items.append(tails[b])
                            candidates.append(items)
                i += 1
        # count freqset
        prune_basis = []
        for i in range(len(data)):
            prune_basis.append(0)
        k_item_freq = {}
        for candidate in candidates:
            canset = set(candidate)
            canstr = ','.join(candidate)
            k_item_freq[canstr] = 0
            for i in range(len(data)):
                if canset <= data[i]:
                    k_item_freq[canstr] += 1
                    prune_basis[i] += 1
        # prune
        if len(prune_basis) != 0:
            h = 0
            for i in range(len(prune_basis)):
                if prune_basis[i] < k + 1:
                    del data[h]
                else:
                    h += 1

        pre_freqs=[]
        new_k_item_freq={}
        for keyt,v in k_item_freq.items():
            if v>=frequency_threshold:
                pre_freqs.append(keyt.split(','))
                new_k_item_freq[','.join(pre_freqs[-1])] = k_item_freq[keyt]
                rule_freq[','.join(pre_freqs[-1])] = k_item_freq[keyt]
        k_item_freq=new_k_item_freq
        if len(pre_freqs) == 0:
            break
        pre_freqs=sorted(pre_freqs)
        #print('generate', len(pre_freqs), 'Frequent', str(k) + '-Item Set waste time', time.time() - start_time, 's.')
        # save in files
        if save_in_files:
            with open('apriori-freq-' + str(k) + '-itemsets.txt', 'w') as outfile:
                for key in pre_freqs:
                    keystr = ','.join(key)
                    if k_item_freq[keystr] >= frequency_threshold:
                        outfile.write(keystr + ' : ' + str(k_item_freq[keystr]) + '\n')
        del k_item_freq
        #generate association rules
        association_rules = []
        if association_rules_open:
            for i in range(len(pre_freqs)):
                n = len(pre_freqs[i])
                all_subsets = []
                for out in range(1, 2 ** (n - 1)):
                    subset = set()
                    for j in range(n):
                        if (out >> j) % 2 == 1:
                            subset.add(pre_freqs[i][j])
                    all_subsets.append(subset)
                par_set = set(pre_freqs[i])
                par_cnt = rule_freq[','.join(pre_freqs[i])]
                for subset in all_subsets:
                    subset_str = ','.join(sorted(list(subset)))
                    diffset_str = ','.join(sorted(list(par_set.difference(subset))))
                    subset_cnt = rule_freq[subset_str]
                    diffset_cnt = rule_freq[diffset_str]
                    if 1.0 * par_cnt / subset_cnt >= minimum_confidence:
                        association_rules.append([subset_str, diffset_str, 1.0 * par_cnt / subset_cnt])
                    elif 1.0 * par_cnt / diffset_cnt >= minimum_confidence:
                        association_rules.append([diffset_str, subset_str, 1.0 * par_cnt / diffset_cnt])
            if save_in_files:
                with open('apriori-freq-' + str(k) + '-itemsets.txt', 'a') as outfile2:
                    outfile2.write('\n')
                    for itemk in association_rules:
                        outfile2.write('(' + itemk[0] + ') -> (' + itemk[1] + ')' + '\t confidence:' + str(itemk[2]) + '\n')
        print('generate', len(pre_freqs), 'Frequent', str(k) + '-Item Set waste time', time.time() - start_time, 's.')

    return pre_freqs

def fpgrowth(data,minimum_support=0.01,minimum_confidence=0.5,itemset_size=10,association_rules_open=True,save_in_files=False):
    class Node(object):
        def __init__(self, value, count, parent):
            self.value = value
            self.count = count
            self.parent = parent
            self.link = None
            self.children = {}

    class FPTree():
        def update_header(self, node, targetNode):
            while node.link != None:
                node = node.link
            node.link = targetNode

        def update_fptree(self, items, node, headerTable):
            if items[0] in node.children:
                node.children[items[0]].count += 1
            else:
                node.children[items[0]] = Node(items[0], 1, node)
                if headerTable[items[0]][1] is None:
                    headerTable[items[0]][1] = node.children[items[0]]
                else:
                    self.update_header(headerTable[items[0]][1], node.children[items[0]])
            if len(items) > 1:
                self.update_fptree(items[1:], node.children[items[0]], headerTable)

        def create_fptree(self, data_set, min_support):
            item_count = {}
            for t in data_set:
                for item in t:
                    if item not in item_count:
                        item_count[item] = 1
                    else:
                        item_count[item] += 1
            headerTable = {}
            for k in item_count:
                if item_count[k] >= min_support:
                    headerTable[k] = item_count[k]
            freqItemSet = set(headerTable.keys())
            if len(freqItemSet) == 0:
                return None, None
            for k in headerTable:
                headerTable[k] = [headerTable[k], None]
            tree_header = Node('head node', 1, None)
            ite = data_set
            for t in ite:
                localD = {}
                for item in t:
                    if item in freqItemSet:
                        localD[item] = headerTable[item][0]
                if len(localD) > 0:
                    order_item = [v[0] for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)]
                    self.update_fptree(order_item, tree_header, headerTable)
            return tree_header, headerTable

        def find_path(self, node, nodepath):
            if node.parent is not None:
                nodepath.append(node.parent.value)
                self.find_path(node.parent, nodepath)

        def find_cond_pattern_base(self, value, headerTable):
            treeNode = headerTable[value][1]
            cond_pat_base = {}
            while treeNode is not None:
                nodepath = []
                self.find_path(treeNode, nodepath)
                if len(nodepath) > 1:
                    cond_pat_base[frozenset(nodepath[:-1])] = treeNode.count
                treeNode = treeNode.link
            return cond_pat_base

        def create_cond_fptree(self, headerTable, min_support, temp, freq_items, support_data):
            freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
            for freq in freqs:
                freq_set = temp.copy()
                freq_set.add(freq)
                freq_items.add(frozenset(freq_set))
                if frozenset(freq_set) not in support_data:
                    support_data[frozenset(freq_set)] = headerTable[freq][0]
                else:
                    support_data[frozenset(freq_set)] += headerTable[freq][0]
                cond_pat_base = self.find_cond_pattern_base(freq, headerTable)
                cond_pat_dataset = []
                for item in cond_pat_base:
                    item_temp = list(item)
                    item_temp.sort()
                    for i in range(cond_pat_base[item]):
                        cond_pat_dataset.append(item_temp)
                cond_tree, cur_headtable = self.create_fptree(cond_pat_dataset, min_support)
                if cur_headtable is not None:
                    self.create_cond_fptree(cur_headtable, min_support, freq_set, freq_items, support_data)

        def generate(self, data_set, min_support, min_conf,start_time=0.0,association_rules_open=True,save_in_files=False):
            freqItemSet = set()
            support_data = {}
            tree_header, headerTable = self.create_fptree(data_set, min_support)
            self.create_cond_fptree(headerTable, min_support, set(), freqItemSet, support_data)
            max_l = 0
            for i in freqItemSet:
                if len(i) > max_l: max_l = len(i)
            L = [set() for _ in range(max_l)]
            for i in freqItemSet:
                L[len(i) - 1].add(i)

            #for i in range(len(L)):
            #    print('generate', len(L[i]), 'Frequent', str(i+1) + '-Item Set waste time', time.time() - start_time,'s.')


            rule_list = []
            if association_rules_open:
                sub_set_list = []
                for i in range(0, len(L)):
                    for freq_set in L[i]:
                        for sub_set in sub_set_list:
                            if sub_set.issubset(freq_set) and freq_set - sub_set in support_data:
                                conf = support_data[freq_set] / support_data[freq_set - sub_set]
                                big_rule = (freq_set - sub_set, sub_set, conf)
                                if conf >= min_conf and big_rule not in rule_list:
                                    rule_list.append(big_rule)
                        sub_set_list.append(freq_set)
                rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
            for i in range(len(L)):
                print('generate', len(L[i]), 'Frequent', str(i+1) + '-Item Set waste time', time.time() - start_time,'s.')
            if save_in_files:
                with open(save_path, "w") as f:
                    for item in rule_list:
                        f.write("{}=>{}\t{:.3f}\n".format(str(list(item[0])), str(list(item[1])), item[2]))
                    f.close()

            return rule_list


    frequency_threshold = int(len(data) * minimum_support)
    print("minimum_support = %f, minimum_items = %d" % (minimum_support, frequency_threshold))

    start_time = time.time()
    save_path = "fpgrowth.txt"
    fp = FPTree()
    rule_list = fp.generate(data, frequency_threshold, minimum_confidence,start_time,association_rules_open,save_in_files)

    return rule_list


if __name__ == '__main__':

    data=groceries_data()
    #data=unix_data()
    #dummy(data)
    apriori(data,minimum_support=0.01,minimum_confidence=0.5,association_rules_open=True,save_in_files=False)
    #fpgrowth()

