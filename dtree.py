# Fabian Ardeljan
# Data Mining
# Dr. Ye

import numpy as np
import decimal

DISCRETE_STEPS = 5

# Helper Functions

def column(data, n):
    return [row[n] for row in data]

def parseLabels(datapath):
    labels = []
    types = []
    index_count = 0
    with open(datapath + ".names") as namefile:
        line = namefile.readline()
        while line:
            if (":" in line):
                line_parts = line.split(':')
                labels.append(line_parts[0].replace(" ", ""))
                if ("continuous" in line_parts[1]):
                    types.append(1)
                else:
                    types.append(0)
            if (("index" in line) or ("_id" in line)):
                index_count += 1
            line = namefile.readline()
    labels = labels[index_count:]
    types = types[index_count:]
    return [labels, types, index_count]

def removeIndeces(dataset, index_count):
    new_dataset = []
    for row in dataset:
        new_dataset.append(row[index_count:])
    return new_dataset

def formatTestData(datarow, datalabels, datatypes, scale):
    query = {}
    for index in range(len(datalabels)):
        if (datatypes[index] == 0):
            query[datalabels[index]] = datarow[index]
        else:
            query[datalabels[index]] = discretize(datarow[index], scale[index])
    answer = datarow[-1]
    return [query, answer]

# Discretize Continuous Data

def createScale(dataset, datatypes, steps = DISCRETE_STEPS):
    scales = []
    
    for index in range(len(datatypes)):
        scale = []
        if (datatypes[index] == 1):
            max_val = max(column(dataset, index))
            min_val = min(column(dataset, index))
            step = (max_val - min_val)/(steps)
            unit = min_val + step
            while unit < max_val:
                scale.append(unit)
                unit += step
        scales.append(scale)
        
    return scales

def discretize(disc_val, scale):
    if (len(scale) == 0):
        return 0
    if (disc_val < scale[0]):
        return 0
    for i in range(len(scale) - 1):
        if ((scale[i] <= disc_val) and (disc_val < scale[i + 1])):
            return scale[i]
    return scale[-1]

def discretizeColumn(col, scale):
    discrete_col = []
    for item in col:
        discrete_col.append(discretize(item, scale))
    return discrete_col

# Obtain the Split Order

def entropy(Y):
    elements, counts = np.unique(Y, return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def infoGain(col, Y, datatype, scale, index):
    updated_col = col
    if (datatype == 1):
        updated_col = discretizeColumn(col, scale[index])
    
    total_entropy = entropy(Y)
    vals, counts = np.unique(updated_col, return_counts = True)
    
    weighted_entropy = 0
    for val in range(len(vals)):
        matches = [i for i, x in enumerate(updated_col) if x == vals[val]]
        match_entropy = entropy([Y[i] for i in matches])
        weighted_entropy += (counts[val]/np.sum(counts))*match_entropy
        
    info_gain = total_entropy - weighted_entropy
    return info_gain

def giniIndex(col, Y, datatype, scale, index):
    gini_index = 0
    
    updated_col = col
    if (datatype == 1):
        updated_col = discretizeColumn(col, scale[index])
    uniques = np.unique(updated_col)
    for u in range(len(uniques)):
        pos_matches = 0
        neg_matches = 0
        for i in range(len(updated_col)):
            if (updated_col[i] == uniques[u]):
                if (Y[i] == True):
                    pos_matches += 1
                else:
                    neg_matches += 1
        match_total = pos_matches + neg_matches
        gini_index += (1 - (((pos_matches/match_total)**2) + ((neg_matches/match_total)**2))) * (match_total/len(updated_col))
    
    return gini_index

def getAttributeInfo(train_data, datatypes, scale, selection):
    attribute_info = []
    
    Y = column(train_data, -1)
    for x in range(len(train_data[0]) - 1):
        X = column(train_data, x)
        if (selection == "gini_index"):
            info = giniIndex(X, Y, datatypes[x], scale, x)
        else:
            info = infoGain(X, Y, datatypes[x], scale, x)
            if (selection == "gain_ratio"):
                info = info/entropy(X)
        attribute_info.append(info)
    return attribute_info

def getSplitOrder(train_data, datatypes, scale, selection):
    split_order = []
    
    info = getAttributeInfo(train_data, datatypes, scale, selection)
    if (selection == "gain_ratio"):
        print("Gain Ratio: " + str(info))
    elif (selection == "gini_index"):
        print("GINI Index: " + str(info))
    else:
        print("Information Gain: " + str(info))
    
    for i in range(len(info)):
        max_pos = info.index(max(info))
        split_order.append(max_pos)
        info[max_pos] = 0
    return split_order

# Build the Decision Tree

def buildTree(train_data, datalabels, datatypes, scale, split_order, parent_node_class):
    
    # If the training datas or feature list is empty, return the value of the parent node
    if ((len(train_data) == 0) or (len(split_order) == 0)):
        return parent_node_class
    
    # If all target values have the same value, return that value
    if ((len(np.unique(column(train_data, -1)))) == 1):
        return train_data[0][-1]
    
    #Set the default answer as the most common value of the current node
    parent_node_class = np.unique(column(train_data, -1))[np.argmax(np.unique(column(train_data, -1), return_counts=True)[1])]

    # If none of the stopping conditions are met, grow the tree
    feature = split_order[0]
    tree = {datalabels[feature]:{}}
    
    if (datatypes[feature] == 0):
        col = column(train_data, feature)
        for value in np.unique(col):
            subdata = []
            for example in train_data:
                disc_value = example[feature]
                data_value = value
                if (disc_value == data_value):
                    subdata.append(example)
            subtree = buildTree(subdata, datalabels, datatypes, scale, split_order[1:], parent_node_class)
            tree[datalabels[feature]][value] = subtree
        return(tree)
    else:
        col = discretizeColumn(column(train_data, feature), scale[feature])
        for value in np.unique(col):
            value = value
            subdata = []
            for example in train_data:
                disc_value = discretize(example[feature], scale[feature])
                try:
                    data_value = decimal.Decimal(value)
                except:
                    data_value = decimal.Decimal(0)
                if (disc_value == data_value):
                    subdata.append(example)
            subtree = buildTree(subdata, datalabels, datatypes, scale, split_order[1:], parent_node_class)
            tree[datalabels[feature]][value] = subtree
        return tree

def createTree(train_data, datalabels, datatypes, scale, selection):
    split_order = getSplitOrder(train_data, datatypes, scale, selection)
    print("Split Order:" + str(split_order))
    base_node_class = np.unique(column(train_data, -1))[np.argmax(np.unique(column(train_data, -1), return_counts=True)[1])]
    tree = buildTree(train_data, datalabels, datatypes, scale, split_order, base_node_class)
    print("Decision Tree: " + str(tree))
    return tree
    
def predict(query, tree, default = True):
    for key in list(query.keys()):
        if (key in list(tree.keys())):
            try:
                result = tree[key][query[key]]
            except:
                return default
            
            result = tree[key][query[key]]
            
            if (isinstance(result, dict)):
                return predict(query, result)
            else:
                return result