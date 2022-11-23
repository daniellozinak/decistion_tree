import csv
from treelib import Tree


def load_data(filename):
    output = []
    with open(filename) as csv_file:
        parsed_file = csv.reader(csv_file, delimiter=";")
        for row in parsed_file:
            temp_row = []
            for variable in row:
                try:
                    temp_row.append(float(variable))
                except:
                    temp_row.append(variable)
            if len(temp_row) > 0:
                output.append(temp_row)

    return output


# calculates gini index for a condition
def gini(data, condition, classes=[0, 1, 2], class_index=4):
    # matrix as a counter for each class
    sub_table_a = [0] * len(classes)
    sub_table_b = [0] * len(classes)
    table = [sub_table_a, sub_table_b]

    # initialize variables
    gini_yes = 1
    gini_no = 1

    for row in data:
        # get class index
        index = (list(classes).index(row[class_index]))
        # add one to class counter based on if condition is fulfilled or not
        table[0 if condition(row) else 1][index] += 1

    n_table_a = sum(sub_table_a)
    n_table_b = sum(sub_table_b)

    if n_table_a == 0 or n_table_b == 0 or len(data) == 0:
        return 1

    for cls in sub_table_a:
        gini_yes -= (cls / n_table_a) ** 2
    for cls in sub_table_b:
        gini_no -= (cls / n_table_b) ** 2

    return (gini_yes * n_table_a / len(data)) + \
           (gini_no * n_table_b / len(data))


# generates the best condition of an attribute at 'row_index'
def choose_condition(data, row_index, classes=[0, 1, 2], class_index=4):
    values = [row[row_index] for row in data]
    values.sort()
    previous = -1
    min_gini = 1
    min_index = 0
    for index in range(len(values) - 1):
        avg = (values[index] + values[index + 1]) / 2

        if avg != previous:
            condition = lambda row: row[row_index] < avg
            gini_index = gini(data, condition, classes, class_index)
            if gini_index < min_gini:
                min_gini = gini_index
                min_index = index
        previous = avg

    # print("RESULT")
    # print(f"{values[min_index]} : {min_gini}")
    return min_gini, lambda row: row[row_index] < values[min_index]


# splits data based on 'condition'
def split_data(data, condition):
    is_condition = [row for row in data if condition(row)]
    not_condition = [row for row in data if not condition(row)]
    return is_condition, not_condition


def get_class_members(data, class_index):
    return set([row[class_index] for row in data])


def get_decision_tree(data, classes, class_index, tree, side, prev_side):
    # 1 iteration
    min_gini = 1
    condition = None
    cls = None

    # iterate over attributes and pick the best condition
    for i in range(class_index):
        current_gini = choose_condition(data, i, classes, class_index)[0]
        if current_gini < min_gini:
            min_gini = current_gini
            condition = choose_condition(data, i, classes, class_index)[1]

    # pick chosen class
    classes = [row[class_index] for row in data]
    cls = max(set(classes), key=classes.count)

    # if node is root
    if side is None:
        tree.create_node([cls, min_gini, data, condition], "root")
    else:
        prev_side = "root" if prev_side is None else prev_side
        tree.create_node([cls, min_gini, data, condition], side, parent=prev_side)

    if min_gini == 0 or min_gini == 1:
        return

    split = split_data(data, condition)
    left = split[0]
    right = split[1]

    if len(left) == 0 or len(right) == 0:
        return

    get_decision_tree(left, classes, class_index, tree, "l" if side is None else side + "l", side)
    get_decision_tree(right, classes, class_index, tree, "r" if side is None else side + "r", side)


def classify(tree, row):
    node = tree.get_node("root")

    cls_index = 0
    gini_index = 1
    condition_index = 3

    # while node is not a leaf
    while node.tag[gini_index] > 0:
        if len(node.successors(tree.identifier)) == 0:
            break
        # GO LEFT
        if node.tag[condition_index](row):
            next = node.successors(tree.identifier)[0]
            node = tree.get_node(next)

        # GO RIGHT
        else:
            next = node.successors(tree.identifier)[1]
            node = tree.get_node(next)

    return node.tag[cls_index]


def get_statistics(tree, testing_data, class_index):
    predicted = 0
    for row in testing_data:
        predicted_class = classify(tree, row)
        if predicted_class == row[class_index]:
            predicted += 1
    print(f"success rate is {predicted}/{len(testing_data)} ")


# executes decision tree
# assumes class is at LAST index !!!
def execute():
    training_data = load_data("data/iris.csv")[:120]
    testing_data = load_data("data/iris.csv")[-30:]
    class_index = len(training_data[0]) - 1
    classes = get_class_members(training_data, class_index)

    tree = Tree()
    # node of the tree defined as [class ,gini index, data, condition]
    get_decision_tree(training_data, classes, class_index, tree, None, "root")
    print("generated tree")
    tree.show()
    get_statistics(tree, testing_data, class_index)


execute()
