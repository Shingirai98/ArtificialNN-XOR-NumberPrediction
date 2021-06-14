import random
from Perceptron import Perceptron

#
A = 0
B = 0
NOT = Perceptron(1, bias=0.75)
OR = Perceptron(2, bias=-0.5)
AND = Perceptron(2, bias=-1.5)


def OR_P():
    generate_training_set = True
    num_train = 2000
    generate_validation_set = True
    num_valid = 100

    training_examples = [[1.0, 1.0],
                         [1.0, 0.0],
                         [0.0, 1.0],
                         [0.0, 0.0]]

    training_labels = [1.0, 1.0, 1.0, 0.0]
    validate_examples = training_examples
    validate_labels = training_labels

    if generate_training_set:

        training_examples = []
        training_labels = []
        low = random.uniform(-0.25, 0.25)
        high = random.uniform(0.75, 1.25)
        for i in range(num_train):
            training_examples.append([random.choice((low, high)), random.choice((low, high))])
            training_labels.append(1.0 if training_examples[i][0] >= 0.76 or training_examples[i][1] >= 0.76 else 0.0)

        if generate_validation_set:
            validate_examples = []
        validate_labels = []

        for i in range(num_train):
            validate_examples.append([random.choice((random.uniform(-0.25, 0.25), random.uniform(0.75, 1.25))),
                                      random.choice((random.uniform(-0.25, 0.25), random.uniform(0.75, 1.25)))])
            validate_labels.append(1.0 if validate_examples[i][0] >= 0.76 or validate_examples[i][1] >= 0.76 else 0.0)

    valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        OR.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)  # Validate it
        if i == 2000:
            break

    return ""  # OR.weights


def AND_P():
    generate_training_set = True
    num_train = 2000
    generate_validation_set = True
    num_valid = 100

    training_examples = [[1.0, 1.0],
                         [1.0, 0.0],
                         [0.0, 1.0],
                         [0.0, 0.0]]

    training_labels = [1.0, 0.0, 0.0, 0.0]

    validate_examples = training_examples
    validate_labels = training_labels

    if generate_training_set:

        training_examples = []
        training_labels = []

        for i in range(num_train):
            training_examples.append([random.uniform(-0.26, 1.26), random.uniform(-0.26, 1.26)])
            # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
            training_labels.append(1.0 if training_examples[i][0] >= 0.76 and training_examples[i][1] >= 0.76 else 0.0)

    if generate_validation_set:

        validate_examples = []
        validate_labels = []

        for i in range(num_train):
            validate_examples.append([random.uniform(-0.26, 1.26), random.uniform(-0.26, 1.26)])
            validate_labels.append(1.0 if validate_examples[i][0] >= 0.76 and validate_examples[i][1] >= 0.76 else 0.0)

    valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
		# print('------ Iteration ' + str(i) + ' ------')
		# print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)  # Validate it
		# print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 2000:
            break

    return ""  # AND.weights


def NOT_P():
    training_examples = [1.0, 0.0]
    training_labels = [0.0, 1.0]
    validate_examples = training_examples
    validate_labels = training_labels
    training_examples = []
    training_labels = []
    num_train = 2000
    for i in range(num_train):
        training_examples.append(random.uniform(-0.26, 1.26))
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
        training_labels.append(1.0 if training_examples[i] <= 0.76 else 0.0)

    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        validate_examples.append(random.uniform(-0.26, 1.26))
        validate_labels.append(1.0 if validate_examples[i] <= 0.76 else 0.0)

    valid_percentage = NOT.validateNOT(validate_examples, validate_labels, verbose=True)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        NOT.trainNOT(training_examples, training_labels, 0.2)  # Train our Perceptron
        # print('------ Iteration ' + str(i) + ' ------')
		# print(NOT.weights)
        valid_percentage = NOT.validateNOT(validate_examples, validate_labels, verbose=True)  # Validate it
		# print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 2000:
            break

    return ""  # NOT.weights


print("Training GATE_0...")
OR_P()
print("Training GATE_1...")
AND_P()
# NOT = Perceptron(1, bias=-1)
print("Training GATE_2...")
NOT_P()
print("Constructing Network...")
C = AND.activate([A, B])
D = OR.activate([A, B])
E = NOT.activateNOT(C)
XOR_Output =  AND.activate([E, D])

print("AND weights:", AND.weights)
print("AND-Out:", C)
print("OR weights:", OR.weights)
print("OR-Out:", D)
print("NOT weight:", NOT.weights)
print("Not-Out:", E)
print("Output:", AND.activate([E, D]))
