import random
from Perceptron import Perceptron

#
A = eval(input("Enter first input: "))
B = eval(input("Enter second input: "))
NOT = Perceptron(1, bias=1)
OR = Perceptron(2, bias=-0.5)
AND = Perceptron(2, bias=-1.5)


def OR_P():
    generate_training_set = True
    num_train = 500
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

        for i in range(num_train):
            training_examples.append([random.random(-0.25,1.25), random.random(-0.25,1.25)])
            # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
            training_labels.append(1.0 if training_examples[i][0] >= 0.76 or training_examples[i][1] >= 0.76 else 0.0)

    if generate_validation_set:

        validate_examples = []
        validate_labels = []

        for i in range(num_train):
            validate_examples.append([random.random(-0.25,1.25), random.random(-0.25,1.25)])
            validate_labels.append(1.0 if validate_examples[i][0] >= 0.76 or validate_examples[i][1] >= 0.76 else 0.0)


    valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        OR.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(OR.weights)
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)  # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 500:
            break

    return OR.weights


def AND_P():
    generate_training_set = True
    num_train = 100
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
            training_examples.append([random.random(-0.25,1.25), random.random(-0.25,1.25)])
            # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
            training_labels.append(1.0 if training_examples[i][0] >= 0.76 and training_examples[i][1] >= 0.76 else 0.0)

    if generate_validation_set:

        validate_examples = []
        validate_labels = []

        for i in range(num_train):
            validate_examples.append([random.uniform(-0.25,1.25), random.uniform(-0.25,1.25)])
            validate_labels.append(1.0 if validate_examples[i][0] >= 0.76 and validate_examples[i][1] >= 0.76 else 0.0)

    valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)  # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 500:
            break

    return AND.weights


def NOT_P():
    training_examples = [1.0, 0.0]
    training_labels = [0.0, 1.0]
    validate_examples = training_examples
    validate_labels = training_labels
    training_examples = []
    training_labels = []
    num_train = 100
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
        print('------ Iteration ' + str(i) + ' ------')
        print(NOT.weights)
        valid_percentage = NOT.validateNOT(validate_examples, validate_labels, verbose=True)  # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 500:
            break

    return NOT.weights


print("The Weights for the OR gate is: ", OR_P())
print("The Weights for the AND gate is: ",AND_P())
# NOT = Perceptron(1, bias=-1)
# print("The Weights for the NOT gate is: ", NOT_P())

C = AND.activate([A, B])
print("AND-Out:", C)
D = OR.activate([A, B])
print("OR-Out:", D)
#
E = NOT.activateNOT(C)
print("Not-Out:", E)
#
print("Output:", AND.activate([E, D]))
# create Perceptrons
# AND_G = Perceptron(2, bias=-1.0)
# OR_G = Perceptron(2, bias=-1.0)
# NOT_G = Perceptron(1, bias=-1.0)
# Train perceptrons

# Combine Perceptrons

# Validate XOR Gate


# generate_training_set = False
# num_train = 100
# generate_validation_set = False
# num_valid = 100
#
# training_examples = [[1.0, 1.0],
# 					[1.0, 0.0],
# 					[0.0, 1.0],
# 					[0.0, 0.0]]
#
# training_labels = [0.0, 1.0, 1.0, 0.0]
#
# validate_examples = training_examples
# validate_labels = training_labels
#
#
# if generate_training_set:
#
# 	training_examples = []
# 	training_labels = []
#
# 	for i in range(num_train):
# 		training_examples.append([random.random(), random.random()])
# 		# We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
# 		training_labels.append(1.0 if training_examples[i][0] > 0.8 and training_examples[i][1] > 0.8 else 0.0)
#
# if generate_validation_set:
#
# 	validate_examples = []
# 	validate_labels = []
#
# 	for i in range(num_train):
# 		validate_examples.append([random.random(), random.random()])
# 		validate_labels.append(1.0 if validate_examples[i][0] > 0.8 and validate_examples[i][1] > 0.8 else 0.0)
#
#
# # Create Perceptron
# AND = Perceptron(2, bias=-1.0)
#
# print(AND.weights)
# valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)
# print(valid_percentage)
# i = 0
# while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
#
# 	i += 1
#
# 	AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
# 	print('------ Iteration ' + str(i) + ' ------')
# 	print(AND.weights)
# 	valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True) # Validate it
# 	print(valid_percentage)
#
# 	# This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
# 	# You shouldn't need to do this as your networks may require much longer to train.
# 	if i == 500:
# 		break
