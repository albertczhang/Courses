import nn
import math


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        flag = True  # flag = true when we still need to update
        while flag:
            flag = False
            for x, y in dataset.iterate_once(1):
                yy = nn.as_scalar(y)
                if self.get_prediction(x) != yy:
                    self.w.update(x, yy)
                    flag = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    num_weights = 1
    batch_size = 20  # must divide 200
    rate = 0.001


    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 2
        self.rate = 0.004
        self.w1 = nn.Parameter(1, 50)
        self.b1 = nn.Parameter(1, 50)
        self.w2 = nn.Parameter(50, 1)
        self.b2 = nn.Parameter(1, 1)
        # self.w3 = nn.Parameter(42, 1)
        # self.b3 = nn.Parameter(1, 1)
        # self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        result = nn.Linear(x, self.w1)

        result = nn.AddBias(result, self.b1)

        result = nn.ReLU(result)

        result = nn.Linear(result, self.w2)
        result = nn.AddBias(result, self.b2)

        # result = nn.ReLU(result)
        # result = nn.Linear(result, self.w3)
        # result = nn.AddBias(result, self.b3)
        return result

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            loss = 1
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad[0], -self.rate)
                self.b1.update(grad[1], -self.rate)
                self.w2.update(grad[2], -self.rate)
                self.b2.update(grad[3], -self.rate)
                # self.w3.update(grad[4], -self.rate)
                # self.b3.update(grad[5], -self.rate)
            if nn.as_scalar(loss) < 0.02:
                return



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 20
        self.rate = 0.02
        self.w1 = nn.Parameter(784, 80)
        self.b1 = nn.Parameter(1, 80)
        self.w2 = nn.Parameter(80, 80)
        self.b2 = nn.Parameter(1, 80)
        self.w3 = nn.Parameter(80, 40)
        self.b3 = nn.Parameter(1, 40)
        self.w4 = nn.Parameter(40, 10)
        self.b4 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        result = nn.Linear(x, self.w1)
        result = nn.AddBias(result, self.b1)

        result = nn.ReLU(result)

        result = nn.Linear(result, self.w2)
        result = nn.AddBias(result, self.b2)

        result = nn.ReLU(result)

        result = nn.Linear(result, self.w3)
        result = nn.AddBias(result, self.b3)

        result = nn.ReLU(result)

        result = nn.Linear(result, self.w4)
        result = nn.AddBias(result, self.b4)


        return result

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        flag1 = True
        flag2 = True
        # flag3 = False
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])

                self.w1.update(grad[0], -self.rate)
                self.b1.update(grad[1], -self.rate)
                self.w2.update(grad[2], -self.rate)
                self.b2.update(grad[3], -self.rate)
                self.w3.update(grad[4], -self.rate)
                self.b3.update(grad[5], -self.rate)
                self.w4.update(grad[6], -self.rate)
                self.b4.update(grad[7], -self.rate)
                # if flag3 and dataset.get_validation_accuracy() > 0.973:
                #     return

            if dataset.get_validation_accuracy() > 0.96 and flag1:
                print("changing")
                self.rate = 0.005
                self.batch_size = 15
                flag1 = False
            if dataset.get_validation_accuracy() > 0.97 and flag2:
                print("changing")
                self.rate = 0.001
                self.batch_size = 20
                flag2 = False
                # flag3 = True
            if dataset.get_validation_accuracy() > 0.973:
                return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"


        self.batch_size = 10
        self.rate = 0.01
        self.f_init = nn.ReLU
        self.wr = nn.Parameter(47, 47)
        self.wh = nn.Parameter(47, 47)
        self.w1 = nn.Parameter(47, 80)
        self.b1 = nn.Parameter(1, 80)
        self.w2 = nn.Parameter(80, 5)
        self.b2 = nn.Parameter(1, 5)



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h_i = self.f_init(xs[0])
        for i in range(len(xs) - 1):
            h_i = nn.Add(nn.Linear(xs[i + 1], self.wr), nn.Linear(h_i, self.wh))

        result = nn.Linear(h_i, self.w1)
        result = nn.AddBias(result, self.b1)
        result = nn.ReLU(result)

        result = nn.Linear(result, self.w2)
        result = nn.AddBias(result, self.b2)
        # print("done")
        return result

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                grad = nn.gradients(loss, [self.wh, self.wr, self.w1, self.b1, self.w2, self.b2])

                self.wh.update(grad[0], -self.rate)
                self.wr.update(grad[1], -self.rate)
                self.w1.update(grad[2], -self.rate)
                self.b1.update(grad[3], -self.rate)
                self.w2.update(grad[4], -self.rate)
                self.b2.update(grad[5], -self.rate)

            if dataset.get_validation_accuracy() > 0.82:
                return
