import pickle

class A:

    def __init__(self):

        self.Binst = B()


class B:

    def __init__(self):

        self.a = 1
        self.b = 2
        self.c = 3


if __name__ == '__main__':

    ab_list = [A(), A()]

    with open('sample_binary', 'wb') as f:
        pickle.dump(ab_list, f)

    with open('sample_binary', 'rb') as f:
        pickle.load(f)

    print(ab_list[0].Binst.c)
    print(ab_list[1].Binst.b)

