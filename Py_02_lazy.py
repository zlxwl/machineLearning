
class Lazy(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property



class Circle(object):
    def __init__(self, raduis):
        self.raduis = raduis

    # @Lazy
    @lazy_property
    def aera(self):
        print('evalute')
        return self.raduis * 3.14 ** 2


if __name__ == "__main__":
    c = Circle(4)
    # print(c.raduis)
    # print(c.aera)
    # print(c.aera)
    # print(c.aera)

    print("before")
    print(c.__dict__)
    c.aera
    print("after")
    print(c.__dict__)
