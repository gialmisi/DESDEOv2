"""This is for purely testing.

"""


class Foo(object):
    """Documentation for Test

    """
    def __init__(self):
        self.__prop: int = None

    @property
    def prop(self):
        return self.__prop

    @prop.setter
    def prop(self, val):
        self.__prop = val
