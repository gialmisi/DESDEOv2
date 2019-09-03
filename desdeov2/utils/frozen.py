"""Define a decorator to prevent the setting of new attributes for objects
outside of the object's __init__ function. The ability to define these new
attributes outside of __init__ is a very common source for bugs. Therefore, we
get rid of it with the cost a bit of extra overhead, but the payoff in the long
run should be definitely worth it.

Thanks to Yoann's response on the StackOverflow thread I found. My version if a
slight modification with an extra argument given to the decorator for logging
purposes.

(Just complete the SO link)
Source: /questions/3603502/prevent-creating-new-attributes-outside-init

"""

from functools import wraps
from logging import Logger
from typing import Any


def frozen(logger: Logger) -> Any:
    """A decorator to prevent the definition of new attributes outside of that
    object's __init__ function.

    Attributes:
        cls (Class): The class to be decorated
        logger (Logger): A logger object to log debug messages

    Returns:
        Class: A class with frozen attributes

    Usage:
        @frozen(loggername)
        class foo(parent):
            ...

    Note: Typehinting this thing seems to be a nightmare.

    """

    def _frozen(cls: Any) -> Any:
        cls.__frozen = False

        def frozensetattr(self, key, value) -> None:
            """Prevent the definition of new attributes. Log a message if a non
            existing attribute is attempted to be defined.

            """
            if self.__frozen and not hasattr(self, key):
                msg = "Class {} is frozen. Cannot set {} = {}".format(
                    cls.__name__, key, value
                )
                logger.debug(msg)
            else:
                object.__setattr__(self, key, value)

        def init_decorator(func: Any) -> Any:
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                func(self, *args, **kwargs)
                self.__frozen = True

            return wrapper

        cls.__setattr__ = frozensetattr
        cls.__init__ = init_decorator(cls.__init__)

        return cls

    return _frozen
