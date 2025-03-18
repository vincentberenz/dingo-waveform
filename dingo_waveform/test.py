from functools import cached_property, wraps


def reinitialize_cached(func):
    @wraps(func)
    def wrapper(self, value):
        func(self, value)
        self._cached = Cached(self._a, self._b)

    return wrapper


class Cached:

    def __init__(self, a: int, b: int) -> None:
        self._construct(a, b)

    def _construct(self, a: int, b: int) -> None:
        self._a = a
        self._b = b

    @cached_property
    def get_sum(self) -> int:
        print("\tcomputing sum:", self._a, self._b)
        return self._a + self._b


class User:

    def __init__(self, a: int, b: int) -> None:
        self._a = a
        self._b = b
        self._cached = Cached(a, b)

    @property
    def a(self) -> int:
        return self._a

    @a.setter
    @reinitialize_cached
    def a(self, value: int) -> None:
        self._a = value

    @property
    def b(self) -> int:
        return self._b

    @b.setter
    @reinitialize_cached
    def b(self, value: int) -> None:
        self._b = value

    def get_sum(self) -> int:
        return self._cached.get_sum


if __name__ == "__main__":

    u = User(1, 2)

    v = u.get_sum()
    print(v)
    v = u.get_sum()
    print(v)

    u.a = 2

    v = u.get_sum()
    print(v)
    v = u.get_sum()
    print(v)
