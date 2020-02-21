class Move():

    def __init__(self, pos, direction):
        self._pos = pos
        self._direction = direction

    def get_position(self):
        return self._pos

    def get_direction(self):
        return self._direction

    def __hash__(self):
        return self._pos[0] + self._pos[1]*13 + self._direction*13**2

    def __str__(self):
        return "Position: (" + str(self._pos[0]) + "," + str(self._pos[1]) + "), Direction: " + str(self._direction)