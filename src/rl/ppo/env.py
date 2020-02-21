import abc 

class Env(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def is_done(self):
        pass

    @abc.abstractmethod
    def get_state_as_bytestring(self):
        pass

    @abc.abstractmethod
    def draw(self):
        ''' Draw object or state in environment '''
        pass

    @abc.abstractmethod
    def get_possible_moves(self):
        pass

    @abc.abstractmethod
    def calculate_reward(self):
        pass

    @abc.abstractmethod
    def perform_move(self,move):
        # Perform the move

        # Collect next state

        # Calculate the reward

        # Done: gather status somehow

        # return s_next, reward, done
        pass

    @abc.abstractmethod
    def perform_hashed_move(self,hashed_move):
        move = convert_move_from_hash(hashed_move)
        return perform_move(move)

    @abc.abstractmethod
    def convert_move_from_hash(self,hashed_move):
        pass