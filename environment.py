import numpy as np
import pandas as pd
from utils import shift_array


DATA = pd.read_csv("data.csv", index_col="family_id")


class ActionSpace:

    def __init__(self, dimension):
        self.n = dimension

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace:

    def __init__(self, dimension):
        self.data = DATA
        self.n = dimension

    def sample(self):
        sample_state = np.zeros(self.n, dtype=np.int_)
        # sample families positioned
        steps = np.random.randint(0, len(self.data)-1)  # -1 to ensure that at least one family is left to be positioned
        positioned_families = self.data.sample(n=steps)
        # distribute families over the booking state
        for _, family in positioned_families.iterrows():
            position = 11 + np.random.randint(0, 100)
            sample_state[position] += family["n_people"]
        # sample ongoing family
        ongoing_family = self.data[~self.data.index.isin(positioned_families.index)].sample(n=1)
        sample_state[:11] = ongoing_family.values

        return sample_state


class Environment:

    """ Modelisation of the problem as an environment, follows OpenAI gym API. """

    # a state is then given by [family_choice0, ..., family_choice9, family_size, n_people_day_100, ..., n_people_day_0]
    # Hence the family part of the state are the first 11 numbers and the booking part is the following 100 numbers.
    STATE_DIM = 10 + 1 + 100
    # we can place a family at any of the 100 possible days
    # actions are discrete from 0 to 99 and represent the fact that we put
    # a family to a given day with the action being an alias for the day number.
    # From the problem day are sorted from day100 to day1.
    # For us they are indexed from 0 to 99 with 0 being day100 and 99 being day1
    ACTION_DIM = 100
    # constraints
    MIN_PER_DAY = 125
    MAX_PER_DAY = 300

    def __init__(self):
        self._seed = None
        self._families = None
        self._state = None
        # extra attributes for gym compatibility
        self.action_space = ActionSpace(self.ACTION_DIM)
        self.observation_space = ObservationSpace(self.STATE_DIM)
        # extra data for kaggle submission
        self.family_index = None

    def set_seed(self, seed):
        assert isinstance(seed, int), f"seed should be an int, given type is {type(seed)}"
        self._seed = seed

    def reset(self):
        """ resets the environment and return an initial state """
        self._families = DATA.copy()
        self._state = np.zeros(100 + 10 + 1, dtype=np.int_)
        self._next_family_state()
        return self._state

    def _next_family_state(self):
        """ draws a family from the non positioned families and sets it to the family part of
        the state """
        if len(self._families > 0):
            # sample a family
            family = self._families.sample(n=1, random_state=self._seed)
            # remove the family from the families to be placed
            self._families.drop(family.index, inplace=True)
            # update the family part of the state vector
            self._state[:11] = family.values
            self.family_index = family.index[0]
        else:
            self._state[:11] = np.zeros(11)
            self.family_index = None

    def _next_booking_state(self, action):
        """ sets the family of the current state to the given day by 'action' """
        family_size = self._get_family_size()
        day_index = self._get_day_index(action)
        self._state[day_index] += family_size

    def _next_state(self, action):
        """ sets the next state according to the current state and the action taken """
        self._next_booking_state(action)
        self._next_family_state()

    def _get_family_size(self):
        """ gets the family size data from the current state. """
        return self._state[10]

    def _get_day_index(self, action):
        """ gets the state index of the corresponding action. (offset from the family part) """
        return 11 + action

    def _get_day_occupancy(self):
        """ gets the occupancy of each day. (offset from the family part)
         the vector has the form [n_people_day_100, ..., n_people_day_1] """
        return self._state[11:]

    def _get_family_data(self):
        """ returns the choices for the family (array) and the number of people in the family (int).
        the vector of choices has the form [choice_0, ..., choice_9]. """
        return self._state[:10], self._state[10]

    @staticmethod
    def get_day_from_action(action):
        """ We encode actions from 0 to 99 but day are given from 1 to 100. with 100 being
        the farthest day from christmas. This function maps an action to the days from the problem. """
        return 100 - action

    def _rank_choice(self, choices, action):
        """ finds the rank of the given action according to family choices. -1 means that the given
        action is not part of the family choices """
        day = self.get_day_from_action(action)
        rank, = np.where(choices == day)
        if rank.size == 0:
            return -1
        else:
            return rank[0]  # there is one fake dim in the output of np.where, hence the [0]

    @staticmethod
    def _preference_cost(choice_rank, family_size):
        if choice_rank == 0:
            return 0
        elif choice_rank == 1:
            return -50
        elif choice_rank == 2:
            return -(50 + 9 * family_size)
        elif choice_rank == 3:
            return -(100 + 9 * family_size)
        elif choice_rank == 4:
            return -(200 + 9 * family_size)
        elif choice_rank == 5:
            return -(200 + 18 * family_size)
        elif choice_rank == 6:
            return -(300 + 18 * family_size)
        elif choice_rank == 7:
            return -(300 + 36 * family_size)
        elif choice_rank == 8:
            return -(400 + 36 * family_size)
        elif choice_rank == 9:
            return -(500 + 36 * family_size + 199 * family_size)
        else:
            return -(500 + 36 * family_size + 398 * family_size)

    def _accounting_penalty(self):
        occupancy = self._get_day_occupancy()
        shifted_occupancy = shift_array(occupancy)
        exponents = 0.5 + (1/50) * np.abs(occupancy - shifted_occupancy)
        costs = (1/400) * (occupancy - 125) * (occupancy ** exponents)
        return -costs.sum()

    def _min_constraint_penalty(self):
        occupancy = self._get_day_occupancy()
        if not np.all(occupancy >= self.MIN_PER_DAY):
            broken_number = len(np.where(occupancy < self.MIN_PER_DAY)[0])
            return -broken_number * 2000
        else:
            return 0

    def _max_constraint_penalty(self):
        occupancy = self._get_day_occupancy()
        if not np.all(occupancy <= self.MAX_PER_DAY):
            broken_number = len(np.where(occupancy > self.MAX_PER_DAY)[0])
            return -2000 * broken_number
        else:
            return 0

    def _get_reward(self, action):
        choices, size = self._get_family_data()
        rank = self._rank_choice(choices, action)
        return self._preference_cost(rank, size)

    def step(self, action):
        assert isinstance(action, int) and action in range(0, 100), \
            f"action should be an int in the range 0 to 100, given action is {action}"
        reward = self._get_reward(action)   # as we use self, must be called before _next_state()
        done = True if len(self._families) == 0 else False
        self._next_state(action)
        # we can check if we violate the max constraint in the arrival state and directly send a large
        # negative reward to the agent
        reward += self._max_constraint_penalty()
        if done:
            # for the accounting cost, we take it into account at the end of the episode as it is computed
            # from the global distribution of families.
            reward += self._accounting_penalty()
            # for the min constraint, we have to wait for all families to be positioned
            # as we do for the max constraint, we send a large negative reward if its violated
            reward += self._min_constraint_penalty()
        return self._state, reward, done, None
