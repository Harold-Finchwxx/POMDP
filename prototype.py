import random

import numpy as np
import pomdp_py as pp
import math


class State(pp.Observation):
    def __init__(self, human_mode, human_strategy, continuous_flag=0, continuous_period=4, v_upperbound=60.0 / 3.6,
                 v_lowerbpound=-5 / 3.6, t_delta=0.2, v1=20 / 3.6, v2=20 / 3.6, p1=-4, p2=-4, a1=0.0, a2=0.0, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.p1 = p1
        self.p2 = p2

        """
        if random.uniform(0, 1) < 0.5:
            self.mod = 'aggressive'
        else:
            self.mod = 'conservative'
        """

        if human_mode != 'aggressive' and human_mode != 'conservative':
            raise ValueError('Invalid human mode {}'.format(human_mode))
        else:
            self.human_mode = human_mode

        self.v1 = v1
        self.v2 = v2
        self.a1 = a1
        self.a2 = a2

        if self.v1 > v_upperbound:
            self.v1 = v_upperbound
        if self.v2 < v_upperbound:
            self.v2 = v_upperbound

        if self.v1 < v_lowerbpound:
            self.v1 = v_lowerbpound
        if self.v2 < v_lowerbpound:
            self.v1 = v_lowerbpound

        if human_strategy != "go" and human_strategy != "stop":
            raise ValueError('Invalid human strategy {}'.format(human_strategy))
        else:
            self.human_strategy = human_strategy

        self.continuous_period = continuous_period
        self.continuous_flag = continuous_flag % continuous_period

        self.t_delta = t_delta

        self.vector = [self.v1, self.v2, self.p1, self.p2]

        self.show = {'human_mode': self.human_mode,
                     'human_strategy': self.human_strategy,
                     'human_position': self.p1,
                     'human_velocity': self.v1,
                     'auto_vehicle_position': self.p2,
                     'auto_vehicle_velocity': self.v2}


class Action(pp.Action):
    def __init__(self, name):
        if name != 'go' and name != 'stop':
            raise ValueError('Invalid action name')
        self.name = name


class Observation(pp.Observation):
    def __init__(self, v1, v2, p1, p2, human_strategy):
        self.v1 = v1
        self.v2 = v2
        self.p1 = p1
        self.p2 = p2
        self.vector = [v1, v2, self.p1, self.p2]

        if human_strategy != 'go' and human_strategy != 'stop':
            raise ValueError('Invalid human strategy {}'.format(human_strategy))
        else:
            self.human_strategy = human_strategy


class ObservationModel(pp.ObservationModel):
    def __init__(self, noise_coeff=0.1, obser_accuracy=0.95, high_threshold=10 / 3.6, low_threshold=5 / 3.6):
        self.noise_coeff = noise_coeff
        self.obser_accuracy = obser_accuracy
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def probability(self, observation, next_state, action):
        if observation.human_strategy == next_state.human_strategy:
            return self.obser_accuracy
        else:
            return 1 - self.obser_accuracy

    def sample(self, next_state, action):
        noise_v1 = next_state.v1 * random.gauss(0, self.noise_coeff)
        noise_v2 = next_state.v2 * random.gauss(0, self.noise_coeff)
        noise_p1 = next_state.p1 * random.gauss(0, self.noise_coeff)
        noise_p2 = next_state.p2 * random.gauss(0, self.noise_coeff)

        threshold = random.uniform(0, 1)
        if threshold < self.obser_accuracy:
            observation_human_strategy = next_state.human_strategy
        else:
            if next_state.human_strategy == 'go':
                observation_human_strategy = 'stop'
            else:
                observation_human_strategy = 'go'

        return Observation(next_state.v1 + noise_v1, next_state.v2 + noise_v2, next_state.p1 + noise_p1,
                           next_state.p2 + noise_p2, observation_human_strategy)


class KinematicsModel:
    def __init__(self, v, a, p, t_delta):
        self.v, self.a = v, a
        self.p = p

        self.p = self.p + self.v * t_delta + 0.5 * self.a * (t_delta ** 2)
        self.v = self.v + self.a * t_delta


class TransitionModel(pp.TransitionModel):
    def __init__(self, strategy_inherit_factor=0.95,
                 agress_go_go=0.9,
                 agress_stop_go=1 - 1e-4,
                 conser_go_go=0.1,
                 conser_stop_go=0.7):
        super(TransitionModel, self).__init__()
        self.agress_go_go = agress_go_go
        self.agress_stop_go = agress_stop_go
        self.conser_go_go = conser_go_go
        self.conser_stop_go = conser_stop_go
        self.strategy_inherit_factor = strategy_inherit_factor

    def probability(self, next_state, state, action):
        if next_state.human_strategy == state.human_strategy:
            return self.strategy_inherit_factor
        else:
            return 1 - self.strategy_inherit_factor

    def sample(self, state, action):
        if state.continuous_flag > 0:
            if random.uniform(0, 1) < self.strategy_inherit_factor:
                human_strategy = state.human_strategy
            else:
                if state.human_strategy == 'go':
                    human_strategy = 'stop'
                else:
                    human_strategy = 'go'
        else:

            if state.human_mode == 'aggressive':
                if action.name == 'go':

                    if random.uniform(0, 1) < self.agress_go_go:
                        human_strategy = 'go'
                    else:
                        human_strategy = 'stop'

                if action.name == 'stop':

                    if random.uniform(0, 1) < self.agress_stop_go:
                        human_strategy = 'go'
                    else:
                        human_strategy = 'stop'

            else:
                if action.name == 'go':

                    if random.uniform(0, 1) < self.conser_go_go:
                        human_strategy = 'go'
                    else:
                        human_strategy = 'stop'

                if action.name == 'stop':

                    if random.uniform(0, 1) < self.conser_stop_go:
                        human_strategy = 'go'
                    else:
                        human_strategy = 'stop'

        if human_strategy == 'go':
            a1 = random.gauss(2.8, 0.81)
        else:
            a1 = random.gauss(-6, 0.49)

        if action.name == 'go':
            a2 = random.gauss(2.8, 0.81)
        else:
            a2 = random.gauss(-6, 0.81)

        auto_vehicle = KinematicsModel(state.v2, a2, state.p2, state.t_delta)
        human_vehicle = KinematicsModel(state.v1, a1, state.p1, state.t_delta)

        continuous_flag = state.continuous_flag
        continuous_flag += 1

        return State(human_mode=state.human_mode, human_strategy=human_strategy,
                     continuous_flag=continuous_flag, continuous_period=state.continuous_period,
                     v_upperbound=state.v_upperbound, v_lowerbound=state,
                     t_delta=state.t_delta,
                     a1=a1, a2=a2,
                     v1=human_vehicle.v, v2=auto_vehicle.v,
                     p1=human_vehicle.p, p2=auto_vehicle.p)


class PolicyModel(pp.RolloutPolicy):
    ACTIONS = {Action(act)
               for act in {'go', 'stop'}}

    def sample(self, state):
        return random.sample(self.ACTIONS, 1)[0]

    def rollout(self, state, *args, **kwargs):
        return self.sample(state)

    def get_all_actions(self, *args, state=None, history=None, **kwargs):
        return PolicyModel.ACTIONS


class RewardModel(pp.RewardModel):
    def __init__(self, time_cost=0.5, distance_cost=100.0, distance_scale=0.7):
        super(RewardModel, self).__init__()
        self.time_cost = time_cost
        self.distance_cost = distance_cost
        self.distance_scale = distance_scale

    def _reward_func(self, state, action):
        return (self.time_cost * state.t_delta +
                self.distance_cost * math.log(self.distance_scale * abs(state.p1 - state.p2)))

    def sample(self, state, action, next_state):
        return self._reward_func(state, action)


class HumaninLoopProblem(pp.POMDP):

    def __init__(self, init_true_state=None, init_belief=None,
                 human_mode=None, continuous_period=4,
                 v_upperbound=60.0 / 3.6, v_lowerbpound=-5 / 3.6,
                 t_delta=0.2, v1_init=20 / 3.6, v2_init=20 / 3.6,
                 p1_init=-4, p2_init=-4, a1_init=0.0, a2_init=0.0,
                 noise_coeff=0.1, obser_accuracy=0.95,
                 high_threshold=10 / 3.6, low_threshold=5 / 3.6,
                 strategy_inherit_factor=0.95,
                 agress_go_go=0.9,
                 agress_stop_go=1 - 1e-4,
                 conser_go_go=0.1,
                 conser_stop_go=0.7,
                 time_cost=0.5, distance_cost=100.0, distance_scale=0.7):
        self.init_true_state = random.choice([State(human_mode='aggressive', human_strategy='go',
                                                    continuous_period=continuous_period),
                                              State(human_mode='aggressive', human_strategy='stop',
                                                    continuous_period=continuous_period),
                                              State(human_mode='conservative', human_strategy='go',
                                                    continuous_period=continuous_period),
                                              State(human_mode='conservative', human_strategy='stop',
                                                    continuous_period=continuous_period)])

        self.init_belief = pp.Histogram({State(human_mode='aggressive', human_strategy='go',
                                               continuous_period=continuous_period): 0.25,
                                         State(human_mode='aggressive', human_strategy='stop',
                                               continuous_period=continuous_period): 0.25,
                                         State(human_mode='conservative', human_strategy='stop',
                                               continuous_period=continuous_period): 0.25,
                                         State(human_mode='conservative', human_strategy='go',
                                               continuous_period=continuous_period): 0.25})

        if init_belief is not None:
            self.init_belief = init_belief
        if self.init_true_state is not None:
            self.init_true_state = init_true_state

        agent = pp.Agent(init_belief,
                         PolicyModel(),
                         TransitionModel(),
                         ObservationModel(),
                         RewardModel())

        env = pp.Environment(init_true_state,
                             TransitionModel(),
                             RewardModel())

        super().__init__(agent, env, name="HumaninLoopProblem")


def test_planner(problem_instance, planner, time_constraint=30, t_delta=0.2, position_constraint=4):
    num_steps = time_constraint / t_delta
    i = 0
    while i < num_steps and problem_instance.env.state.p2 < position_constraint:
        i += 1

        action = planner.plan(problem_instance.agent)

        print("====Step {}=====".format(i))
        print("True state:", problem_instance.env.state.show)
        print("Belief state:", problem_instance.agent.cur_belief)
        print("Action:", action)

        reward = problem_instance.env.state_transition(action, execute=True)
        print("Reward:", reward)

        real_observation = problem_instance.agent.observation_model.sample(human_in_loop_problem.env.state, action)
        print(">>Real observation:", real_observation)

        problem_instance.agent.update_history(action, real_observation)
        planner.update(problem_instance.agent, action, real_observation)
        if isinstance(planner, pp.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
        if isinstance(problem_instance.agent.cur_belief, pp.Histogram):
            new_belief = pp.update_histogram_belief(problem_instance.agent.cur_belief,
                                                    action, real_observation,
                                                    problem_instance.agent.observation_model,
                                                    problem_instance.agent.transition_model)

            problem_instance.agent.set_belief(new_belief)


if __name__ == "__main__":
    human_in_loop_problem = HumaninLoopProblem()

    pouct = pp.POUCT(max_depth=10, discount_factor=0.95,
                     planning_time=10.0, exploration_const=110,
                     rollout_policy=human_in_loop_problem.agent.policy_model)

    pomcp = pp.POMCP(max_depth=10, discount_factor=0.95,
                     planning_time=10.0, exploration_const=110,
                     rollout_policy=human_in_loop_problem.agent.policy_model)

    test_planner(human_in_loop_problem, pouct)
