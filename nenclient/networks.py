import zmq
import nengo
from nengo.solvers import LstsqL2
import numpy as np


class Client:
    def __init__(self):
        self.context = zmq.Context()
        print("Connecting to server...")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def put(self, t, x):
        self.socket.send(b"%lf %lf" % (x[0], x[1]))

    def get(self, t):
        message = self.socket.recv()
        nums = [float(i) for i in message.decode().split(",")]
        return np.array(nums)


def posFilter(t, x):
    return np.array([x[0], x[1]])


def velFilter(t, x):
    return np.array([x[2], x[3]])


def stateFilter(t, x):
    return np.array([x[0], x[1], x[2], x[3]])


def goalPosFilter(t, x):
    return np.array([x[4], x[5]])


def goalVelFilter(t, x):
    return np.array([x[6], x[7]])


def goalFilter(t, x):
    return np.array([x[4], x[5], x[6], x[7]])


def velMultFunc(x):
    return [x[0] * (x[0] * x[0] + x[1] * x[1])**.25,
            x[1] * (x[0] * x[0] + x[1] * x[1])**.25]


def vSquaredctrl():
    client = Client()

    # Set up neurons
    model = nengo.Network()

    with model:

        # Neuron ensembles
        vel_e = nengo.Ensemble(1000, dimensions=2, radius=400)
        pos_e = nengo.Ensemble(1000, dimensions=2, radius=100)

        vel_mult = nengo.Ensemble(1000, dimensions=2, radius=400)
        f_ctrl = nengo.Ensemble(1000, dimensions=2, radius=400)

        # Non-neural nodes and functions
        socket_in = nengo.Node(client.get, size_out=8)
        pos_in = nengo.Node(posFilter, size_in=8, size_out=2)
        vel_in = nengo.Node(velFilter, size_in=8, size_out=2)
        goalPos_in = nengo.Node(goalPosFilter, size_in=8, size_out=2)
        goalVel_in = nengo.Node(goalVelFilter, size_in=8, size_out=2)
        socket_out = nengo.Node(client.put, size_in=2)

        # Neural connections
        nengo.Connection(socket_in, pos_in)
        nengo.Connection(socket_in, vel_in)
        nengo.Connection(socket_in, goalPos_in)
        nengo.Connection(socket_in, goalVel_in)

        nengo.Connection(pos_in, pos_e)
        nengo.Connection(goalPos_in, pos_e, transform=[[-1, 0], [0, -1]])
        nengo.Connection(vel_in, vel_e)
        nengo.Connection(goalVel_in, vel_e, transform=[[-1, 0], [0, -1]])
        nengo.Connection(vel_e, vel_mult, function=velMultFunc)

        nengo.Connection(pos_e, f_ctrl)
        nengo.Connection(vel_mult, f_ctrl)
        nengo.Connection(f_ctrl, socket_out)

    return model


def vInhibitctrl(neuronCount=1000):
    client = Client()
    # Set up neurons
    model = nengo.Network()

    with model:

        # Neuron ensembles
        vel_e = nengo.Ensemble(neuronCount, dimensions=2, radius=100)
        pos_e = nengo.Ensemble(neuronCount, dimensions=2, radius=500)
        f_ctrl = nengo.Ensemble(neuronCount, dimensions=2, radius=50)

        # Non-neural nodes and functions
        socket_in = nengo.Node(client.get, size_out=8)
        pos_in = nengo.Node(posFilter, size_in=8, size_out=2)
        vel_in = nengo.Node(velFilter, size_in=8, size_out=2)
        goalPos_in = nengo.Node(goalPosFilter, size_in=8, size_out=2)
        goalVel_in = nengo.Node(goalVelFilter, size_in=8, size_out=2)
        socket_out = nengo.Node(client.put, size_in=2)

        # Neural connections
        nengo.Connection(socket_in, pos_in)
        nengo.Connection(socket_in, vel_in)
        nengo.Connection(socket_in, goalPos_in)
        nengo.Connection(socket_in, goalVel_in)

        nengo.Connection(pos_in, pos_e)
        nengo.Connection(goalPos_in, pos_e, transform=[[-1, 0], [0, -1]])
        nengo.Connection(vel_in, vel_e)
        nengo.Connection(goalVel_in, vel_e, transform=[[-2, 0], [0, -2]])
        nengo.Connection(vel_e, pos_e.neurons,
                         transform=[[.1, .1]] * neuronCount)

        nengo.Connection(pos_e, f_ctrl)
        nengo.Connection(vel_e, f_ctrl)
        nengo.Connection(f_ctrl, socket_out)

    return model


def modelController(neuronCount=100, tau=0.001):
    client = Client()
    # Set up neurons
    model = nengo.Network()

    with model:
        # Neuron ensembles
        error = nengo.Ensemble(neuronCount, dimensions=4, radius=100, label='error')
        control = nengo.Ensemble(neuronCount, dimensions=2, radius=100, label='control')
        enModel = nengo.Ensemble(neuronCount, dimensions=4, radius=100, label='enModel')
        feedback = nengo.Ensemble(neuronCount, dimensions=4, radius=100, label='feedback')
        oracle = nengo.Ensemble(neuronCount, dimensions=4, radius=100)

        # Non-neural nodes and functions
        socket_in = nengo.Node(client.get, size_out=8, label='socket_in')
        state = nengo.Node(stateFilter, size_in=8, size_out=4, label='state')
        goal = nengo.Node(goalFilter, size_in=8, size_out=4, label='goal')
        socket_out = nengo.Node(client.put, size_in=2, label='socket_out')

        # Neural connections

        # Split up socket input into current observed state and goal state
        nengo.Connection(socket_in, state)
        nengo.Connection(socket_in, goal)

        # Create error signal between current state and goal
        nengo.Connection(state, error, synapse=tau)
        nengo.Connection(goal,
                         error,
                         synapse=tau,
                         transform=[[-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, -1]])

        # Naive first estimate of control signal, where f = 1*error_p + 4*error_v
        nengo.Connection(error,
                         control,
                         synapse=tau,
                         solver=LstsqL2(weights=True),
                         transform=[[1, 0, 10, 0],
                                    [0, 1, 0, 10]])

        # Create model of system by double integrating control
        model_conn = nengo.Connection(control,
                                      enModel,
                                      synapse=tau,
                                      solver=LstsqL2(weights=True),
                                      transform=[[0, 0],
                                                 [0, 0],
                                                 [-tau, 0],
                                                 [0, -tau]])
        model_fb_conn = nengo.Connection(enModel,
                                         enModel,
                                         synapse=tau,
                                         solver=LstsqL2(weights=True),
                                         transform=[[1, 0, 1, 0],
                                                    [0, 1, 0, 1],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])
        # Create feedback signals that compare the models to the actual states
        nengo.Connection(enModel,
                         feedback,
                         synapse=tau,
                         solver=LstsqL2(weights=True))
        nengo.Connection(state,
                         feedback,
                         synapse=tau,
                         transform=[[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, -1]])

        # Implement learning rules for models
        model_conn.learning_rule_type = nengo.PES(learning_rate=1e-7)
        model_fb_conn.learning_rule_type = nengo.PES(learning_rate=1e-7)
        nengo.Connection(feedback,
                         model_conn.learning_rule)
        nengo.Connection(feedback,
                         model_fb_conn.learning_rule)

        # Route control signal back out to the socket
        nengo.Connection(control,
                         socket_out,
                         synapse=tau)

    return model


def divider(neuronCount=1000, tau=0.002):
    # Set up neurons
    model = nengo.Network()

    with model:
        dividend = nengo.Node(size_in=1, size_out=1)
        divisor = nengo.Node(size_in=1, size_out=1)
        combine = nengo.Ensemble(neuronCount, dimensions=2, radius=200, label='combine')
        quotient = nengo.Ensemble(neuronCount, dimensions=1, radius=10, label='quotient')

        nengo.Connection(dividend,
                         combine,
                         synapse=tau,
                         transform=[[1], [0]])
        nengo.Connection(divisor,
                         combine,
                         synapse=tau,
                         transform=[[0], [1]])
        nengo.Connection(combine,
                         quotient,
                         synapse=tau,
                         solver=LstsqL2(weights=True),
                         function=lambda x: x[1] / (x[0] + 0.001))

    return model
