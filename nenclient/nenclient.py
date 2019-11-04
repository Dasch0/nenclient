#
#     Hello World client in Python
#     Connects REQ socket to tcp://localhost:5555
#     Sends "Hello" to server, expects "World" back
#

import zmq
import nengo
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


def goalPosFilter(t, x):
    return np.array([x[4], x[5]])


def goalVelFilter(t, x):
    return np.array([x[6], x[7]])


def velMultFunc(x):
    return [x[0] * (x[0] * x[0] + x[1] * x[1])**.25,
            x[1] * (x[0] * x[0] + x[1] * x[1])**.25]


print("-----NENCLIENT------")

client = Client()

# Set up neurons
model = nengo.Network()

with model:

    # Neuron ensembles
    vel_error = nengo.Ensemble(1000, dimensions=2, radius=400)
    pos_error = nengo.Ensemble(1000, dimensions=2, radius=100)

    vel_mult = nengo.Ensemble(1000, dimensions=2, radius=400)
    f_controller = nengo.Ensemble(1000, dimensions=2, radius=400)

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

    nengo.Connection(pos_in, pos_error)
    nengo.Connection(goalPos_in, pos_error, transform=[[-1, 0], [0, -1]])
    nengo.Connection(vel_in, vel_error)
    nengo.Connection(goalVel_in, vel_error, transform=[[-1, 0], [0, -1]])
    nengo.Connection(vel_error, vel_mult, function=velMultFunc)

    nengo.Connection(pos_error, f_controller)
    nengo.Connection(vel_mult, f_controller)
    nengo.Connection(f_controller, socket_out)

if __name__ == "__main__":
    #    Do 10 requests, waiting each time for a response
    with nengo.Simulator(model) as sim:
        while (1):
            sim.step()
