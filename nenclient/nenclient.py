#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import nengo
import numpy as np
import ctypes

class Client:
  def __init__(self):
    self.context = zmq.Context()
    print("Connecting to server...")
    self.socket = self.context.socket(zmq.REQ)
    self.socket.connect("tcp://localhost:5555")

  def put(self, t, x):
    self.socket.send_string("t=" + str(t) + ", x=" + str(x))

  def get(self, t):
    message = self.socket.recv()
    print("Received reply [ %s ]" % (message))
    return np.array([20, 0])

  print("-----NENCLIENT------")



client = Client()
client.put(0, "start")

# Set up neurons
model = nengo.Network()
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # and represents a 2-dimensional signal
    neurons = nengo.Ensemble(500, dimensions=2, radius=200)

    inNode = nengo.Node(client.get)
    outNode = nengo.Node(client.put, size_in = 2)
    # The indices in neurons define which dimension the input will project to
    nengo.Connection(inNode, neurons)
    nengo.Connection(neurons, outNode)

    neurons_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.01)

if __name__ == "__main__":

  #  Do 10 requests, waiting each time for a response
  with nengo.Simulator(model) as sim:
    while (1):
        sim.step()
