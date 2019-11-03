#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import nengo
import numpy as np

class Controller:
  
  def __init__(self):
    self.vector = np.array([0,0])

  def update(self):
    self.vector = np.array([0,0])

  def get(self, t):
    return self.vector


if __name__ == "__main__":
  print("-----NENCLIENT------")

  controller = Controller()

  context = zmq.Context()

  #  Socket to talk to server
  print("Connecting to hello world server…")
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://localhost:5555")

  # Set up neurosns
  model = nengo.Network(label='2D Representation')
  with model:
      # Our ensemble consists of 100 leaky integrate-and-fire neurons,
      # and represents a 2-dimensional signal
      neurons = nengo.Ensemble(100, dimensions=1, radius=200)

      # Create input nodes representing the sine and cosine
      ctrlNode = nengo.Node(output = 0.1)

      # The indices in neurons define which dimension the input will project to
      nengo.Connection(ctrlNode, neurons)

      nengo.Connection(neurons, neurons)

      neurons_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.01)

  #  Do 10 requests, waiting each time for a response
  with nengo.Simulator(model) as sim:
    while (1):

        controller.update()
        print("Sending request …")
      
        sim.step()
        out = sim.data[neurons_probe]
        if len(out) > 1:
          data = out[-1]
        else:
          data = out[0]

        socket.send_string(str(data))
        #  Get the reply.
        message = socket.recv()
        print("Received reply [ %s ]" % (message))
