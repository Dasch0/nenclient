#
#     Hello World client in Python
#     Connects REQ socket to tcp://localhost:5555
#     Sends "Hello" to server, expects "World" back
#

import nengo
import networks


print("-----NENCLIENT------")

model = networks.vInhibitController()

if __name__ == "__main__":
    #    Do 10 requests, waiting each time for a response
    with nengo.Simulator(model) as sim:
        while (1):
            sim.step()
