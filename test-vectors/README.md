# `pdaqp.[ch]` for inter-operability testing

So far, the hardware generator module utilizes the serialized algorithm files
`pdaqp.[ch]` as the input. Three use cases are presented:

- PID controller
- MPC controller
- Power plant controller

... to primarily test the parser logic.

Delete this folder when the module accepts the `cvxgen.PDAQPInterface` as the
first-class intermediate representation.