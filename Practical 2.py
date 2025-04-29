def mcCulloch_pitts_andnot(x1, x2):
    # Weights for x1 and x2
    w1 = 1
    w2 = -1
    threshold = 1

    # Net input to the neuron
    net_input = (x1 * w1) + (x2 * w2)

    # Activation function (step function)
    output = 1 if net_input >= threshold else 0

    return output

# Test all input combinations
print("x1 x2 | ANDNOT(x1,x2)")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = mcCulloch_pitts_andnot(x1, x2)
        print(f" {x1}  {x2} |     {result}")
