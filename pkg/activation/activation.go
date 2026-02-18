package activation

import "fmt"

// ActivationFunc is the forward pass
type ActivationFunc func(float64) float64

// ActivationDeriv is the backward pass
// 'y' represents the value after activation (the neuron's .Value)
type ActivationDeriv func(y float64) float64

type ActivationPair struct {
	Forward  ActivationFunc
	Backward ActivationDeriv
}

type ActivationType string

const (
	Sigmoid  ActivationType = "sigmoid"
	ReLU     ActivationType = "relu"
	Identity ActivationType = "identity"
)

var ActivationRegistry = map[ActivationType]ActivationPair{
	Sigmoid: {
		Forward:  Sigmoid_Func,
		Backward: Sigmoid_Derivative,
	},
	ReLU: {
		Forward:  ReLU_Func,
		Backward: ReLU_Derivative,
	},
	Identity: {
		Forward:  func(x float64) float64 { return x },
		Backward: func(y float64) float64 { return 1 },
	},
}

func (a ActivationType) Get() ActivationPair {
	pair, ok := ActivationRegistry[a]
	if !ok {
		panic(fmt.Sprintf("activation function %s not found in registry", a))
	}

	return pair
}
