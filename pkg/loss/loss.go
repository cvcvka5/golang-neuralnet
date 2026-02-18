package loss

import (
	"fmt"
)

// Define specific types for the functions to ensure type safety
type LossFunc func(prediction, expected []float64) (float64, error)
type LossDeriv func(prediction, expected float64, n int) float64

// LossPair groups the function with its derivative
type LossPair struct {
	Forward  LossFunc
	Backward LossDeriv
}

type Type string

const MSE Type = "mse"

// Use a map of structs instead of interface slices
var LossRegistry = map[Type]LossPair{
	MSE: {
		Forward:  MSE_Func,
		Backward: MSE_Derivative,
	},
}

// Helper methods to access the registry
func (l Type) Get() LossPair {
	pair, ok := LossRegistry[l]
	if !ok {
		panic(fmt.Sprintf("loss function %s not found in registry", l))
	}
	return pair
}
