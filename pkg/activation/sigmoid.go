package activation

import "math"

// Sigmoid_Func implements the classic S-curve: 1 / (1 + e^-z)
func Sigmoid_Func(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Sigmoid_Derivative calculates the gradient.
// Note: This version takes the ACTIVATED value (y), not the raw sum (z).
// Formula: f'(z) = f(z) * (1 - f(z))
func Sigmoid_Derivative(y float64) float64 {
	return y * (1.0 - y)
}
