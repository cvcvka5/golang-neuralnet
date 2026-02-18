package loss

import "fmt"

func MSE_Func(prediction []float64, expected []float64) (float64, error) {
	if len(prediction) != len(expected) {
		return 0, fmt.Errorf("shape mismatch: pred %d, exp %d", len(prediction), len(expected))
	}

	var sum float64
	for i := range prediction {
		diff := prediction[i] - expected[i]
		sum += diff * diff
	}
	return sum / float64(len(prediction)), nil
}

func MSE_Derivative(prediction, expected float64, n int) float64 {
	return (2.0 / float64(n)) * (prediction - expected)
}
