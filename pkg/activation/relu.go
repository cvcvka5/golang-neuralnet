package activation

func ReLU_Func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func ReLU_Derivative(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}
