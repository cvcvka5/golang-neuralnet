package dataset

import "math/rand"

type ColorData struct {
	Input, Output []float64
	Name          string
}

func Get() []ColorData {
	return []ColorData{
		// --- WARM COLORS (Target: 1.0) ---
		{[]float64{1.0, 0.0, 0.0}, []float64{1.0}, "Pure Red"},
		{[]float64{1.0, 0.5, 0.0}, []float64{1.0}, "Orange"},
		{[]float64{1.0, 1.0, 0.0}, []float64{1.0}, "Yellow"},
		{[]float64{0.6, 0.0, 0.0}, []float64{1.0}, "Dark Red"},
		{[]float64{1.0, 0.8, 0.6}, []float64{1.0}, "Peach"},
		{[]float64{0.9, 0.2, 0.4}, []float64{1.0}, "Pink/Crimson"},
		{[]float64{0.5, 0.2, 0.0}, []float64{1.0}, "Brown"},
		{[]float64{1.0, 0.4, 0.4}, []float64{1.0}, "Salmon"},

		// --- COOL COLORS (Target: 0.0) ---
		{[]float64{0.0, 0.0, 1.0}, []float64{0.0}, "Pure Blue"},
		{[]float64{0.0, 1.0, 0.0}, []float64{0.0}, "Pure Green"},
		{[]float64{0.0, 1.0, 1.0}, []float64{0.0}, "Cyan"},
		{[]float64{0.2, 0.4, 0.8}, []float64{0.0}, "Sky Blue"},
		{[]float64{0.0, 0.3, 0.1}, []float64{0.0}, "Forest Green"},
		{[]float64{0.5, 0.0, 0.5}, []float64{0.0}, "Indigo/Purple"},
		{[]float64{0.1, 0.1, 0.4}, []float64{0.0}, "Navy"},
		{[]float64{0.7, 0.9, 1.0}, []float64{0.0}, "Light Blue"},

		// --- NEUTRAL/EDGE CASES ---
		{[]float64{0.2, 0.2, 0.2}, []float64{0.0}, "Dark Grey (Cool-ish)"},
		{[]float64{0.8, 0.8, 0.8}, []float64{1.0}, "Light Grey (Warm-ish)"},
		{[]float64{0.0, 0.0, 0.0}, []float64{0.0}, "Black"},
		{[]float64{1.0, 1.0, 1.0}, []float64{1.0}, "White"},
	}
}

func Shuffle(data []ColorData) {
	for i := len(data) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
}
