package main

import (
	"fmt"

	dataset "github.com/cvcvka5/golang-neuralnet/dataset"
	nnet "github.com/cvcvka5/golang-neuralnet/pkg"
	"github.com/cvcvka5/golang-neuralnet/pkg/activation"
)

func main() {
	// 1. Setup: 3 inputs (R,G,B), 1 output (Warm/Cool), 0.1 Learning Rate
	// We used a slightly deeper 8x8 hidden layer architecture
	nn, err := nnet.New(3, 1, 0.1, 8, 8)
	if err != nil {
		panic(err)
	}

	// 2. Set activations: Layer 0 is Identity, Hidden/Output are Sigmoid
	nn.SetActivationFunc(activation.Sigmoid, 1, 2, 3)

	// 3. Load the expanded dataset
	dataList := dataset.Get()

	fmt.Println("Starting Training...")

	// 10,000 epochs to make sure it really "sees" the decision boundary
	for epoch := 0; epoch < 10000; epoch++ {
		var totalLoss float64

		// Shuffle every epoch so the network doesn't just memorize the order
		dataset.Shuffle(dataList)

		for _, data := range dataList {
			// Step A: How wrong are we? (Evaluate)
			loss, _ := nn.Evaluate(data.Input, data.Output)

			// Step B: Fix the weights (Backward)
			nn.Backward(data.Output)

			totalLoss += loss
		}

		// Print progress every 1000 epochs to keep an eye on things
		if epoch%1000 == 0 {
			avgLoss := totalLoss / float64(len(dataList))
			fmt.Printf("Epoch %d - Avg Loss: %.6f\n", epoch, avgLoss)
		}
	}

	fmt.Println("\n--- Training Complete ---")

	// 4. Test on a "Mystery" Color (Olive-ish Green)
	test := []float64{0.1, 0.7, 0.3}

	// We use the clean Predict API for the user
	res, _ := nn.Predict(test)

	fmt.Printf("Testing Color [R:0.1, G:0.7, B:0.3]\n")
	fmt.Printf("Prediction: %.2f%% Warm\n", res[0]*100)

	if res[0] < 0.5 {
		fmt.Println("Result: I think that's a COOL color.")
	} else {
		fmt.Println("Result: I think that's a WARM color.")
	}

	// 5. Save the brain so we don't have to do this again!
	saveErr := nn.Save("color_brain.json")
	if saveErr != nil {
		fmt.Printf("Warning: Could not save brain: %v\n", saveErr)
	} else {
		fmt.Println("Brain saved to color_brain.json successfully.")
	}
}
