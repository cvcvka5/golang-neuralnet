package internal

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"neuralnet/pkg/activation"
	"neuralnet/pkg/loss"
	"os"
)

type Network struct {
	Layers       [][]*Neuron
	LossType     loss.Type
	LearningRate float64
	currentLoss  *float64 // Used to guard against Backward() calls without Evaluate()
}

// New builds the network structure and wires everything up
func New(inputSize int, outputSize int, learningRate float64, hiddenLayerSizes ...int) (*Network, error) {
	nn := &Network{
		LearningRate: learningRate,
		LossType:     loss.MSE,
	}

	// Create the layer blueprint
	sizes := append([]int{inputSize}, hiddenLayerSizes...)
	sizes = append(sizes, outputSize)

	for i, size := range sizes {
		layer := make([]*Neuron, size)
		for j := 0; j < size; j++ {
			// Small random bias to start
			b := rand.Float64()*0.2 - 0.1
			n := &Neuron{Bias: &b}

			// Layer 0 is just a pass-through; others get Sigmoid by default
			if i == 0 {
				n.ActivationType = activation.Identity
			} else {
				n.ActivationType = activation.Sigmoid
			}
			layer[j] = n
		}
		nn.Layers = append(nn.Layers, layer)
	}

	// Connect the dots (Fully Connected)
	if err := nn.connectNeurons(); err != nil {
		return nil, err
	}

	return nn, nil
}

func (nn *Network) connectNeurons() error {
	if len(nn.Layers) < 2 {
		return errors.New("cannot connect a network with less than 2 layers")
	}

	for i := 0; i < len(nn.Layers)-1; i++ {
		for _, n1 := range nn.Layers[i] {
			for _, n2 := range nn.Layers[i+1] {
				w := rand.Float64()*2 - 1 // Weight range [-1, 1]

				// Both neurons share a pointer to the weight for easy updates
				n1.Next = append(n1.Next, &Connection{Other: n2, Weight: &w})
				n2.Previous = append(n2.Previous, &Connection{Other: n1, Weight: &w})
			}
		}
	}
	return nil
}

func (nn *Network) SetActivationFunc(t activation.ActivationType, layerIndexes ...int) {
	for _, idx := range layerIndexes {
		for _, n := range nn.Layers[idx] {
			n.ActivationType = t
		}
	}
}

// ZeroGrad wipes neuron values so they don't accumulate across passes
func (nn *Network) ZeroGrad() {
	for i := 1; i < len(nn.Layers); i++ {
		for _, n := range nn.Layers[i] {
			n.Value = 0
		}
	}
}

// forward is the internal math engine (Predict and Evaluate use this)
func (nn *Network) forward(input []float64) ([]float64, error) {
	nn.ZeroGrad()

	if len(nn.Layers[0]) != len(input) {
		return nil, fmt.Errorf("input mismatch: got %d, want %d", len(input), len(nn.Layers[0]))
	}

	// 1. Fill Input Layer
	for i, n := range nn.Layers[0] {
		n.Value = input[i]
		n.Propagate()
	}

	// 2. Compute Hidden & Output Layers
	for i := 1; i < len(nn.Layers); i++ {
		isOutput := i == len(nn.Layers)-1
		results := make([]float64, len(nn.Layers[i]))

		for j, n := range nn.Layers[i] {
			n.Value += *n.Bias
			n.Activate()
			if !isOutput {
				n.Propagate()
			}
			results[j] = n.Value
		}

		if isOutput {
			return results, nil
		}
	}
	return nil, nil
}

// Predict is the clean public API for users
func (nn *Network) Predict(input []float64) ([]float64, error) {
	return nn.forward(input)
}

// Evaluate calculates loss based on a target
func (nn *Network) Evaluate(input, target []float64) (float64, error) {
	prediction, err := nn.forward(input)
	if err != nil {
		return 0, err
	}

	l, err := nn.LossType.Get().Forward(prediction, target)
	if err != nil {
		return 0, err
	}

	nn.currentLoss = &l
	return l, nil
}

// Backward runs backpropagation (The Chain Rule)

func (nn *Network) Backward(target []float64) error {
	if nn.currentLoss == nil {
		return errors.New("must call Evaluate() before Backward()")
	}

	outputLayer := nn.Layers[len(nn.Layers)-1]
	lossPair := nn.LossType.Get()

	// 1. Compute Output Deltas
	for i, n := range outputLayer {
		lossGrad := lossPair.Backward(n.Value, target[i], len(outputLayer))
		actGrad := n.ActivationType.Get().Backward(n.Value)
		n.Delta = lossGrad * actGrad
	}

	// 2. Distribute Blame Backwards
	for i := len(nn.Layers) - 2; i > 0; i-- {
		for _, n := range nn.Layers[i] {
			var errorSignal float64
			for _, conn := range n.Next {
				errorSignal += conn.Other.Delta * (*conn.Weight)
			}
			n.Delta = errorSignal * n.ActivationType.Get().Backward(n.Value)
		}
	}

	// 3. Apply Weight & Bias Updates
	for i := 1; i < len(nn.Layers); i++ {
		for _, n := range nn.Layers[i] {
			*n.Bias -= nn.LearningRate * n.Delta
			for _, conn := range n.Previous {
				*conn.Weight -= nn.LearningRate * n.Delta * conn.Other.Value
			}
		}
	}

	nn.currentLoss = nil
	return nil
}

func (nn *Network) Save(filename string) error {
	data, err := json.MarshalIndent(nn, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (nn *Network) Load(filename string) error {
	fileData, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	return json.Unmarshal(fileData, nn)
}
