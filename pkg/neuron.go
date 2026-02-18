package internal

import "github.com/cvcvka5/golang-neuralnet/pkg/activation"

type Connection struct {
	Other  *Neuron
	Weight *float64
}

type Neuron struct {
	// State
	Value float64 `json:"value"` // y (activated)
	Z     float64 `json:"z"`     // raw sum
	Delta float64 `json:"delta"` // gradient

	// Parameters
	Bias     *float64      `json:"bias"`
	Next     []*Connection `json:"-"`
	Previous []*Connection `json:"-"`

	// Configuration
	ActivationType activation.ActivationType `json:"activation_type"`
}

func (n *Neuron) Propagate() {
	for _, conn := range n.Next {
		conn.Other.Value += n.Value * (*conn.Weight)
	}
}

// Activate applies the chosen activation function to the current value
func (n *Neuron) Activate() {
	pair := n.ActivationType.Get()
	if pair.Forward != nil {
		n.Z = n.Value
		n.Value = pair.Forward(n.Z)
	}
}
