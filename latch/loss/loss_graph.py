from typing import Any, Dict, List, Tuple

import jax
from jax import numpy as jnp

from .gates import Gate


class GateEdge:
    def __init__(self, source: str, target: str, gate: Gate):
        self.source = source
        self.target = target
        self.gate = gate


class LossNode:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value


class LossGateGraph:
    def __init__(self, edges: List[GateEdge] = []):
        self.edges = edges

    def _get_incoming_edges(self, node: str) -> List[GateEdge]:
        incoming = []
        for edge in self.edges:
            if edge.target == node:
                incoming.append(edge)
        return incoming

    def compute_node(
        self,
        node: str,
        inputs: Dict[str, jax.Array],
        thunk: Dict[str, jax.Array] = {},
    ) -> Any:
        # Check if the node has already been computed
        if node not in thunk:
            # Check if the node is a target of an edge
            incoming_edges = self._get_incoming_edges(node)
            # Compute the node's parents, then the node
            # Start with a gate value of 1.0 for root nodes
            incoming_gate_vals: List[jax.Array] = [jnp.array(1.0)]

            for incoming_edge in incoming_edges:
                # Compute the parent node
                parent_gate_val = self.compute_node(incoming_edge.source, inputs, thunk)
                # Compute the gate value
                parent_val = inputs[incoming_edge.source]
                gate_val = incoming_edge.gate.compute(parent_val)
                incoming_gate_vals.append(gate_val)
                incoming_gate_vals.append(parent_gate_val)

            # Compute the final gate value
            incoming_gate_val_array = jnp.array(incoming_gate_vals)
            gate_val = jnp.min(incoming_gate_val_array)  # type: ignore
            # Add the node to the thunk
            thunk[node] = gate_val

        # Return the node value and the gate value
        return thunk[node]

    def forward(self, inputs: Dict[str, jax.Array]):
        """Computes the gate values for the graph.

        Args:
            inputs (Dict[str, float]): A dictionary of input values for the graph.

        Returns:
            Dict[str, jax.Array]: A dictionary of gate values for the graph.
        """

        thunk = {}

        for input_node in inputs.keys():
            self.compute_node(input_node, inputs, thunk)

        return thunk
