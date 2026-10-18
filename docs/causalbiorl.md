# CausalBioRL — reinforcement learning

CausalBioRL is the RL half of NeoRx. It supplies Gymnasium environments
that simulate biological systems, agents that use a structural causal
model as a world model, and an integration environment where the agent
drives the real discovery pipeline rather than an ODE toy.

The motivating distinction: a model-free agent learns *what correlates
with reward in the trajectories it has seen*. An agent with an SCM as its
world model can ask what would happen under an intervention it has not
tried — which is the same question the [causal engine](identification.md)
asks of the knowledge graph, one level down.

## The environments

Four environments register with Gymnasium on import:

```python
>>> import gymnasium as gym
>>> import neorx.causalbiorl   # registers on import
>>> [env_id for env_id in ("GeneticToggle-v0", "MetabolicPathway-v0",
...                        "CellGrowth-v0", "DrugDiscovery-v0")
...  if env_id in gym.registry]
['GeneticToggle-v0', 'MetabolicPathway-v0', 'CellGrowth-v0',
 'DrugDiscovery-v0']

```

| Environment | Simulates | Obs | Action | Max steps |
|---|---|---|---|---|
| `GeneticToggle-v0` | Two mutually repressing genes (Gardner, Cantor & Collins, 2000) | 2 | 2 inducers | 200 |
| `MetabolicPathway-v0` | A five-enzyme pathway under Michaelis–Menten kinetics | 11 | 5 expression levels | 500 |
| `CellGrowth-v0` | Microbial population under logistic growth, nutrient limit, toxin pulses | 5 | feed rate, waste removal | 500 |
| `DrugDiscovery-v0` | The real NeoRx pipeline | composite | hierarchical | 100 |

The first three are real published models with real ODEs, not reward
shaping dressed as biology. The toggle switch is bistable: depending on
initial conditions it settles to gene-A-high or gene-B-high, and the
agent's inducers weaken one gene's repression of the other.

```python
>>> from neorx.causalbiorl.envs import GeneticToggleSwitchEnv
>>> env = GeneticToggleSwitchEnv()
>>> obs, info = env.reset(seed=0)
>>> obs.shape, env.action_space.shape
((2,), (2,))
>>> sorted(info)
['distance_to_target', 'state', 'step']
>>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
>>> isinstance(reward, float), terminated, truncated
(True, False, False)

```

Every environment's `info` carries the interpretable state alongside the
observation — enzyme levels, population size, distance to target — so a
trajectory can be read rather than only scored.

## Structural causal models

An SCM parameterises each edge `Xᵢ → Xⱼ` of a causal graph with either a
linear function (fast, interpretable) or a small neural network
(flexible).

```python
>>> import networkx as nx
>>> from neorx.causalbiorl.causal import StructuralCausalModel
>>> graph = nx.DiGraph([("s0", "s1"), ("s1", "s2")])
>>> scm = StructuralCausalModel(graph, state_dim=3, action_dim=1,
...                             mechanism="linear", epochs=5)
>>> sorted(scm.graph.nodes)
['s0', 's1', 's2']

```

Fit it on observed transitions with `scm.fit(states, actions,
next_states)`, forward-simulate with `scm.predict(state, action)`, and
intervene with `scm.do(intervention, state, action)` — Pearl's
do-operator, setting a variable to a fixed value and propagating through
the graph rather than conditioning on it.

`get_edge_strengths` exposes the learned per-edge magnitudes, and
`get_edge_provenance` says where each edge came from —
`augment_graph` can add edges beyond the ones discovered, and keeping
the two distinguishable matters if the graph is going to be reported.

`from_disease_graph` builds an SCM from a NeoRx `DiseaseGraph`, which is
the bridge between the two halves of the project.
`neorx.causalbiorl.causal.CausalDiscovery` learns a graph from data when
you do not have one.

## The drug-discovery environment

`DrugDiscovery-v0` is where this stops being a simulation. One episode is
one disease campaign with a budget of generate-and-screen cycles.

```
R-GCN encodes the disease graph    → 128-D state embedding
UCB1 selects a target              (level 1, discrete)
CEM navigates GenMol latent space  (level 2, continuous 128-D Δz)
SurrogateDockingModel              → binding score in ~1 ms
MolScreen                          → QED / SA / filters
MirrorFold                         → stability
AdaptiveRewardLearner              → multi-objective reward
```

The action is hierarchical: *which target to invest in*, then *where to
move in latent space*. The state concatenates the graph embedding, the
current best molecule's features, and a per-target summary.

```python
>>> from neorx.causalbiorl.causal import DiseaseGraphEncoder
>>> encoder = DiseaseGraphEncoder()
>>> encoder.embedding_dim
128

```

The encoder is a relational GCN over 13 relation types — the twelve
[edge types](graph.md#the-data-model) plus a self-relation — with basis
decomposition so the relation-specific weights do not blow up the
parameter count.

### The docking surrogate

Real Vina docking is seconds to minutes per ligand. An RL agent needs
thousands of evaluations per episode, so the environment uses a learned
surrogate over Morgan fingerprints:

```python
>>> from neorx.causalbiorl.causal import smiles_to_fingerprint
>>> smiles_to_fingerprint("CC(=O)Oc1ccccc1C(O)=O").shape
(2048,)

```

!!! warning "A surrogate score is not a docking result"
    `SurrogateDockingModel` returns an estimate from a 2048-bit
    fingerprint and a 128-D target vector. It is fast enough to train an
    agent against and it is **not** a binding affinity. Anything the RL
    mode proposes should be re-docked with [DockBot](dockbot.md) before
    the number is reported.

### The reward

```python
>>> from neorx.causalbiorl.causal import OBJECTIVE_NAMES
>>> OBJECTIVE_NAMES
['binding', 'qed', 'sa', 'novelty', 'causal', 'stability']

```

`AdaptiveRewardLearner` weights these six adaptively over an episode
rather than fixing a scalarisation up front. Note that this is a
*different* weighting from the [composite scorer's](scoring.md) six
dimensions — the overlap in names is not an overlap in numbers.

## Agents

```python
>>> from neorx.causalbiorl.agents import (CausalAgent, PPOAgent,
...                                       SACAgent, RandomAgent)
>>> CausalAgent.__name__
'CausalAgent'

```

`CausalAgent` does hierarchical planning against an SCM world model.
`PPOAgent` and `SACAgent` wrap stable-baselines3 as model-free
baselines, and `RandomAgent` is the floor any claim of learning has to
clear. `CausalPlanner` and `HierarchicalPlanner` are the planning
components on their own.

## Running it

```bash
neorx causalbiorl train --env GeneticToggle-v0
neorx causalbiorl benchmark
neorx causalbiorl play --env CellGrowth-v0
neorx causalbiorl visualise
```

From Python, the pipeline's RL mode:

```python
from neorx.core import run_rl_pipeline

result = run_rl_pipeline("HIV", n_episodes=20, top_n_targets=5)
```

!!! warning "Requires network and real time"
    `run_rl_pipeline` builds a live disease graph first, then runs the
    episodes. It is not a doctest for both reasons.

## Benchmark numbers

The `causalbiorl_bench` experiment in the repository's `experiments/`
package measures agents against environments. Its numbers come from run
records rather than from this page — see
[Experiments and run records](experiments.md) for how to run it and where
the records land. The reason that indirection exists is in the
[provenance history](experiments.md#why-the-runner-owns-persistence): an
earlier version of this benchmark lost 881 seconds of measurement to a
timeout, and the lost cell was retyped by hand into a figure script.
