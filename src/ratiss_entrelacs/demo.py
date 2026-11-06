"""Small smoke tasks for branch, E/I, timing-plasticity and logic mechanisms."""
from __future__ import annotations

import numpy as np

from .cells import DendriticConfig, DendriticPopulation
from .logic import LogicGate, LogicGateBindings, LogicGateLayer
from .matrix_processor import DynamicMatrixProcessor, MatrixProcessorConfig, STDPConfig


def _logic_control_demo() -> None:
    config = MatrixProcessorConfig(
        units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
        matrix_learning_rate=1.0, weight_decay=0.0, threshold=0.1, seed=5,
    )
    circuit = LogicGateLayer(2, [LogicGate("both_crossed", "AND", (0, 1))])
    processor = DynamicMatrixProcessor(
        1,
        config=config,
        input_weights=np.full((2, 1), 5.0),
        initial_weights=np.array([[0.0, 0.5], [0.5, 0.0]]),
        logic_layer=circuit,
        gate_bindings=LogicGateBindings(route="both_crossed", plasticity="both_crossed"),
    )
    first = processor.step([1.0])
    second = processor.step([1.0])
    print("\nPlacement des portes : seuils → événements → AND → contrôle du tick suivant")
    print(f"tick 0 : nouveaux événements={first.events.tolist()} | AND appliqué (événements précédents)={first.logic_outputs['both_crossed']} | routes={first.active_edges}")
    print(f"tick 1 : AND des événements du tick 0={second.logic_outputs['both_crossed']} | routes actives={second.active_edges}")


def _stdp_timing_demo() -> None:
    """Contrast causal and anti-causal event order on one toy connection."""
    config = MatrixProcessorConfig(
        units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
        matrix_learning_rate=1.0, weight_decay=0.0, threshold=0.5,
        plasticity_rule="stdp", row_budget=1.0, seed=9,
    )
    initial_e = np.array([[0.0, 0.0], [0.2, 0.0]])

    def run_order(order: tuple[int, int]) -> float:
        processor = DynamicMatrixProcessor(
            2,
            config=config,
            input_weights=[[10.0, 0.0], [0.0, 10.0]],
            initial_excitatory_weights=initial_e,
            stdp_config=STDPConfig(
                tau_pre=0.5, tau_post=0.5,
                a_plus=0.2, a_minus=0.2,
            ),
        )
        before = float(processor.excitatory_weights[1, 0])
        for unit in order:
            signal = np.zeros(2)
            signal[unit] = 1.0
            processor.step(signal)
        return float(processor.excitatory_weights[1, 0]) - before

    causal = run_order((0, 1))
    anti_causal = run_order((1, 0))
    print("\nSTDP jouet — variation de W_E[récepteur 1, source 0] :")
    print(f"pré→post : {causal:+.4f} | post→pré : {anti_causal:+.4f}")


def main() -> None:
    print("RATISS-ENTRELACS — noyau inspiré du calcul cérébral (hypothèse v0.4)")
    print("Branches dendritiques, courants E/I, intégration somatique, adaptation et événements.")
    print("Les expériences ci-dessous sont des smoke tests, pas des preuves d'avantage.\n")

    config = MatrixProcessorConfig(
        units=16,
        top_k=4,
        dt=0.1,
        tau=0.5,
        rhythm_strength=0.5,
        matrix_learning_rate=0.02,
        weight_decay=0.01,
        seed=17,
    )
    rng = np.random.default_rng(17)
    branch_inputs = np.zeros((config.units, 2, 3), dtype=np.float64)
    branch_inputs[:, 0, 0] = rng.normal(0.0, 1.0 / np.sqrt(2.0), config.units)
    branch_inputs[:, 0, 2] = rng.normal(0.0, 1.0 / np.sqrt(2.0), config.units)
    branch_inputs[:, 1, 1] = rng.normal(0.0, 1.0 / np.sqrt(2.0), config.units)
    branch_inputs[:, 1, 2] = rng.normal(0.0, 1.0 / np.sqrt(2.0), config.units)
    initial_e = rng.uniform(0.0, 0.015, (config.units, 2, config.units))
    initial_i = rng.uniform(0.0, 0.008, (config.units, 2, config.units))
    dendrites = DendriticPopulation(
        config.units,
        DendriticConfig(
            tau_dendrite=0.2, tau_membrane=0.5, tau_adaptation=4.0,
            dendrite_gain=1.2, adaptation_gain=0.1,
            adaptation_jump=0.08, threshold=0.55, reset_value=0.0,
            branch_nonlinearity="tanh",
        ),
        branches=2,
        branch_gains=[1.0, 0.8],
    )
    processor = DynamicMatrixProcessor(
        input_dim=3,
        output_dim=2,
        config=config,
        dendritic_layer=dendrites,
        input_branch_weights=branch_inputs,
        initial_excitatory_weights=initial_e,
        initial_inhibitory_weights=initial_i,
    )
    history_a = np.array([1.0, 0.0, 0.0])
    history_b = np.array([0.0, 1.0, 0.0])
    common_final = np.array([0.0, 0.0, 1.0])
    patterns = (
        (np.stack([history_a] * 4 + [common_final] * 4), np.array([1.0, 0.0])),
        (np.stack([history_b] * 4 + [common_final] * 4), np.array([0.0, 1.0])),
    )
    initial_weights = processor.weights

    for _ in range(30):
        for sequence, target in patterns:
            processor.reset()
            for signal in sequence:
                step = processor.step(signal, learn_matrix=True)
            processor.calibrate_readout(target, activity=step.activity, learning_rate=0.3)

    # Calibrate the readout again on the frozen trained dynamics so its
    # training states match the subsequent inference condition.
    for _ in range(30):
        for sequence, target in patterns:
            processor.reset()
            for signal in sequence:
                step = processor.step(signal, learn_matrix=False)
            processor.calibrate_readout(target, activity=step.activity, learning_rate=0.3)

    learned_weights = processor.weights
    predictions = []
    final_states = []
    active_edges = []
    spike_counts = []
    for sequence, _ in patterns:
        processor.reset()
        spike_count = 0
        for signal in sequence:
            step = processor.step(signal, learn_matrix=False)
            spike_count += int(np.count_nonzero(step.events))
        predictions.append(step.output)
        final_states.append(step.activity)
        active_edges.append(step.active_edges)
        spike_counts.append(spike_count)

    prediction_array = np.stack(predictions)
    targets = np.stack([target for _, target in patterns])
    rmse = float(np.sqrt(np.mean((prediction_array - targets) ** 2)))
    memory_separation = float(np.linalg.norm(final_states[0] - final_states[1]))
    branch_potentials = step.compartments["dendritic_potential"]
    branch_norms = np.linalg.norm(branch_potentials, axis=0)
    print("signal final identique :", common_final.tolist())
    print(f"sorties après calibration : {np.round(prediction_array, 3).tolist()}")
    print(f"RMSE sur ces deux histoires : {rmse:.4f} | distance d'état final : {memory_separation:.3f}")
    print(f"événements de seuil par histoire : {spike_counts} | norme d'adaptation finale : {np.linalg.norm(step.compartments['adaptation']):.3f}")
    print(f"normes des deux branches dendritiques : {np.round(branch_norms, 3).tolist()}")
    print(f"courants E/I du dernier pas : {np.linalg.norm(step.compartments['excitatory_current']):.3f} / {np.linalg.norm(step.compartments['inhibitory_current']):.3f}")
    print(f"liens actifs branche×voie par pas : {active_edges} (top-k par branche={config.units * config.top_k}; E/I peuvent partager un lien)")
    print(f"variation de la matrice agrégée ||W_final − W_initial||₂ : {np.linalg.norm(learned_weights - initial_weights):.4f}")
    print("Le tri inspecte encore les matrices denses : aucun gain de vitesse ou d'énergie établi.")
    print("\nContrôles de recherche : branche unique/multiple, E/I gelées/plastiques, co-activité/STDP, rythme et portes.")
    _stdp_timing_demo()
    _logic_control_demo()


if __name__ == "__main__":
    main()
