# Sources transdisciplinaires utilisées 🗺️

Le dépôt n’embarque **aucun clone complet** des projets cités. Deux modules de `Travaux` sont adaptés avec attribution et licence Apache-2.0 (`domains/ssh_reference.py`, `topology.py`); les autres opérateurs sont des interfaces ou réimplémentations indépendantes. Les détails figurent dans `THIRD_PARTY_NOTICES.md`. « Adapté de » ne veut pas dire « reproduit intégralement » ni « validé par ». 

## Rythme ↔ structure ↔ topologie 🥁🕸️

### [`RATISS-SIMULTANEITE`](https://github.com/jonathansearch/RATISS-SIMULTANEITE)

- `simmat_v3_opt.py` : phases de Kuramoto, fréquences propres, drive commun, bruit local et couplage en anneau.
- L’observable `Rd` mesure le verrouillage collectif; le README distingue ce résultat d’un effet sur la parité globale, indique un scan exploratoire et rappelle qu’il s’agit de calcul classique (pas d’intrication ou de speedup quantique).
- Dans ENTRELACS, l’anneau forcé est une référence de domaine, pas l’architecture totale.

### [`ratiss-focal`](https://github.com/jonathansearch/ratiss-focal)

- `experiences/exp22_kuramoto_fort.py` mesure ensemble synchronie et `P_sig` pendant une focalisation.
- `experiences/exp52_phase_attracteur.py` compare la phase initiale et l’état final pour classifier le comportement comme déterministe, stochastique ou monostable.
- `experiences/exp68_neurons_topologiques.py` couple phases, positions et densités locales; l’exploration demande si le partage sensoriel/organisation géométrique modifie la synchronie.
- `couplings.py` rend exécutables deux adaptateurs tirés de ces relations : phase→ancrages radiaux et densité locale→biais de phase. Ils restent des opérateurs remplaçables, pas une loi complète du processeur.
- Ce sont des expériences du dépôt avec leurs protocoles et limites : elles inspirent des boucles de couplage plus riches que le simple anneau, sans transformer leur vocabulaire neuronal en modèle obligatoire.

### [`ratiss-neuro`](https://github.com/jonathansearch/ratiss-neuro) — fork de référence

- `ratiss_neuro/kuramoto.py` présente une horloge lente, une enveloppe rapide et un terme de cohérence/entropie.
- `ratiss_neuro/snn.py` sert de repère conceptuel analogique→événement : état de membrane continu, seuil, décharge et adaptation. `cells.py` propose maintenant plusieurs branches abstraites, canaux E/I, sommation dendrite→soma et adaptation; il n’importe pas le code AdEx ni toute une dynamique biologique. L’intégrateur fuyant reste disponible comme contrôle.
- La règle STDP à paires/traces implémentée est une hypothèse logicielle minimale à tester, avec coefficients indépendants et budget borné; elle n’est pas présentée comme une reproduction d’une règle biologique particulière ni comme une découverte.
- Cette piste rappelle que rythmes et sorties événementielles peuvent être comparés; elle n’est ni nécessaire au processeur ni une preuve de conscience.

## Champ, trajectoire, topologie et tissage 🌊

### [`Travaux`](https://github.com/jonathansearch/Travaux) — Apache-2.0

- `ratiss_photoinduced/ssh_model.py`, `experiment_driven.py` : chaîne SSH jouet, drive temporel et propagation RK4.
- `experiment_hysteresis.py` et `ktn_phase4_hysteresis.py` : parcours aller-retour et mesure d’hystérésis.
- `topology.py` : persistance Vietoris–Rips H0/H1 sur profils relationnels dérivés des corrélations; `P_sig` n’est pas un score universel.
- `ktn_woven.py` : textures synthétiques de brins tissés, perturbation optique modélisée et régénération par seed; ce ne sont pas les données d’un cristal KTN:Li réel.
- Le dépôt documente aussi des limites de modèle (SSH jouet, tailles finies, dephasing simplifié, textures KTN synthétiques) et des essais QPU séparés des simulations.

## Routage, association, mémoire et multiplicité 🔀

### [`RATISS-X`](https://github.com/jonathansearch/RATISS-X) et [`RATISS-VEGA`](https://github.com/jonathansearch/RATISS-VEGA)

- X : routes GPS, chemins d’activations, tranches MLP et moteur NumPy lié à un modèle/artefacts.
- VEGA : le fichier local `model/modeling_ratiss_vega.py` contient `VegaRouter` (logits de gate → softmax/température), dispatch soft ou top-1, supervision optionnelle des routes et équilibrage auxiliaire; le dépôt est MIT.
- Dans `matrix_processor.py`, `weighted_fold` transpose seulement le motif de sélection pondérée/température/top-k au niveau des relations de la matrice, en y ajoutant phase et activité. C’est une réimplémentation adaptée, pas le routeur du Transformer ni son code copié.
- `topk_route` / `sparse_mix` restent des primitives autonomes séparées.

### [`RATISS-ONE`](https://github.com/jonathansearch/RATISS-ONE) et [`Ratiss-GI`](https://github.com/jonathansearch/Ratiss-GI)

- ONE distingue un tissu relationnel et une mécanique séquentielle; documente la diffusion locale normalisée par degré, l’épisode et les overlays d’arcs.
- Motif de conception possible : mémoire/association distincte du mécanisme de lecture ou de sortie. Ce n’est pas une obligation de copier la métaphore neuronale.
- Licence README : CC BY-NC-SA 4.0; aucun code n’est importé.

### [`RATISS-PHOTON`](https://github.com/jonathansearch/RATISS-PHOTON)

- Simulations de chemins multiples, phase, courants et bain thermique; le dépôt publie aussi une limite/falsification sur la redistribution entre branches dans son modèle.
- Idée à comparer à la sélection top-k : agrégation cohérente, annulation, sélection ou mélange ne sont pas forcément la même opération.

## Contrôle, moteurs et validation 🔁⚖️

- [`RATISS-Omni`](https://github.com/jonathansearch/RATISS-Omni) : bus d’état partagé et contrôle en boucle fermée; motif système pour le couplage observables→action.
- [`RATISS-QVM`](https://github.com/jonathansearch/RATISS-QVM), [`ratiss-continuums`](https://github.com/jonathansearch/ratiss-continuums) : moteurs de simulation, canaux, échelles et observables dans des univers numériques distincts.
- [`RATISS-Framework`](https://github.com/jonathansearch/RATISS-Framework), [`RATISS-ETALONS`](https://github.com/jonathansearch/RATISS-ETALONS), [`defi1-topological-lie-detector`](https://github.com/jonathansearch/defi1-topological-lie-detector) : provenance, contrôles instrumentaux, critères préalables et exemple de refutation honnête.
- [`RATISS-VERSE-I`](https://github.com/jonathansearch/RATISS-VERSE-I) : solver partagé, distinction observables dégénérées, limites et matrice de falsifiabilité.

## Sous-systèmes non retenus comme cœur unique

Les autres familles — photonique, QPU, Navier–Stokes, fusion, cosmologie, biologie, génomique, agents, cryptographie, outils de laboratoire, éducation et infrastructures — restent dans la cartographie complète de `PORTFOLIO_MAP.md` et `GITHUB_INVENTORY_2026-11-06.md`. Elles ne sont pas déclarées inutiles : elles ne sont simplement pas toutes des composants de calcul compatibles sans médiation. Les interfaces de domaine permettront d’ajouter ou retirer un couplage sans prétendre que toutes les analogies sont équivalentes.
