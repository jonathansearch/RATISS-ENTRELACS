# RATISS-ENTRELACS 🕸️🌌⚙️

**Processeur algorithmique in silico à matrice pondérée dynamique**  
RATISS Labs · Yaoundé 🇨🇲 · dépôt local initialisé le 6 novembre 2026

> **Hypothèse inspirée du calcul cérébral, à prototyper :** intégration analogique de type dendrite/soma, matrice pondérée évolutive, rythmes, adaptation et événements de seuil; un repli dynamique sélectionne les interactions actives, puis des portes logiques contrôlent les opérations suivantes. On vise des principes de calcul inspirés du cerveau, pas une copie complète de son anatomie, de ses émotions ou de sa conscience — et pas un LLM. 🧠⚡🕸️

## Pourquoi « ENTRELACS » ? 🧶

Le nom rappelle que plusieurs mécanismes — matrice, rythme, état continu, repli des relations, mémoire de parcours et lecture — peuvent entrer dans le calcul. Mais le livrable central n’est pas un simple câblage entre domaines : c’est une **règle algorithmique explicite** qui transforme l’état et la matrice à chaque pas.

## Le programme de recherche 🚀

Le but est de formaliser une logique de calcul propre, simulable avant tout matériel : matrice de relations variable, intégration continue, repli pondéré, modulation rythmique et calibration. Les dépôts servent de sources de mécanismes et de contrôles; ils ne remplacent pas l’algorithme.

## Algorithme central — hypothèse logicielle v0.4 🧮

Le prototype `DynamicMatrixProcessor` applique à chaque pas cette boucle :

```text
entrée u(t) ─► branches dendritiques E/I ─► soma + adaptation ─► état x, événements e
                    ▲                             │                         │
                    │                             ▼                         ▼
          repli top-k rythmé ◄── voies W_E/W_I ◄── co-activité ou STDP ◄── timing e(t)
                    │
                    └──────────────────── readout calibrable ────────────► sortie y(t)
```

1. Une **matrice dirigée et pondérée** `W(t)` conserve les relations; elle n’est pas assimilée à une matrice de poids figée.
2. Le **repli pondéré** utilise phases, activité et poids pour sélectionner top-k relations et former `W_eff(t)`, sans supprimer les autres liens de `W(t)`.
3. Les entrées traversent des **branches dendritiques distinctes**; leurs courants excitateurs et inhibiteurs restent séparés avant leur intégration somatique continue.
4. Les seuils somatiques produisent des événements; adaptation, reset et rythme modulent la dynamique.
5. Une couche de **portes logiques** (`AND`, `OR`, `XOR`, `NOT`…) peut contrôler routes, plasticité ou readout au tick suivant.
6. Deux règles candidates sont disponibles : co-activité locale et plasticité temporelle par paires (STDP), avec budget combiné des poids E/I.
7. Un readout optionnel est calibré à partir de cibles.

Le sinus rythmique, le repli top-k, la forme des branches, les coefficients E/I et la règle STDP par traces sont **des hypothèses de travail à tester et remplacer**. Aucun de ces choix n’est attribué à Jonathan comme règle définitive, ni présenté comme une loi cérébrale fidèle. Détails et équations : [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

Les mécanismes de `Travaux`, FOCAL, NEURO, VEGA, ONE, X, Omni et PHOTON sont des pistes de réutilisation ou de comparaison : champ conduit, `P_sig`, hystérésis, analogique/événementiel, routage, chemins multiples, feedback et calibration. La cartographie les relie sans confondre ces composants avec le nouvel algorithme. ✨

## Les domaines, côte à côte 🧭

| Domaine | Représentation possible | Héritage à examiner |
|---|---|---|
| Cadence et phases | horloges, phases, impulsions ou événements | `RATISS-SIMULTANEITE`, expériences de phase de `ratiss-focal` |
| Champ / état évolutif | amplitudes, corrélations, énergie ou état simulé | SSH conduit dans `Travaux`, simulateurs de `RATISS-QVM` et `ratiss-continuums` |
| Graphe et topologie | liens actifs, cycles, signatures à différentes échelles | `RATISS-ONE`, `RATISS-X`, `Travaux`, outils topologiques |
| Routage | sélection et mélange de chemins parcimonieux | `RATISS-X`, routeur MoE de `RATISS-VEGA` |
| Mémoire de trajectoire | hystérésis, traces, overlays ou états lents | `Travaux`, `ratiss-focal`, mémoire séparée de `RATISS-ONE` |
| Contrôle / bus | échange d’observables et action en boucle fermée | `RATISS-Omni`, interfaces des moteurs QVM/NAVIER/FUSION |

## Ce qui est déjà construit dans le dépôt 🛠️

- `matrix_processor.py` : **algorithme central inspiré du calcul cérébral** — repli rythmique top-k de voies E/I conservées séparément, poids par branche, intégration continue et choix de plasticité co-activité/STDP.
- `cells.py` : branches dendritiques indépendantes, sommation somatique, courants E/I, adaptation, seuil/reset; activation locale linéaire ou `tanh` paramétrable.
- `logic.py` : couche booléenne optionnelle; les événements de seuil pilotent routes, mise à jour de matrice et readout au tick suivant.
- `routing.py` : primitive top-k séparée, issue des familles X/VEGA; le repli du processeur est défini dans son propre algorithme.
- `system.py`, `engine.py` : infrastructures génériques de simulation, utiles pour les expériences mais pas l’algorithme central.
- `domains/ssh_reference.py`, `topology.py` : adaptations de `Travaux` pour champ SSH conduit et observable `P_sig`, avec attribution/licence.
- `couplings.py`, `metrics.py`, `baselines/phase_ring.py` : adaptateurs de phase↔géométrie, observables séparées et contrôle de phase optionnel.
- `docs/ALGORITHM.md`, `docs/EXPERIMENT_MATRIX.md` et `docs/COUPLING_MAP.md` : formules du prototype, ablations et cartographie des idées sources.

Les observables ne sont pas fusionnées en un « score magique ». L’évaluation gardera un **vecteur de mesures** (activité, routes, événements, matrice, erreur de tâche et coût). La tâche externe et ses critères restent à définir avant toute affirmation de performance. 📊

## Exécution 💻

```bash
python -m pip install -e .
python -m ratiss_entrelacs              # algorithme matriciel + diagnostics des mécanismes
python -m unittest discover -s tests -v # règles du processeur + contrôles logiciels
```

La démo met au premier plan le noyau matrice + population dendrite→soma; les tests ciblés couvrent aussi les branches distinctes, les courants E/I, la trace des compartiments et la potentiation/dépression selon l’ordre des événements. Les portes `AND` restent une couche de contrôle au tick suivant. **Aucune supériorité sur une tâche n’est établie** : les contrôles comparatifs restent à mener.

## Rigueur sans réduction ⚖️

On garde l’ambition large sans transformer l’hypothèse en fait établi. Les contrôles portent d’abord sur l’algorithme : rythme nul/actif, matrice gelée/adaptative, repli sparse/dense, histoire conservée/effacée et readout calibré/non calibré, à budget égal. Les échecs restent dans le journal. 🔬🧯

Ce dépôt est **local**, n’importe aucun poids de LLM et ne simule pas un appareil physique. Les deux adaptations de `Travaux` sont attribuées et fournies sous Apache-2.0; la licence générale du code original reste à choisir avant publication. Aucun commit ni push de RATISS-ENTRELACS n’a été fait. 🔒🚫⬆️

## Étape de recherche suivante 🔬

Les trois extensions candidates sont maintenant **implémentées et testées** : branches séparées, voies E/I distinctes et STDP par paires. La suite n’est pas d’empiler des mécanismes, mais de mener les ablations appariées : branche unique/multiple, courant signé/E-I, co-activité/STDP, E/I gelées ou plastiques, portes actives/bypassées, puis tâche tenue à l’écart et budget de calcul égal. La démo demeure un smoke test, pas un résultat de généralisation. 💥
