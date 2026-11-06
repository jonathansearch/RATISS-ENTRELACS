# Hypothèse d’architecture — ENTRELACS 🕸️⚙️

**Hypothèse de travail formalisée par l’assistant à partir des textes et codes RATISS.** La version logicielle v0.4 propose des équations exécutables ancrées dans des principes cérébraux (branches dendritiques, intégration somatique, événements, adaptation et plasticité temporelle); elle ne prétend ni être une loi établie ni être la définition finale de l’algorithme de Jonathan.

## 1. Le fil commun que je vois 🌌

Les projets se situent dans des domaines différents — calcul clairsemé, rythmes et phases, graphes, topologie, champs physiques simulés, mémoire, contrôle en boucle fermée. Ce qui peut les relier n’est pas une analogie superficielle (« tout est un cerveau » ou « tout est un oscillateur »), mais une structure logicielle commune :

1. un domaine porte un **état**;
2. un autre domaine envoie une **influence** par un couplage déclaré;
3. un **rythme ou événement** décide quand et dans quel ordre l’influence agit;
4. la trajectoire résultante peut modifier l’organisation des liens et laisser une **trace**;
5. plusieurs observables interrogent le résultat;
6. une boucle de contrôle peut utiliser ces mesures pour agir sur l’étape suivante.

C’est l’hypothèse d’un **processeur à domaines entrelacés** : le calcul se trouve dans les interactions entre représentations, et non nécessairement dans un seul tenseur dense ou un seul substrat biologique. 🧩

## 2. Boucles de couplage repérées dans tes dépôts 🔁

### A. Phase ↔ géométrie ↔ topologie — `ratiss-focal`

- `experiences/exp68_neurons_topologiques.py` fait évoluer des phases sur un graphe et un ensemble de positions; une densité locale intervient dans l’évolution des phases. La géométrie et la synchronie ne sont donc pas simplement deux sorties indépendantes dans cette exploration.
- `experiences/exp22_kuramoto_fort.py` suit synchronie et `P_sig` ensemble pendant une focalisation.
- `experiences/exp52_phase_attracteur.py` compare conditions initiales et états finaux pour distinguer attracteur, déterminisme, stochastique et monostabilité.

**Interprétation :** le rythme peut être à la fois une dynamique et une variable de couplage vers l’organisation spatiale/graphique; inversement, une structure peut contraindre le rythme. Cette boucle mérite une expérience explicite avec ablations, plutôt que d’être retirée au profit d’une seule métrique de phase.

### B. Drive → trajectoire → phase topologique → mémoire de parcours — `Travaux`

La chaîne SSH pilotée dans le temps relie un drive externe à l’évolution d’orbitales, aux corrélations, à des signaux de bord et à `P_sig`. La rampe aller-retour mesure une hystérésis. Le projet KTN:Li ajoute des textures tissées synthétiques, des perturbations locales et une régénération du motif.

**Interprétation :** une architecture de calcul peut conserver une trace de la manière dont elle a atteint son état final; la même entrée finale peut éventuellement produire une sortie différente selon le parcours. Cette mémoire de trajectoire est une candidate différente d’une mémoire de tokens ou d’un poids fixe.

### C. Graphe associatif ↔ activation sélective ↔ sortie — `RATISS-ONE`, `RATISS-X`, `RATISS-VEGA`

- ONE sépare le tissu relationnel et l’organe séquentiel; il décrit diffusion locale avec normalisation par degré et overlays dynamiques.
- X explore des chemins d’activation clairsemés, des tranches de MLP et un routage GPS.
- VEGA a un routeur appris entre experts d’un Transformer.

**Interprétation :** séparation des fonctions, routage conditionnel et mise à jour localisée peuvent servir de motifs de conception, sans réutiliser le modèle linguistique comme fondation et sans importer l’architecture neuro-symbolique entière.

### D. État observé ↔ action ↔ nouvel état — `RATISS-Omni`

Omni assemble des moteurs au moyen d’un bus d’état et ferme une boucle de contrôle. **Interprétation :** la future architecture peut être interactive : des mesures internes modifient le drive et le prochain chemin. Les observables ne sont donc pas nécessairement un rapport final; elles peuvent être des signaux actifs de contrôle.

### E. Interférence de chemins et environnement — `RATISS-PHOTON`

Le simulateur expose plusieurs chemins, phases, courants et un bain thermique, avec des résultats et des limites documentés dans son univers numérique. **Interprétation :** concurrence, combinaison ou suppression de chemins constitue une autre famille de mécanismes à comparer au simple routeur top-k; l’analogie ne prouve rien sur le photon réel.

## 3. Le substrat logiciel visé 🧱

La représentation de l’état devrait pouvoir être hétérogène :

```text
S(t) = { champs continus, phases/événements, graphe de relations,
         mémoire de trajectoire, état des routes, variables de contrôle }
```

Une étape générale peut être écrite sans prétendre connaître encore la loi :

```text
S(t + Δt) = F( S(t), entrées(t), couplages(t), ordonnanceur(t), bruit(t) )
```

`F` ne sera pas assimilée à un simple câblage. Le nouveau `DynamicMatrixProcessor` propose une première loi de calcul : état continu, matrice `W(t)` qui évolue, repli pondéré en `W_eff(t)`, rythme et readout calibrable. Le `InterlacedSystem` générique reste disponible pour des extensions. Le sinus et la règle locale de co-activité sont des hypothèses provisoires, pas une démonstration déjà fournie ni la définition finale du rythme.

## 4. Lectures multiples — pas un score unique 📊

| Famille d’observable | Exemple déjà représenté dans le dépôt | Question distincte |
|---|---|---|
| Ordre de phase | `phase_coherence` | Les unités se verrouillent-elles ensemble ? |
| Structure de graphe | `graph_cycle_rank` | Des cycles relationnels apparaissent-ils ? |
| Persistance topologique | `P_sig` dans `Travaux` et d’autres dépôts | Quels motifs persistent quand l’échelle varie ? |
| Mémoire de parcours | `hysteresis_area` | Aller et retour donnent-ils des états différents ? |
| Routage | `topk_route` / `sparse_mix` | Quels chemins ont réellement été activés ? |
| Résultat externe | tâche à définir | Est-ce que le système résout le problème posé ? |
| Coût | temps, mémoire, opérations | Quel coût pour cette sortie ? |

Ces signaux doivent d’abord rester séparés. Une combinaison en un score ne viendra qu’après qu’on ait montré, avec les contrôles, que chaque terme contribue à une tâche et qu’aucun ne double compte un autre.

## 5. Ce que la v0.4 formalise et ajoute 🛠️

- Le projet porte le nom **ENTRELACS**; la cadence n’est pas le nom ni l’unique mécanisme.
- Le livrable algorithmique est explicite : `DynamicMatrixProcessor` maintient un état continu, une matrice pondérée évolutive et un rythme; il replie la matrice en routes top-k, adapte les relations localement et calibre un readout.
- `DendriticPopulation` maintient maintenant plusieurs branches indépendantes par unité, un transfert local configurable (linéaire ou saturant), la sommation au soma, l’adaptation, le seuil et le reset.
- Les voies excitatrices/inhibitrices ont des poids non négatifs conservés séparément, peuvent coexister sur une même paire, sont repliées ensemble et restent bornées par un budget combiné.
- La plasticité choisit entre co-activité locale et STDP pair-à-pair à traces exponentielles; l’ordre causal/anti-causal et les facteurs E/I sont testables.
- `weighted_fold_channels` rend le repli variable par branche, activité et compatibilité rythmique; le tri inspecte encore des matrices denses, sans gain de coût établi.
- `LogicGateLayer` combine les événements de seuil en commandes pour routes, plasticité ou readout au pas suivant, sans remplacer le calcul continu.
- Deux domaines logiciels issus de `Travaux` — SSH conduit et `P_sig` — sont adaptés sous licence et attribués.
- Les adaptateurs FOCAL phase→géométrie et densité→biais de phase, les observables, le routeur et le cadre multi-domaines restent des mécanismes réutilisables ou comparatifs, pas la définition du processeur.
- Les équations et les contrôles sont décrits dans `docs/ALGORITHM.md`; les dépôts et idées associées restent cartographiés sans être déclarés inutiles.

## 6. Comment cette hypothèse peut être mise à l’épreuve 🔬

1. **Branches :** comparer branche unique/multiple, routages identiques/distincts et transfert local linéaire/saturant.
2. **Voies E/I :** comparer l’agrégat signé, les canaux séparés, leur gel et leur plasticité; vérifier le budget combiné.
3. **Timing :** comparer co-activité et STDP; permuter l’ordre pre/post, les traces, le délai et les coefficients excitateur/inhibiteur.
4. **Cellule et portes :** intégrerateur simple contre dendrite/soma/adaptation; portes bypassées contre commandes au tick suivant.
5. **Matrice et mémoire :** `W` adaptative/gelée, repli top-k/dense, rythme actif/inactif, et deux histoires vers la même entrée finale.
6. **Tâche :** choisir une opération calculable et une baseline simple; mesurer sortie/coût, pas seulement la beauté des traces.
7. **Critique :** contrôles de suppression, permutations, déphasage, bruit, ordre de mise à jour et seed; garder les axes transdisciplinaires comme expériences distinctes.

L’ambition reste large; la preuve devra être locale et cumulative. Le premier test ne doit pas décider à lui seul que les autres domaines sont inutiles. ⚖️
