# Matrice d’expériences — programme ENTRELACS 🧪

But : tester d’abord l’algorithme de matrice dynamique à repli pondéré, puis ses relations avec les mécanismes des dépôts. Les valeurs, tâches et critères d’une comparaison doivent être fixés avant de regarder les résultats. Les contrôles ci-dessous sont des propositions de protocole, pas des résultats.

| ID | Question algorithmique / transdisciplinaire | Construction / intervention | Contrôles et ablations | Mesures à garder séparées |
|---|---|---|---|---|
| A0 — boucle MDRP | Les équations sont-elles stables et exécutées comme défini ? | branches, voies E/I, repli top-k, intégration continue, plasticité et readout | mêmes entrées/graines; vérifier dt≤τ, budget L1 E+I cumulé par récepteur, événements | activité, poids par voie, densité active, sorties, erreurs numériques |
| A1 — valeur du repli | Le repli pondéré apporte-t-il quelque chose à une tâche fixée ? | comparer `k<n` à matrice dense `k=n` | même nombre d’unités, même budget, ordre des sources permuté | qualité de tâche, coût, routes retenues |
| A2 — valeur de l’adaptation | L’évolution de `W` est-elle utile au lieu de simplement variable ? | co-activité vs STDP vs matrices gelées (`η=0`) | mêmes états, événements, entrées et graines; poids E/I initiaux identiques | qualité, ΔW_E/ΔW_I, dérive, stabilité, coût |
| A3 — rôle du rythme | La porte rythmique et la compatibilité changent-elles le calcul ? | rythme interne/externe proposé | `ρ=0`, rythme permuté, rythme constant; même budget | état, événement, routes, sensibilité aux phases |
| A4 — couche logique | Où et quand les portes agissent-elles utilement ? | événements de seuil → portes booléennes → contrôle des routes, plasticité ou readout au tick suivant | gates bypassées; portes permutées; délai retiré/ajouté | latence, routes activées, poids modifiés, erreur de tâche |
| A5 — branches dendritiques | Des branches distinctes et leur transfert local changent-ils l’intégration somatique ? | une branche vs plusieurs; routage identique vs entrées/récurrence par branche; linéaire vs `tanh` | mêmes courants totaux; permutation des branches; gains égaux/inégaux | potentiels par branche, membrane, événements, adaptation, erreur de tâche |
| A6 — voies E/I | Des matrices E/I séparées se comportent-elles différemment du courant signé agrégé ? | poids non négatifs séparés, éventuellement coexistants sur une paire; repli E/I conjoint | E/I combinées; voie I nulle; signe et ordre des sources permutés | courants E/I, poids par voie, budget total, routes et erreur |
| A7 — plasticité événementielle | L’ordre et l’intervalle des décharges modifient-ils les poids comme le prévoit la règle candidate ? | co-activité vs STDP à traces; pré→post et post→pré | coïncidence simultanée; traces effacées/permutées; A+/A− nuls; plasticité inhibitrice gelée | ΔW_E, ΔW_I, traces, stabilité, tâche tenue à l’écart |
| E0 — instruments | Les observables rendent-elles les valeurs connues ? | phases égales/étalées; graphe arbre/cycle; branches égales/différentes; route top-k avec scores imposés | valeurs analytiques connues; permutation de l’ordre | R, β₁, aire, indices et poids de route |
| E1 — rythme × géométrie | La rétroaction phase↔structure change-t-elle l’évolution ? | petite version du couplage FOCAL EXP-68; comparer phase→géométrie seulement, géométrie→phase seulement, boucle complète | graphe/positions fixes; couplage coupé; seed apparié | R(t), variation de structure, P_sig si l’instrument est étalonné |
| E2 — drive × transition | Le même état final dépend-il du calendrier du drive ? | rampe constante vs plusieurs vitesses, modèle SSH de `Travaux` | statique; drive sans franchir le seuil; trajectoire adiabatique de référence | gap, bord, corrélations, P_sig, erreur numérique |
| E3 — mémoire de trajet | Deux histoires vers la même commande finale laissent-elles une trace exploitable ? | rampe aller-retour; suite d’événements alternative; perturbation puis récupération | pas d’aller-retour; histoire inversée; état réinitialisé | aire d’hystérésis, écart d’état final, performance sur tâche |
| E4 — topologie × routage | Une signature de structure améliore-t-elle la sélection des chemins ? | utiliser un observable relationnel pour fournir les scores de route | scores aléatoires, uniformes, top-k dense, top-k sans signature | routes choisies, coût actif, sortie sur tâche tenue à l’écart |
| E5 — chemins multiples | Les chemins doivent-ils être additionnés, pondérés ou sélectionnés ? | comparer règles de combinaison inspirées des simulations multi-chemins | chemin unique, mélange sans phase, phase randomisée, branche bloquée | sortie, interférence, sensibilité au bruit, coût |
| E6 — feedback | La mesure interne peut-elle choisir le drive suivant de manière stable ? | bus d’état + loi de commande injectable, motif Omni | drive ouvert; contrôleur sans observable; capteur permuté; limites de commande | erreur de tâche, oscillations de contrôle, coût, stabilité |
| E7 — mémoire associative | Une représentation graphe + mise à jour locale aide-t-elle une tâche sans séquence de tokens ? | petit graphe de relations avec overlay local; séparation stockage/routage | graphe dense; mémoire gelée; update global; contrôle avec permutation | récupération exacte, modifications collatérales, coût |
| E8 — bruit croisé | Quels domaines protègent ou amplifient les autres sous perturbation ? | bruit local, bruit partagé, bruit corrélé entre domaine rythme et domaine structure | bruit à zéro; bruit seulement sur une interface; amplitude appariée | chaque observable, performance tâche, variabilité entre graines |

## Règles pour ne pas biaiser le programme ⚖️

1. Garder le **même budget de calcul** quand on compare sparse/dense ou une règle de routage.
2. Garder l’**entrée, les graines et le protocole appariés** pour les ablations.
3. Ne pas optimiser un seuil sur les mêmes cas qui servent à annoncer le succès.
4. Conserver les mesures vectorielles; publier les valeurs qui n’appuient pas l’hypothèse.
5. Ajouter une tâche externe claire : une belle transition ou une signature topologique n’est pas automatiquement un calcul utile.
6. Faire évoluer la matrice hypothétique à partir des contrôles et des résultats, en gardant un historique des règles essayées plutôt qu’en les effaçant.
