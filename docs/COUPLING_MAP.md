# Carte des couplages — comment les dépôts peuvent se parler 🔗

Ce document rassemble les liens possibles entre des idées déjà présentes dans les dépôts. **Un lien proposé n’est pas une fusion déjà implémentée ni une découverte validée.** Le but est précisément de ne pas jeter des dimensions parce qu’elles ne rentrent pas dans le premier petit banc de test.

## 1. Le réseau de domaines 🕸️

```text
                         ┌───────────────────────────┐
                         │  Rythme / événements      │
                         │  SIMULTANEITE, FOCAL      │
                         └─────┬───────────────┬─────┘
                               │               │
                    synchronie│               │drive/ordonnancement
                               ▼               ▼
  ┌──────────────────┐   ┌───────────────┐   ┌────────────────────┐
  │ Graphe / routes  │◄─►│ État / champ  │◄─►│ Géométrie / tissage│
  │ ONE, X, VEGA     │   │ Travaux, QVM  │   │ FOCAL, KTN synth.  │
  └──────┬───────────┘   └─────┬─────────┘   └──────────┬─────────┘
         │                      │                        │
         └─────────────┬────────┴────────────────────────┘
                       ▼
             ┌───────────────────────┐
             │ Trajectoire / mémoire │
             │ hystérésis, overlays  │
             └───────────┬───────────┘
                         ▼
             ┌───────────────────────┐
             │ Observables multiples │
             │ phase · graphe · P_sig│
             │ route · coût · tâche  │
             └───────────┬───────────┘
                         ▼
             ┌───────────────────────┐
             │ Contrôle / bus        │───► nouveau drive ou route
             │ Omni, QVM, engines    │
             └───────────────────────┘
```

## 2. Liens concrets à étudier

| Couplage | Sources à relier | Ce qui circule | Question à poser |
|---|---|---|---|
| Phase ↔ champ géométrique | FOCAL EXP-68 / EXP-22 | `R(t)`, densité locale, déplacement d’ancres, `P_sig` | Une rétroaction entre phase et géométrie produit-elle un régime absent quand on coupe l’un des sens ? |
| Drive ↔ transition ↔ topologie | `Travaux/experiment_driven.py`, `ssh_model.py`, `topology.py` | drive `δ(t)`, trajectoire, matrice de corrélation, signaux de bord, persistance | Le calendrier d’entrée modifie-t-il le résultat ou seulement sa vitesse ? |
| Aller/retour ↔ mémoire | `Travaux/experiment_hysteresis.py`, `ktn_phase4_hysteresis.py` | histoire, branche, aire d’hystérésis, régénération | Deux histoires vers la même commande finale laissent-elles des états computationnels distincts ? |
| Structure ↔ sélection de chemin | ONE, X, VEGA, mesures topologiques | graphe, scores de route, masque actif, sortie des experts | Une lecture topologique peut-elle orienter un calcul clairsemé mieux qu’un score dense ou uniforme ? |
| Chemins multiples ↔ interférence / agrégation | `RATISS-PHOTON`, VEGA/X | amplitudes ou scores de chemin, phase, routes retenues | Quand faut-il additionner, moyenner, sélectionner ou annuler des chemins ? |
| Observables ↔ contrôle fermé | `RATISS-Omni`, QVM, FUSION/NAVIER | cohérence, énergie/état, drive et accusé de réception | Une mesure interne peut-elle guider l’étape suivante sans produire une boucle instable ? |
| Rythme multi-échelle ↔ mémoire | SIMULTANEITE, FOCAL, `ratiss-neuro` | phase rapide, enveloppe lente, état de mémoire | Un signal lent organise-t-il les interactions rapides, et l’inverse ? |

## 3. Le rôle de chaque domaine dans le futur processeur

- **Cadence** : peut être une horloge, un train d’événements, un rythme interne ou une règle de synchronisation. Le modèle de phase n’en est qu’un exemple.
- **Tissu relationnel** : décrit quels éléments interagissent; sa structure peut changer, être routée, ou rester partiellement stable.
- **Champ / état** : transporte l’information transformée; peut être réel, complexe, discret ou hybride selon les sources et la démonstration.
- **Mémoire de parcours** : conserve une dépendance à l’histoire par un état, une boucle ou une modification locale des liens.
- **Coupleur** : traduit un signal entre domaines sans imposer une seule représentation à tous.
- **Ordonnanceur** : déclenche les opérations. Il doit pouvoir être synchronisé, événementiel ou défini par la règle rythmique de Jonathan.
- **Observateur** : collecte un vecteur de mesures, jamais automatiquement un verdict.
- **Contrôleur** : utilise un sous-ensemble de mesures pour modifier le drive ou le routage; doit être testé par ablation.

## 4. Logique d’intégration 🛠️

Le dépôt contient maintenant un **algorithme central**, distinct du simple câblage de domaines :

- `DynamicMatrixProcessor` : état continu `x`, matrice dirigée évolutive `W`, rythme `φ`, repli pondéré top-k en `W_eff`, intégration temporelle, événements de seuil et readout calibrable;
- `DendriticPopulation` : branches dendritiques indépendantes, courants E/I séparés, sommation somatique, adaptation, seuil/reset et transfert de branche linéaire ou saturant;
- `DynamicMatrixProcessor` : poids E/I par branche et deux plasticités sélectionnables — co-activité ou STDP pair-à-pair par traces; ces règles restent des hypothèses de calcul, pas des modèles biologiques validés;
- `LogicGateLayer` : portes booléennes qui consomment les franchissements de seuil et contrôlent routes, plasticité ou readout au tick suivant;
- `weighted_fold` : transforme `W` en routes actives sparse en fonction des poids, de l’activité et de la compatibilité rythmique;
- `InterlacedSystem`, `Snapshot` et `Scheduler` : infrastructure générique secondaire pour d’autres expériences et ordonnanceurs;
- `topk_route` / `sparse_mix` : primitives séparées de routage;
- adaptateurs phase↔géométrie FOCAL et domaines SSH / `P_sig` adaptés de `Travaux`, avec provenance/licence documentées;
- observables distinctes de phase, graphe, persistance et hystérésis.

La première règle de rythme et de plasticité codée est une **hypothèse proposée pour rendre le principe simulable**. Elle n’est ni attribuée à Jonathan ni déclarée définitive. Elle pourra être modifiée à partir de la réflexion, des résultats de contrôles ou d’une règle ultérieurement spécifiée.
