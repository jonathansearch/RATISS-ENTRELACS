# Balayage du GitHub RATISS 🗺️🇨🇲

**Consultation du 6 novembre 2026.** L’API publique GitHub a retourné **81 dépôts**, forks compris; les 81 README ont été balayés pour cartographier le portefeuille. Le code a ensuite été approfondi sur les domaines les plus liés à cette architecture : `Travaux`, `RATISS-SIMULTANEITE`, `ratiss-focal`, `RATISS-Omni`, VEGA/X, ONE, les moteurs topologiques et leurs protocoles. Cela ne prétend pas être un audit ligne par ligne des centaines de milliers de lignes des 81 dépôts et forks.

L’inventaire local précédent contenait 75 noms. Sept noms étaient absents de cette copie (`RATISS-ACADEMY`, `RATISS-DICTIONNAIRE-TOPOLOGIQUE`, `RATISS-ENGLISH`, `RATISS-VEGA`, `RATISS-VERSE-I`, `RATISS-X`, `defi1-topological-lie-detector`); `DISCORD-RATISS` ne ressortait plus dans la liste actuelle. La liste nominative complète et datée est conservée dans [GITHUB_INVENTORY_2026-11-06.md](GITHUB_INVENTORY_2026-11-06.md). Les dépôts n’ont pas été clonés, afin de préserver le plafond du workspace. 📦

## Ce qui se relie — sans jeter les domaines 🔗

| Domaine | Dépôts et code regardés | Logique qui peut s’entrelacer | Limite à garder visible |
|---|---|---|---|
| Rythmes / phases | `RATISS-SIMULTANEITE`; `ratiss-focal/exp22`, `exp52`, `exp68`; fork `ratiss-neuro/kuramoto.py` | drive partagé, synchronie, rythme multi-échelle, phase couplée à une densité/structure | une synchronie n’est pas automatiquement un calcul utile; certaines expériences restent exploratoires |
| Champs / transitions / trajectoires | `Travaux/ratiss_photoinduced/ssh_model.py`, `experiment_driven.py`, `experiment_hysteresis.py`; QVM, Continuums | drive temporel modifiant un état, transitions et dépendance au chemin | SSH est un modèle jouet; simulation et résultat matériel doivent rester distincts |
| Tissage / topologie / forme | `Travaux/ktn_woven.py`, `topology.py`; FOCAL, Continuums, moteurs topologiques, dictionnaire | géométrie relationnelle, cycles, `P_sig`, persistence, invariants comme observables | le KTN de ce code est synthétique; aucun invariant unique ne certifie la qualité d’un calcul |
| Graphes / mémoire / routes | ONE, GI, X, VEGA, VRN, LeWM | diffusion associative, update local, mémoire distincte de sortie, sparse routing | les dépôts ont des objectifs et licences différents; les architectures ne sont pas fusionnées telles quelles |
| Chemins multiples | `RATISS-PHOTON`, X/VEGA | sélection, mélange, interférence ou annulation de chemins | les résultats de photon sont dans un monde simulé, pas une mesure du photon réel |
| Bus / feedback / multi-moteurs | `RATISS-Omni`, QVM, NAVIER, FUSION | observables qui reviennent vers un contrôleur et modifient le drive suivant | chaque loi de contrôle a ses hypothèses; le bus doit rester modulaire |
| Méthode / épreuves | Framework, ETALONS, Archives, GitHub-Audit, `defi1-topological-lie-detector` | contrôles connus, ablations, provenance, résultats négatifs conservés | tests verts et hash valident l’exécution annoncée, pas la théorie entière |
| Simulations de domaines | VERSE-I, PHOTON, PLANCK, NAVIER, NUCLEAIRE, FUSION, GCR, synchrotron-24, dose12, etc. | solveurs numériques, champs, pas de temps, contrôles de régime | une analogie transdisciplinaire est un pont à tester, pas une identité de mécanisme |

## Les boucles de rétroaction importantes 🧩

1. **Phase → géométrie → densité → phase** : les explorations FOCAL `exp68` mettent en relation oscillateurs, positions et densité locale. Ce lien peut être central pour ENTRELACS; le modèle ne doit donc pas s’arrêter à mesurer `R(t)`.
2. **Drive → trajectoire → bord/topologie → hystérésis** : `Travaux` relie un protocole temporel à la trajectoire d’état puis à des observables; la montée et la descente peuvent différer.
3. **Topologie/graphes → routage → calcul → nouvelle structure** : ONE, X, VEGA et les outils topologiques fournissent des pistes qui peuvent agir à plusieurs endroits du cycle, pas seulement être une sonde après calcul.
4. **Mesure → contrôleur → nouveau drive** : le motif Omni ouvre une boucle fermée; les mesures deviennent des signaux d’action plutôt qu’un tableau final.
5. **Chemins → combinaison → interférence/choix** : `RATISS-PHOTON` ouvre un vocabulaire différent du top-k; les chemins ne sont pas toujours simplement sélectionnés ou additionnés.
6. **Mémoire → histoire → seuil/routage** : hystérésis, overlays et états lents peuvent fournir une mémoire computationnelle distincte d’un historique de tokens.

La cartographie détaillée est dans [COUPLING_MAP.md](COUPLING_MAP.md); les sources sont analysées dans [SOURCES.md](SOURCES.md), et une suite d’expériences transversales proposée dans [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

## Autres familles du portefeuille prises en compte 🌍

- **Langage, agents et apprentissage :** `RATISS-FIRST`, `Ratiss-GI`, `Ratiss-Fusion-stark-`, `Ratiss-experimental-IA-`, Aeon, Scientist Agent, ODV, Skynet, Cypher ODV, `documentation-ia`, `RATISS-ACADEMY`, `RATISS-ENGLISH`.
- **Biologie, signaux, génomique et données :** `RATISS-Neuralink-POC`, BIOLAB, `ratiss-bio`, bases-donnees, forks SPAdes / Unicycler / hifiasm / canu / Flye / megahit.
- **Infrastructure, calcul, communication et propriété :** HPC, GRID, Colab Agent, Crypto-VOLT, Crypto-net-veo, sites, portfolio, Discord, profil GitHub, Metaculus bot.
- **Références scientifiques, audits et archives :** Archives, Framework, ETALONS, GitHub-Audit, audit-public, DEEPDIVE, science-2, V10 Physical Complexity Audit, atelier, collapse-program, GTT, SYNTRIUM, QPU-COSMOS, quantum-circuit-studio, `ratiss-neuro` (fork).

Ce sont des familles du même écosystème; elles ne sont pas supprimées ni déclarées inutiles lorsqu’elles ne deviennent pas un composant du premier exécutable.

## Synthèse — le rôle du nouveau dépôt 🧶

**ENTRELACS** n’est pas une fusion de 81 dépôts et n’est plus rétréci à `CADENCE`. Le premier algorithme est une **matrice dynamique à repli pondéré** :

```text
signal + rythme → état continu x(t) ← W_eff(t) ← repli top-k de W(t)
                           │                       ↑
                           └─ co-activité / calibration ─┘
```

`DynamicMatrixProcessor` exécute maintenant une première version hypothétique de cette boucle. Les mécanismes de `Travaux`, FOCAL, NEURO, VEGA/X, ONE, Omni et PHOTON restent reliés dans la cartographie pour inspirer des variantes et contrôles; ils ne sont pas tous fusionnés dans le cœur. La version actuelle doit encore être comparée sur des tâches et ses règles provisoires doivent rester révisables. 💥