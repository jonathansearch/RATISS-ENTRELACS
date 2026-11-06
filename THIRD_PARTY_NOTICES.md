# Notices de composants adaptés 📜

Le cœur original de RATISS-ENTRELACS n’a pas encore de licence publique; aucune licence de dépôt n’est choisie ici.

## Composants adaptés de `jonathansearch/Travaux`

Les modules suivants sont des adaptations sous Apache License 2.0 de logiques présentes dans [`Travaux`](https://github.com/jonathansearch/Travaux), Copyright 2026 RATISS Labs :

- `src/ratiss_entrelacs/domains/ssh_reference.py` — adaptation du modèle SSH et de la propagation RK4 de `ratiss_photoinduced/ssh_model.py`.
- `src/ratiss_entrelacs/topology.py` — adaptation de la filtration Vietoris–Rips et de l’observable `P_sig` de `ratiss_photoinduced/topology.py`.

Les adaptations ajoutent validations d’entrées, annotations de types, points d’extension et fonctions séparées; elles ne constituent pas une validation nouvelle des résultats physiques. Le texte complet de la licence applicable à ces adaptations se trouve dans [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). Les autres composants du dépôt restent séparés et ne sont pas présentés comme dérivés de ces fichiers.
