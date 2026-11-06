# Algorithme candidat — processeur inspiré du calcul cérébral 🧠⚙️

**Version logicielle : 0.4. Hypothèse formalisée et simulable, pas mécanisme biologique validé.** L’inspiration est explicite : intégration dendritique et somatique, voies excitatrices/inhibitrices, seuils et décharges, adaptation, rythmes et plasticité dépendante du calendrier des événements. La matrice adaptative et son repli top-k sont des mécanismes candidats à l’intérieur de ce cadre — pas une fin matricielle générique détachée du cerveau.

Les équations ci-dessous sont des choix de prototype : elles ne copient pas un modèle neuronal publié dans son intégralité, ne représentent pas le cerveau entier et n’établissent aucun avantage computationnel. Le but est de pouvoir les ablater, les critiquer et les remplacer. 🔬

## 1. État et conventions

Pour `n` unités, `B` branches par unité et une entrée `u(t)`, le système tient :

- `x ∈ ℝⁿ` : état continu somatique; dans la variante dendritique, il lit la membrane après reset;
- `W_E, W_I ∈ ℝ₊ⁿˣᴮˣⁿ` : poids non négatifs excitateurs/inhibiteurs par récepteur, branche et source;
- `W = Σ_b (W_E[:,b,:] − W_I[:,b,:])` : vue signée agrégée, conservée pour compatibilité et inspection;
- `d ∈ ℝⁿˣᴮ` : potentiels dendritiques par branche, `v ∈ ℝⁿ` : membrane somatique, `a ∈ ℝⁿ` : adaptation;
- `φ ∈ [0,2π)ⁿ` : phases; `R ∈ ℝᵐˣⁿ` : lecture calibrable de sortie;
- des poids d’entrée par branche `B_in ∈ ℝⁿˣᴮˣᵈ` et, en mode STDP, des traces d’événements pré/post.

`W_E[i,b,j]` et `W_I[i,b,j]` désignent des influences de la source `j` vers le récepteur `i`, via sa branche `b`. Les paramètres sont en unités logicielles arbitraires, sans prétention de valeurs physiologiques ni d’implémentation physique.

API : `initial_weights` accepte l’ancienne matrice signée `(n,n)` (ou un tenseur `(n,B,n)`) et la sépare par signe; `initial_excitatory_weights` et `initial_inhibitory_weights` prennent des magnitudes non négatives `(n,n)` ou `(n,B,n)`. Les entrées explicites `input_branch_weights` ont la forme `(n,B,d)`. Les propriétés `weights`, `excitatory_weights` et `inhibitory_weights` exposent des copies agrégées, tandis que les propriétés `*_branch_weights` gardent la dimension des branches.

## 2. Repli rythmé des relations E/I

La compatibilité entre phases du récepteur et de la source est :

```text
Cᵢⱼ = (1 − ρ) + ρ · (1 + cos(φᵢ − φⱼ)) / 2
Aᵢᵦⱼ = W_E[i,b,j] + W_I[i,b,j]
Sᵢᵦⱼ = Aᵢᵦⱼ · Cᵢⱼ · (c₀ + |xⱼ|)
```

Pour chaque couple récepteur-branche, on choisit jusqu’à `k` sources selon `S`. La softmax répartit la masse de chaque branche, puis un facteur commun rescale au besoin toutes les branches du récepteur afin que `Σ_b,j (W_E_eff + W_I_eff) ≤ b`. Les voies E/I restent séparées même si elles partagent la même paire source-cible :

```text
αᵢᵦⱼ = softmax(Sᵢᵦⱼ / T)
mᵢᵦ = min(b, Σⱼ retenues Aᵢᵦⱼ)
p̃ᵢᵦⱼ = mᵢᵦ · αᵢᵦⱼ · Cᵢⱼ
sᵢ = min(1, b / Σᵦ,ⱼ p̃ᵢᵦⱼ)
pᵢᵦⱼ = sᵢ · p̃ᵢᵦⱼ
W_E_eff = p · W_E / (W_E + W_I)
W_I_eff = p · W_I / (W_E + W_I)
```

Les deux matrices effectives sont nulles hors des liens sélectionnés. Le noyau d’interaction utilise ces liens, mais **le code inspecte encore les poids denses pour scorer et trier** : aucun gain de vitesse, coût ou énergie n’est établi.

Le motif top-k/softmax est une hypothèse de sélection relationnelle. Le maintien de voies E/I distinctes évite qu’un lien excitateur et un lien inhibiteur co-localisés s’annulent avant l’intégration; les poids sont toujours bornés ensemble. L’agrégat signé `W` n’est qu’une vue, pas l’état interne complet.

## 3. Entrée, branches dendritiques, soma et événements

L’entrée pondérée de chaque branche est séparée en courant positif et négatif. Pour les connexions récurrentes, la partie rectifiée `rⱼ = max(xⱼ, 0)` sert de proxy analogique de relâchement; les amplitudes des courants E/I sont donc non négatives :

```text
J_E[i,b] = Σⱼ W_E_eff[i,b,j] · rⱼ + max((B_in · u)[i,b], 0)
J_I[i,b] = Σⱼ W_I_eff[i,b,j] · rⱼ + max(−(B_in · u)[i,b], 0)
```

La porte d’intégration vaut `g_i = (1−ρ) + ρ·p_i`, avec `p_i=(1+sin φ_i)/2` par défaut ou une impulsion externe. Elle est partagée par les branches d’une unité. La variante simple (une branche, intégrateur fuyant) reste un contrôle. La variante dendritique met à jour chaque compartiment séparément :

```text
d[i,b] ← d[i,b] + dt/τ_d · (g_i·(J_E[i,b] − J_I[i,b]) − d[i,b])
q[i,b] = d[i,b]                       # branche linéaire
# ou q[i,b] = tanh(d[i,b])             # saturation locale optionnelle
s[i] = Σ_b gain_b · q[i,b]
a⁻[i] = a[i] · exp(−dt/τ_a)
v*[i] = v[i] + dt/τ_m · (g_d · s[i] − g_a · a⁻[i] − v[i])
e[i] = 1 si v*[i] ≥ θ
v[i] ← reset si e[i]=1, sinon v*[i]
a[i] ← a⁻[i] + jump · e[i]
x[i] ← v[i]
```

Chaque branche garde son propre potentiel et son courant E/I; leur signal est combiné au soma. L’adaptation, le seuil et le reset agissent somatiquement. Le calcul exact actuel est résumé par `DendriticPopulation`; la non-linéarité `tanh` locale est optionnelle, pas une loi imposée. Les vecteurs par unité fournis à plusieurs branches sont distribués également; `input_branch_weights` permet un routage explicitement distinct.

Dans le mode intégrateur simple, `x` suit une cible `tanh(Σ_b (J_E−J_I))` par Euler fuyant,

```text
xᵢ(t+dt) = xᵢ(t) + (dt/τ)·gᵢ·(tanh(Σ_b(J_E−J_I)) − xᵢ(t))
```

et `events` marque le franchissement ascendant du seuil. Les deux variantes gardent un état continu; les événements servent de signaux de lecture/commande, pas de remplacement du calcul analogique.

## 4. Plasticité : deux règles candidates 🧬

La configuration choisit `plasticity_rule="coactivity"` (contrôle historique) ou `plasticity_rule="stdp"`. Les deux utilisent l’enveloppe rythmique et le budget combiné E/I; `matrix_learning_rate=0` gèle l’apprentissage.

### Co-activité locale

```text
Dᵢⱼ = η·dt·Cᵢⱼ·xᵢ(t+dt)·xⱼ(t)
W_E ← exp(−λdt)·W_E + max(D, 0)/B
W_I ← exp(−λdt)·W_I + max(−D, 0)/B
```

Les incréments positifs alimentent la voie E et les négatifs la voie I. Ce partage reproduit la mise à jour signée sur l’agrégat sans écraser des poids E et I distincts déjà présents. Les poids sont ensuite bornés à `Σ_b,j (W_E + W_I)[i,b,j] ≤ b`.

### STDP par paires et traces exponentielles

Les événements d’une unité jouent le rôle de décharge présynaptique sur ses sorties et postsynaptique sur ses entrées. Les traces sont d’abord amorties, puis les paires d’événements séparées dans le temps modifient les poids :

```text
p⁻[j] = p[j] · exp(−dt/τ_pre)       q⁻[i] = q[i] · exp(−dt/τ_post)
Δ[i,j] = A_plus·e[i]·p⁻[j] − A_minus·q⁻[i]·e[j]
Δ_E = η · C ⊙ Δ / B                 Δ_I = η · s_I · C ⊙ Δ / B
W_E ← max(0, exp(−λdt)·W_E + Δ_E)
W_I ← max(0, exp(−λdt)·W_I + Δ_I)
p ← p⁻ + e                         q ← q⁻ + e
```

Ainsi une décharge pré puis post favorise le terme `A_plus`; l’ordre post puis pré active le terme `A_minus`. Les événements simultanés n’ont pas d’effet à délai nul dans cette règle. Les traces avancent même si la porte logique suspend l’update de poids, afin que l’historique temporel reste explicite; la porte ne décide que de la modification de matrice. `s_I` règle le même noyau candidat sur la magnitude inhibitrice. Ce choix est configurable et **ne prétend pas être la règle de plasticité inhibitrice du cerveau**. Les amplitudes, fenêtres, signe, retard d’update et règles E/I sont à ablater.

### Configuration minimale des trois extensions

```python
import numpy as np
from ratiss_entrelacs import (
    DendriticConfig, DendriticPopulation, DynamicMatrixProcessor,
    MatrixProcessorConfig, STDPConfig,
)

n, branches, input_dim = 8, 2, 3
cells = DendriticPopulation(
    n, DendriticConfig(branch_nonlinearity="tanh"),
    branches=branches, branch_gains=[1.0, 0.8],
)
branch_input = np.zeros((n, branches, input_dim))
branch_input[:, 0, 0] = 1.0       # exemple : feature 0 vers branche 0
branch_input[:, 1, 1] = 1.0       # feature 1 vers branche 1
processor = DynamicMatrixProcessor(
    input_dim,
    config=MatrixProcessorConfig(units=n, top_k=3, plasticity_rule="stdp"),
    dendritic_layer=cells,
    input_branch_weights=branch_input,
    initial_excitatory_weights=np.zeros((n, branches, n)),
    initial_inhibitory_weights=np.zeros((n, branches, n)),
    stdp_config=STDPConfig(tau_pre=0.2, tau_post=0.2),
)
```

## 5. Portes logiques : contrôle au tick suivant

```text
état continu → détecteurs de seuil → événements e(t)
                                      ↓
                             portes AND/OR/XOR/NOT…
                                      ↓
                routes, plasticité, readout au tick t+dt
```

`LogicGateLayer` lit les événements du tick précédent. Les portes ne remplacent ni l’intégration dendritique ni les poids continus; elles commandent un tick ultérieur, ce qui rend le délai testable et évite une boucle combinatoire instantanée.

## 6. Boucle complète d’un pas

1. Évaluer les bindings logiques à partir des événements précédents.
2. Calculer l’impulsion rythmique et les compatibilités de phase.
3. Replier, branche par branche, les poids E/I en routes actives; appliquer le contrôle de routage.
4. Projeter l’entrée vers les branches, séparer ses signes et calculer les courants E/I récurrents.
5. Mettre à jour les dendrites, puis membrane/adaptation somatiques; émettre les événements.
6. Avancer les traces STDP; modifier les poids par la règle choisie si l’apprentissage et sa porte sont actifs; appliquer le budget.
7. Produire le readout `y=R·x` si autorisé; avec une cible, calibrer par LMS normalisé `e=y_cible−R·x`, `R←R+μ·e·xᵀ/(ε+||x||²)`.
8. Avancer les phases; conserver activité, événements, courants et compartiments dans la trace.

Implémentation : `src/ratiss_entrelacs/cells.py` (`DendriticPopulation`), `matrix_processor.py` (`DynamicMatrixProcessor`, `weighted_fold_channels`, `STDPConfig`) et `logic.py` (`LogicGateLayer`).

## 7. Contrôles indispensables avant une affirmation de calcul

Comparer sur mêmes tâches, entrées, graines et budget :

- une branche contre plusieurs, routage identique contre routage distinct, transfert local linéaire contre saturant;
- courant signé agrégé contre voies E/I indépendantes; voies E/I fixes contre plasticité séparée;
- co-activité contre STDP, ordre causal contre anti-causal, traces permutées, événements simultanés, STDP désactivée;
- cellule dendrite/soma contre intégrateur simple; adaptation activée/coupée; seuil et reset variés;
- rythme nul/actif/permuté; top-k/dense et portes bypassées/actives;
- résultat externe, stabilité, routes, nombre d’événements, poids E/I, mémoire, coût et densité active, gardés comme mesures distinctes.

La suite de tests vérifie des invariants logiciels et des micro-cas de timing. La démo de mémoire demeure un smoke test; ni supériorité, généralisation, économie d’énergie, mécanisme biologique complet ou réalisation physique ne sont établis. Une tâche tenue à l’écart et des contrôles appariés sont le prochain travail de recherche. 🧪
