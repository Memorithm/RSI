# AUTOPILOT — coder seul sous supervision (loop engineering)

> Objectif : passer du *prompting* à la *supervision*. L'humain donne un
> **objectif**, l'agent pose **une** batterie de questions, produit une **spec
> contractuelle**, la découpe en tâches vérifiables, code en autonomie, et
> livre **PR après PR**. L'humain ne fait plus que trois choses : valider la
> spec, valider les tests, reviewer les PRs. Sa review devient la **fonction
> de fitness terminale** du système.

Ce document fixe l'architecture AVANT le code : frontières, garde-fous,
métriques et paliers de livraison. Il prolonge `docs/AGENTWORLD_STUDY.md`
(axes 1-5, livrés) et `docs/FLYWHEEL.md` (data flywheel).

---

## 0. Position dans l'écosystème RSI

RSI possède déjà le composant le plus dur d'un agent de codage autonome :
**un juge objectif de son propre travail**. Le gate DGM (build + test + bench,
tout-au-vert, élitisme, anti-bruit, snapshot isolé) est écrit, éprouvé et
attaqué en permanence par `src/chaos.rs`. AUTOPILOT est la couche
*au-dessus du patch* :

| Boucle | Rayon | Existant |
|---|---|---|
| **Intérieure** — DGM | un patch → un verdict mesuré | `src/dgm.rs` ✅ |
| **Simulée** — révision | un patch prédit cassé → corrigé avant le gate | `--revise` ✅ |
| **Extérieure** — AUTOPILOT | un objectif → des PRs | ce document |
| **Apprentissage** — flywheel | des verdicts → un world model meilleur | `src/flywheel.rs` ✅ |
| **Terminale** — humain | une PR → approve / reject | GitHub, déjà là |

Le prompting ne disparaît pas : il est **amorti**. On spécifie une fois, en
structuré, au lieu de re-prompter à chaque étape.

## 1. Le cycle complet

```
OBJECTIF (humain, une phrase)
   │
   ▼
INTAKE ── exploration du dépôt ──► BATTERIE DE QUESTIONS (une seule, groupée)
   │                                        │ réponses humaines
   ▼                                        ▼
SPEC contractuelle (specs/<nom>.md) ◄───────┘
   │ validation humaine → SPEC GELÉE
   ▼
DÉCOMPOSITION ──► DAG de tâches {allowlist, critère de done, budget, deps}
   │
   ▼  pour chaque tâche, dans l'ordre du DAG :
┌─────────────────────────────────────────────────────────────┐
│ régime FEATURE : tests d'abord (mini-PR "tests only",       │
│   validée humain) → implémentation, tests GELÉS             │
│ régime PERF    : gate DGM existant (--bench) inchangé       │
│        boucle proposer → prescreen → gate → accepter        │
│        chaque évaluation réelle → trajectoire flywheel      │
└─────────────────────────────────────────────────────────────┘
   │ gate vert + chaos rehearsal OK
   ▼
PR DRAFT (branche autopilot/<spec>/<tâche>, jamais main)
   │
   ▼
REVIEW HUMAINE ── approve+merge ──► trajectoire POSITIVE ──► flywheel
   │              request changes ─► commentaires = contexte de révision
   │              close ───────────► trajectoire NÉGATIVE ─► flywheel
   ▼
world model + proposeur fine-tunés → les cycles suivants sont plus rapides
```

## 2. Intake — de l'objectif à la spec contractuelle

1. L'agent lit l'objectif, **explore le dépôt d'abord** (code, docs, tests,
   historique) — les questions viennent après l'enquête, jamais avant.
2. Il pose **une seule batterie de questions groupées** : ambiguïtés réelles,
   arbitrages que seul l'humain peut trancher. Pas de dialogue au fil de l'eau.
3. Sortie : `specs/<objectif>.md` contenant :
   - **Contexte** : ce que l'agent a compris (reformulation vérifiable) ;
   - **Critères d'acceptation** : chacun testable mécaniquement ;
   - **Hors-périmètre** : explicite (le scope creep se refuse par écrit) ;
   - **Budget** : étapes LLM, temps mur, tokens ;
   - **Allowlist** : fichiers que l'objectif autorise à toucher.
4. Validation humaine → la spec est **GELÉE**. Toute dérive découverte en
   cours de route = amendement de spec re-validé, jamais une décision
   silencieuse de l'agent.

## 3. Décomposition — spec → tâches vérifiables

- La spec devient un **DAG de tâches**. Chaque tâche :
  `{fichiers cibles (sous-allowlist), critère de done exécutable, budget,
  dépendances}`.
- Granularité cible : **une tâche = une PR reviewable en < 10 minutes**.
  C'est la contrainte dimensionnante — l'humain est la ressource rare.
- Une tâche sans critère de done exécutable est mal née : elle retourne en
  décomposition, elle n'atteint jamais l'exécution.

## 4. Exécution — deux régimes de gate

**Régime PERF** (existant) : la tâche a un benchmark (`--bench`). Le gate DGM
actuel s'applique tel quel : tout-au-vert + gain de score > `min_score_gain`.

**Régime FEATURE** (nouveau) : pas de benchmark — et un piège de circularité :
si l'agent écrit les tests ET le code, il se note lui-même. On casse la
circularité en deux temps :

1. L'agent écrit **les tests d'abord**, depuis la spec → mini-PR « tests
   only ». L'humain la valide vite : c'est la reformulation exécutable de la
   spec, pas de l'implémentation.
2. L'implémentation se fait ensuite avec ces tests comme gate. Les fichiers
   de test sortent de l'allowlist : **l'agent n'a plus le droit d'y toucher**.
   « Corriger le test au lieu du code » devient structurellement impossible.

Dans les deux régimes, chaque évaluation réelle exporte sa trajectoire
(`src/trajectory.rs`) — le flywheel tourne en permanence, gratuitement.

## 5. Émission — une tâche, une branche, une PR

- Branche `autopilot/<spec>/<tâche>`. **Jamais `main`.**
- PR **draft** générée automatiquement :
  - *Quoi / pourquoi* : lien vers la spec et le critère de done ;
  - *Preuves* : tests verts, bench avant/après, chaos rehearsal OK ;
  - *Risques connus* : ce que l'agent n'a pas pu vérifier, dit honnêtement.
- L'agent s'abonne aux événements de la PR (reviews, CI) et itère jusqu'à
  merge, close, ou épuisement du budget.

## 6. Review humaine = fitness terminale

| Verdict humain | Effet boucle | Effet flywheel |
|---|---|---|
| approve + merge | tâche close, suivante du DAG | trajectoire **positive** `{spec, diff, verdict}` |
| request changes | commentaires injectés comme contexte de révision (même mécanique que `--revise`, feedback humain au lieu du world model) | trajectoire intermédiaire |
| close | tâche retourne en décomposition | trajectoire **négative** |

C'est la donnée la plus riche que le système puisse produire : un dataset
`{spec, diff, verdict humain}` fine-tune non seulement le **world model**
(prédire le gate) mais le **proposeur** lui-même (proposer ce que l'humain
accepte). La supervision *est* l'entraînement.

## 7. Garde-fous — chaos-testés AVANT activation

Règle : **aucun garde-fou n'est réputé exister tant que `src/chaos.rs` n'a
pas un scénario adversarial qui prouve qu'il contient l'attaque.**

| Garde-fou | Attaque contenue | Scénario chaos |
|---|---|---|
| Gate tout-au-vert | score menteur, tests rouges | ✅ existant |
| Élitisme | régression adoptée | ✅ existant |
| `min_score_gain` | bruit de mesure « validé » | ✅ existant |
| DRY-RUN / snapshots | écriture de l'arbre vivant | ✅ existant |
| Allowlist par tâche | patch hors périmètre | ✅ existant |
| Tests gelés (régime feature) | « corriger » le test au lieu du code | à écrire |
| Branches only, jamais main | push direct sur main | à écrire |
| Budget par objectif | boucle infinie, coût non borné | à écrire |
| Arrêt sur N échecs consécutifs | acharnement sur tâche insoluble | à écrire |
| Spec gelée | scope creep silencieux | à écrire |

Interdits absolus (aucune configuration ne les lève) : merger sans humain,
modifier ses propres garde-fous, élargir sa propre allowlist, dépasser son
budget. L'auto-modification du moteur reste possible — mais *via PR reviewée*,
comme tout le reste.

## 8. Métriques de boucle (le « loop engineering » proprement dit)

Les boucles deviennent des objets de premier ordre, mesurés :

- **Temps de cycle** : objectif → première PR ; PR → merge.
- **Rendement** : PRs acceptées / PRs émises (le KPI central).
- **Coût par PR acceptée** : tokens + temps mur, toutes tentatives incluses.
- **Latence de fermeture flywheel** : combien de cycles entre un lot de
  verdicts et une amélioration *mesurée* du prescreen (taux de skip, précision).
- **Taux de skip du pré-crible** : builds économisés / builds candidats.

Implémentation : `src/loop_metrics.rs`, qui **agrège l'existant**
(`revisions`, `prescreen_skips`, `trajectories`, verdicts de PR) — pas une
réécriture des boucles.

## 9. Paliers de livraison

| Palier | Contenu | Critère de done |
|---|---|---|
| **0 — infra fiable** (EN COURS, bloquant) | Ollama stable : services concurrents coupés, `OLLAMA_KEEP_ALIVE`, `--timeout` adapté | un run DGM de 20 étapes, deux modèles, **zéro** `os error 11` |
| **1 — intake** | binaire `rsi-autopilot` : objectif → exploration → questions → `specs/<nom>.md` | une spec validée par l'humain sans réécriture manuelle |
| **2 — décomposition + exécution** | DAG de tâches, exécution séquentielle réutilisant le moteur DGM, sortie = diffs locaux | un objectif multi-tâches livré en diffs, gate vert |
| **3 — émission PR** | branches + PR drafts automatiques, abonnement aux reviews | une PR émise, reviewée, mergée sans intervention manuelle hors review |
| **4 — flywheel humain** | ingestion des verdicts de review, `loop_metrics` | rendement et temps de cycle mesurés sur ≥ 10 PRs |
| **5 — multi-objectifs** | priorisation, mémoire longue (specs passées comme knowledge) | deux objectifs menés en parallèle sans collision d'allowlist |

Chaque palier est utilisable seul. On ne commence un palier N+1 qu'avec le
palier N démontré — le palier 0 d'abord : construire l'autonomie sur une
infra qui timeout au premier appel donnerait un agent qui échoue en autonomie
au lieu d'échouer sous supervision.

## 10. Ce que ça ne fait jamais

- Pousser sur `main`.
- Merger une PR.
- Modifier ses garde-fous, son budget, son allowlist, ou `src/chaos.rs`
  en dehors d'une PR explicitement dédiée et reviewée.
- Continuer après épuisement du budget ou N échecs consécutifs.
- Décider seul qu'une spec « voulait dire » autre chose.
