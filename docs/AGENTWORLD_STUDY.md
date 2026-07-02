# Qwen-AgentWorld × RSI — évaluation des possibilités

**Objet évalué** : [QwenLM/Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld)
— « language world model » natif qui **simule des environnements agentiques**
(sept domaines unifiés : MCP, Search, **Terminal**, **SWE**, Android, Web, OS)
en prédisant l'état suivant de l'environnement à partir de l'action de
l'agent et de l'historique. Entraîné CPT→SFT→RL sur 10 M+ de trajectoires
réelles. Ouvert : **Qwen-AgentWorld-35B-A3B** (MoE, 3B actifs, contexte 256K)
et 397B-A17B, plus **AgentWorldBench** (benchmark 7 domaines à rubrique
multidimensionnelle). Licence **Apache 2.0**. GGUF communautaires disponibles
(Ollama-compatibles) ; déployable vLLM/SGLang en API OpenAI-compatible ;
tourne sur GB10/Grace-Blackwell → **le Jetson Thor le fait tourner** (même
gabarit que qwen3-coder:30b).

**Idée-clé pour RSI** : RSI possède un *gate empirique* parfait mais coûteux
(cargo build+test+bench ≈ 1-3 min/candidat). AgentWorld est exactement
l'inverse : un *simulateur* d'environnement Terminal/SWE rapide mais
faillible. Les deux se composent — **le simulateur pré-crible, le réel
tranche** — sans toucher à la doctrine (« le LLM propose, le moteur
dispose » devient « la simulation propose un ordre, le gate dispose »).

---

## Les 7 axes, par valeur décroissante

### 1. Pré-crible simulé de la boucle DGM (« surrogate gate ») — ★ le joyau

Le goulot du DGM est le coût d'évaluation : chaque candidat paie un build
complet. AgentWorld (domaines Terminal/SWE) peut **prédire le verdict** d'un
patch (compile ? tests verts ? sens du score ?) en quelques secondes.
Entonnoir à deux étages dans `dgm.rs` :

```
k candidats LLM → AgentWorld simule le gate → tri/élagage → seuls les
meilleurs paient le VRAI cargo build+test+bench → verdict autoritaire
```

Gain attendu : ×3-10 de candidats explorés par minute de build réel. Le gate
réel reste seul juge (fidélité de simulation = optimisation, jamais vérité).

### 2. Expérience de calibration — ★ à faire EN PREMIER, on a l'or en stock

Avant d'investir : mesurer la fidélité d'AgentWorld **sur nos données**.
L'archive DGM des campagnes Jetson contient des dizaines de patchs avec
verdict réel (ACCEPTÉ/rejeté, compile/tests/score — matmul, transpose, sum,
sha256, json). Rejouer chaque patch en simulation et mesurer
précision/rappel du verdict simulé. Coût : quelques heures de GPU, zéro
build. **Cette seule expérience décide de l'axe 1** — et c'est en soi un
résultat de recherche publiable (fidélité d'un world model sur un vrai dépôt
Rust avec vérité terrain).

### 3. Feedback anticipé dans le prompt du proposeur (Reflexion simulée)

Aujourd'hui le proposeur apprend des rejets *passés* (`recent_rejections`).
Avec AgentWorld : simuler le verdict du patch *avant* de le soumettre et
réinjecter « verdict simulé : casserait sum_matches_kahan » dans le prompt →
le proposeur se corrige **dans le même step**. Boucle interne
propose→simule→révise→évalue-réellement.

### 4. Volant de données (« data flywheel ») — RSI comme producteur

Chaque run DGM produit des trajectoires *réelles* (patch → sortie cargo →
verdict) au format exact dont un world model Terminal/SWE a besoin. Un
exportateur `archive → format trajectoires AgentWorld` permettrait :
(a) de contribuer des données rares (Rust, benchs de perf, vérité terrain) ;
(b) de **fine-tuner un world model spécialisé du dépôt RSI** — un simulateur
du *gate de RSI* — refermant la boucle : RSI s'améliore, ses traces
améliorent le simulateur, qui accélère RSI. C'est l'auto-amélioration à deux
étages — très aligné avec la thèse du projet.

### 5. Répétition de sûreté (chaos simulé, module criticality)

Cas d'usage officiel « simulation contrôlable : perturbations ciblées,
mondes fictifs ». Pour la doctrine de sûreté RSI : rejouer en simulation des
scénarios adversariaux (patch hostile, bench menteur, évaluateur compromis,
prompt-injection dans un paper) et vérifier que les garde-fous (gate,
allow-list, DRY-RUN, RPN/disjoncteur) tiennent — sans jamais toucher un
système réel. Test d'intrusion permanent et gratuit des défenses.

### 6. Ancrage d'Ω et évaluation de l'agent (AgentWorldBench)

`omega_tasks` définit 7 tâches à profils (D,M,R,A,C,V) synthétiques.
AgentWorldBench fournit 7 **domaines réels** avec rubrique — de quoi ancrer
Φ_x/g_x sur des mesures d'agent réelles, et évaluer les agents pilotés par
RSI sur un étalon externe reconnu.

### 7. Simulation du propre MCP de RSI

AgentWorld simule le domaine MCP — et RSI *est* un serveur MCP. On peut
simuler des clients (openclaw, hermes-agent, soullink…) contre la surface
`rsi_*` pour tester l'intégration hors-ligne, et générer des dialogues
d'agents synthétiques pour durcir l'API (fuzzing sémantique).

---

## Intégration dans l'écosystème (comment, concrètement)

AgentWorld n'est **pas un addon-outil** (pas un binaire type `papers`) :
c'est un **modèle**, il se branche par les backends LLM existants :

1. **Court terme (zéro code)** : GGUF sur Ollama du Thor →
   `OllamaClient::new("<tag agentworld>")` fonctionne tel quel (le client
   fixe déjà num_ctx/num_predict). La découverte automatique l'ignorera pour
   le rôle *proposeur* (ce n'est pas un modèle de code) — c'est un rôle
   nouveau : **simulateur**.
2. **Backend OpenAI-compatible** (`llm-openai`, ureq comme `llm-claude-ureq`)
   pour vLLM/SGLang — utile au-delà d'AgentWorld (tout endpoint OpenAI).
3. **`SimulatedEvaluator`** dans `dgm.rs` : même trait `Evaluator`, rend une
   `Fitness` *prédite* + flag `simulated` ; l'engine l'utilise en pré-crible
   (nouvelle config `prescreen: Option<…>`), jamais en juge final.
4. **`rsi-dgm --prescreen-model <tag>`** et outil MCP correspondant.

## Risques & réserves honnêtes

- **Fidélité inconnue sur notre niche** (Rust, benchs perf, gros repo) :
  d'où l'axe 2 d'abord — ne rien construire avant les chiffres.
- **Coût mémoire** : 2 modèles résidents (proposeur 30B + simulateur 35B) ≈
  40-50 Go quantifiés — le Thor (128 Go unifiés) les tient, mais Ollama
  déchargera peut-être l'un pour l'autre (latence de swap à mesurer).
- **Le simulateur peut halluciner des verdicts favorables** : c'est pourquoi
  il n'obtient JAMAIS d'autorité — uniquement l'ordre de passage devant le
  vrai gate (un faux positif coûte un build, comme aujourd'hui ; un faux
  négatif coûte une idée — à surveiller via l'axe 2).
- **Version vision (mmproj)** côté GGUF : servir en mode langage seul
  (`--language-model-only` sous vLLM ; backbone texte seul sous Ollama).

## Plan proposé

| # | Étape | Coût | Décide | État |
|---|-------|------|--------|------|
| 1 | GGUF AgentWorld sur le Thor + sonde manuelle | 1 h | faisabilité brute | ✅ **fait** |
| 2 | **Calibration** : proposeur réel → gate réel (vérité terrain) vs verdict simulé → matrice de confusion (`rsi-simcal`) | ½ j | tout le reste | ✅ **outillé** — à faire tourner |
| 3 | `SimulatedEvaluator` + pré-crible optionnel dans `dgm.rs` | 1-2 j | axe 1 | ⏳ après chiffres de l'étape 2 |
| 4 | Feedback anticipé dans le prompt (axe 3) | ½ j | — | ⏳ |
| 5 | Exportateur de trajectoires (axe 4) | 1 j | flywheel | ⏳ |
| 6 | Chaos simulé / AgentWorldBench (axes 5-6) | recherche continue | — | ⏳ |

### État — sonde réussie, harnais livré (Jetson Thor)

**Étape 1 ✅** : `Qwen-AgentWorld-35B-A3B` (GGUF Q4_K_M, ~21 Go) importé dans
Ollama (via `ollama create` depuis le blob en cache — le `pull` timeoutait à
la finalisation hf.co). Sonde (diff `acc += x` → `acc += x*2.0`) : le modèle a
prédit **exactement** `sum(&[1.0;100]) → 200.0`, test **FAILED**, avec un
`cargo test` réaliste (`left: 200.0, right: 100.0`) et l'explication juste. La
matière est bonne.

**Étape 2 ✅ outillée** : binaire **`rsi-simcal`** + module std-only
`src/simulation.rs` (constructeur de prompt, parseur de verdict — testé sur la
*vraie* sortie de la sonde —, matrice de confusion). Le harnais fait proposer
de vrais patchs (modèle de code), les juge par le **gate réel** (vérité
terrain) ET par le **world model**, et rend une matrice de confusion. Lancer
sur le Thor :

```bash
cargo build --release --features llm-ollama --bin rsi-simcal
rsi-simcal . --goal "optimise kernels" --allow src/kernels.rs \
  --bench "run --release --example bench_kernel" \
  --sim-model agentworld --model qwen3-coder:30b --steps 15
```

Décision go/no-go de l'axe 1 : viser une **exactitude « tests » élevée** avec
**peu de faux négatifs** (fn = amélioration réelle écartée à tort — le seul
coût dangereux d'un pré-crible ; un faux positif ne coûte qu'un build).
