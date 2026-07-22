# Flywheel — auto-amélioration à deux étages (axe 4)

RSI produit, à chaque run DGM, des **trajectoires à vérité terrain** : un patch
(FIND→REPLACE) et le verdict `cargo build`+`test` qu'il a *réellement* produit.
Ces données servent à **fine-tuner un world model spécialisé du dépôt RSI**. La
boucle se referme : RSI s'améliore → ses traces améliorent le simulateur → le
simulateur (pré-crible + révision) accélère RSI.

## 1. Produire les trajectoires (`--export-trajectories`)

```
rsi-dgm . --goal "optimise kernels::matmul" --allow src/kernels.rs --bench "run --release --example bench_kernel" --model qwen3-coder:30b --prescreen-model agentworld --prescreen-num-predict 12288 --revise 2 --proposer-num-predict 8192 --steps 25 --export-trajectories fly_matmul.jsonl
```

`--proposer-num-predict N` relève le plafond de génération du **proposeur** (le
défaut 4096 tronque ses patchs sur de gros fichiers → no-ops, patchs qui ne
s'appliquent pas) → plus de propositions valides, donc plus de trajectoires par
run. `--temperature` / `--top-p` diversifient l'exploration quand le proposeur
répète.

Chaque ligne exportée est une paire `{prompt, completion}` : le `prompt` est
l'invite exacte du world model (fichier + patch), la `completion` une sortie
cargo réaliste terminée par la ligne machine `SIMCAL_VERDICT: compile=…; tests=…`.
Invariant : `parse_sim_verdict(completion)` redonne le verdict réel.

## 2. Accumuler sur des cibles variées (équilibre des classes)

Il faut les trois classes — **compile+pass**, **tests_fail**, **ne compile pas**
— d'où le mélange de cibles-qui-passent (json) et cibles-qui-cassent (matmul,
sum). Les cibles saturées ou où le proposeur cale remplissent vite les classes
négatives (précieuses).

```
for spec in "matmul|src/kernels.rs|bench_kernel" "sum|src/kernels.rs|bench_kernel" "transpose|src/kernels.rs|bench_kernel" "sha256|src/sha256.rs|bench_sha256" "json|src/json.rs|bench_json"; do
  IFS='|' read -r g f b <<< "$spec"
  for seed in 1 2 3; do
    ./target/release/rsi-dgm . --goal "optimise $g" --allow "$f" --bench "run --release --example $b" --model qwen3-coder:30b --prescreen-model agentworld --prescreen-num-predict 12288 --revise 2 --proposer-num-predict 8192 --steps 25 --seed $seed --export-trajectories "fly_${g}_s${seed}.jsonl"
  done
done
```

Le `--seed` diversifie les propositions donc réduit les doublons. Viser
**quelques centaines de paires** avant de fine-tuner (20 paires =
surapprentissage garanti).

## 3. Assembler le dataset (`examples/flywheel_dataset`)

```
cargo run --release --example flywheel_dataset -- fly_*.jsonl --out rsi_wm --chat --eval-frac 0.15
```

Fusionne, **déduplique**, mesure l'**équilibre des classes**, split train/eval
déterministe, et convertit au **format chat** (`{messages:[…]}`) consommable par
unsloth / llama-factory. Viser `équilibre : oui`. Sorties : `rsi_wm_train.jsonl`
et `rsi_wm_eval.jsonl`. Les `.jsonl` sont gitignorés (artefacts d'exécution).

## 4. Fine-tuner (hors dépôt, machine CUDA)

Il faut la **base en safetensors** (pas le GGUF ollama). LoRA sur le format chat :

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
model, tok = FastLanguageModel.from_pretrained("<repo AgentWorld base>", load_in_4bit=True, max_seq_length=20480)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)
ds = load_dataset("json", data_files="rsi_wm_train.jsonl", split="train")
ds = ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
SFTTrainer(model, tok, train_dataset=ds, args=SFTConfig(max_seq_length=20480, num_train_epochs=2)).train()
model.save_pretrained_gguf("agentworld-rsi", tok, quantization_method="q4_k_m")
```

```
ollama create agentworld-rsi -f agentworld-rsi/Modelfile
```

Caveats : AgentWorld est un MoE (35B-A3B) — vérifier le support unsloth de
l'archi (sinon **llama-factory** en repli) ; il faut les poids base HF (ou
convertir le GGUF, plus pénible). C'est un vrai travail d'ingé ML, env-dépendant.

## 5. Calibration v2 — la mesure qui prouve le flywheel

```
cargo build --release --features llm-ollama --bin rsi-simcal
./target/release/rsi-simcal . --goal "optimise kernels" --allow src/kernels.rs --bench "run --release --example bench_kernel" --sim-model agentworld-rsi --model qwen3-coder:30b --steps 20 --sim-num-predict 12288
```

Comparer la matrice de confusion à celle d'`agentworld` **générique** : si le
spécialisé a **plus de verdicts conclus** et **moins de faux positifs** (à zéro
faux négatif maintenu), le flywheel est prouvé — le pré-crible saute davantage de
builds sûrement perdus, la boucle accélère.

## Modules

| Étape | Module / binaire |
|---|---|
| Capture (par run) | `src/trajectory.rs` (`--export-trajectories`) |
| Assemblage (cross-run) | `src/flywheel.rs` + `examples/flywheel_dataset.rs` |
| Calibration | `src/bin/rsi_simcal.rs` + `src/simulation.rs` |
