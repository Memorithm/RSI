# RSI — raccourcis.
.DEFAULT_GOAL := help

.PHONY: help install build test demo connect clean ci

## help : affiche cette aide (cible par défaut)
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //'

## install : compile et connecte RSI à ton agent IA (openclaw, hermes-agent…)
install:
	@./install.sh

## build : compile tous les binaires en release
build:
	cargo build --release --bins --locked

## test : lance toute la suite de tests
test:
	cargo test --locked

## demo : lance la simulation de démonstration
demo:
	cargo run --release --bin rsi-demo

## connect : (re)connecte le serveur MCP aux agents (sans recompiler)
connect:
	cargo run --release --bin rsi-connect

## ci : reproduit en local exactement les checks de la CI GitHub Actions
## (clippy -D warnings + tests, en défaut puis avec les features publiques,
## scirust inclus — même liste que ci.yml). À lancer avant de pousser.
PUBLIC_FEATURES := "wasm observability simd llm-ollama llm-claude-ureq scirust"
ci:
	cargo clippy --all-targets --locked -- -D warnings
	cargo test --locked
	cargo clippy --all-targets --locked --features $(PUBLIC_FEATURES) -- -D warnings
	cargo test --locked --features $(PUBLIC_FEATURES)

## clean : nettoie les artefacts de build
clean:
	cargo clean
	rm -f forge_checkpoint.json
