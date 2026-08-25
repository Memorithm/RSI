# syntax=docker/dockerfile:1
#
# Image RSI — build statique musl du cœur std-only (zéro dépendance externe).
#
# Le périmètre est volontairement le CŒUR (features par défaut) : il ne tire
# aucune crate externe, donc le build est 100 % hors-ligne (pas d'accès
# crates.io) et produit des binaires entièrement statiques (musl). Les binaires
# à features git (rsi-full → forge/octasoma/ccos) sont hors périmètre : ils
# exigent réseau + dépendances amont et ne sont pas inclus ici.
#
#   docker build -t rsi:latest .
#   docker run --rm -i rsi:latest        # rsi-mcp en stdio (serveur MCP)
#                                        # (rsi-mcp ne parse aucun argument :
#                                        #  il lit stdin ; sans `-i`, EOF immédiat)
#   docker run --rm --entrypoint /usr/local/bin/rsi-demo rsi:latest

# ──────────────────────────────────────────────────────────────────────────
# Étage 1 — compilation statique musl.
# rust:alpine cible nativement x86_64-unknown-linux-musl ⇒ binaires statiques.
# NOTE reproductibilité : le tag `1.94-alpine` est flottant — pour un build
# bit-reproductible dans le temps, épingler par digest
# (`FROM rust@sha256:<digest> AS builder`).
# ──────────────────────────────────────────────────────────────────────────
FROM rust:1.94-alpine AS builder

# musl-dev fournit crt0/headers pour l'édition de liens statique.
RUN apk add --no-cache musl-dev

WORKDIR /build

# On copie manifestes + membres du workspace : le crate racine dépend
# (non-optionnellement) de cogno-core et le manifeste racine déclare les
# members — cargo exige leurs sources même pour un build std-only.
# Cargo.lock est requis : sans lui, `cargo` résout tout le manifeste et tente de
# cloner les deps git privées (octasoma/scirust) ⇒ échec.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY forge ./forge
COPY ccos ./ccos
COPY crates ./crates

# Features par défaut = std-only, AUCUNE dépendance crate activée ⇒ build
# hors-ligne. `--locked` fige la résolution sur le lock (pas de clone des deps
# privées). `--bins` saute rsi-full (required-features non activées).
ENV CARGO_NET_OFFLINE=true
RUN cargo build --release --bins --locked --target x86_64-unknown-linux-musl

# ──────────────────────────────────────────────────────────────────────────
# Étage 2 — image d'exécution minimale.
# scratch : les binaires musl sont 100 % statiques ⇒ aucune libc requise.
# ──────────────────────────────────────────────────────────────────────────
FROM scratch AS runtime

LABEL org.opencontainers.image.title="RSI" \
      org.opencontainers.image.description="Recursive Self-Improvement — cœur std-only (serveur MCP + démo)" \
      org.opencontainers.image.source="https://github.com/Memorithm/RSI" \
      org.opencontainers.image.licenses="PolyForm-Noncommercial-1.0.0 OR LicenseRef-Commercial"

ARG BIN_DIR=/build/target/x86_64-unknown-linux-musl/release
COPY --from=builder ${BIN_DIR}/rsi-mcp       /usr/local/bin/rsi-mcp
COPY --from=builder ${BIN_DIR}/rsi-demo      /usr/local/bin/rsi-demo
COPY --from=builder ${BIN_DIR}/rsi-refine    /usr/local/bin/rsi-refine
COPY --from=builder ${BIN_DIR}/rsi-ablate    /usr/local/bin/rsi-ablate
COPY --from=builder ${BIN_DIR}/rsi-loopbench /usr/local/bin/rsi-loopbench

# Non-root (UID numérique : scratch n'a pas /etc/passwd). Le serveur parse de
# l'entrée non fiable sur stdin ⇒ ne pas tourner en root.
USER 65534:65534

# Le serveur MCP parle en stdio ⇒ entrypoint par défaut.
ENTRYPOINT ["/usr/local/bin/rsi-mcp"]
