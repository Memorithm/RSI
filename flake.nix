{
  description = "RSI — Recursive Self-Improvement : moteur d'auto-amélioration encadrée (cœur std-only, sans dépendance externe)";

  inputs = {
    # Révisions épinglées (audit M14) : des refs mutables (`nixos-unstable`,
    # master implicite) résolvaient un snapshot différent à chaque build —
    # rustc changeait de version selon le jour. Un `flake.lock` committé
    # reste l'objectif dès qu'un environnement nix est disponible ; en attendant,
    # l'épinglage par révision rend les inputs immuables.
    nixpkgs.url = "github:NixOS/nixpkgs/56c02bc00adcf003215cc4bd996d6efaf4cff188";
    flake-utils.url = "github:numtide/flake-utils/11707dc2f618dd54ca8739b309ec4fc024de578b";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        manifest = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package;

        # Binaires du CŒUR (features par défaut, std-only). rsi-full est exclu :
        # il exige les features git forge/octasoma/ccos (réseau + amont privé),
        # incompatibles avec le bac à sable Nix (aucun accès réseau).
        coreBins = [
          "rsi-mcp"
          "rsi-demo"
          "rsi-connect"
          "rsi-ablate"
          "rsi-loopbench"
          "rsi-refine"
        ];

        # Double licence (cf. LICENSING.md). PolyForm-Noncommercial n'est pas
        # dans la liste nixpkgs ⇒ licence décrite à la main (non « free »).
        rsiLicense = {
          fullName = "PolyForm Noncommercial 1.0.0 OR Commercial (LicenseRef)";
          url = "https://github.com/Memorithm/RSI/blob/main/LICENSING.md";
          free = false;
        };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "rsi";
          version = manifest.version;
          src = pkgs.lib.cleanSource ./.;

          nativeBuildInputs = [ pkgs.cargo pkgs.rustc ];

          # Le cœur est SANS dépendance externe ⇒ build 100 % hors-ligne,
          # compatible avec le bac à sable Nix scellé (zéro accès réseau).
          # Cargo.lock est VERSIONNÉ et riche (deps optionnelles) : `--locked`
          # fige la résolution ; `--offline` garantit qu'aucune requête
          # registre n'est tentée.
          CARGO_NET_OFFLINE = "true";

          buildPhase = ''
            runHook preBuild
            export CARGO_HOME="$TMPDIR/cargo"
            # --locked : Cargo.lock est versionné ⇒ pas de ré-résolution (donc
            # aucun clone des deps git privées non activées par le défaut).
            cargo build --release --bins --offline --locked
            runHook postBuild
          '';

          # Tests désactivés DANS le build scellé : un test du cœur
          # (`papers_subprocess_path_via_echo`) appelle `/bin/echo` en chemin
          # absolu, absent du bac à sable Nix ⇒ non hermétique. La suite reste
          # déterministe et hors-ligne — on la joue dans `nix develop` (ou en
          # CI), où `/bin/echo` existe : `cargo test`.
          doCheck = false;

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin"
            for b in ${pkgs.lib.concatStringsSep " " coreBins}; do
              install -Dm755 "target/release/$b" "$out/bin/$b"
            done
            runHook postInstall
          '';

          meta = {
            description = manifest.description;
            homepage = "https://github.com/Memorithm/RSI";
            license = rsiLicense;
            mainProgram = "rsi-mcp";
            platforms = pkgs.lib.platforms.unix;
          };
        };

        # `nix run` lance le serveur MCP (stdio) par défaut.
        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/rsi-mcp";
        };
        apps.demo = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/rsi-demo";
        };

        # `nix develop` : toolchain complète (build + lints). NOTE : la cible
        # musl n'est PAS fournie ici (pas de rustup/multi-target dans ce shell) ;
        # pour le build statique Docker, passer par `docker build` directement.
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
          shellHook = ''
            echo "RSI devshell — cargo $(cargo --version)"
            echo "  build :  cargo build --release --bins"
            echo "  tests :  cargo test"
            echo "  lints :  cargo clippy --all-targets"
          '';
        };
      });
}
