#!/usr/bin/env bash

find_sbt() {
  if [[ -n "${SBT_BIN:-}" && -x "${SBT_BIN}" ]]; then
    printf '%s\n' "${SBT_BIN}"
    return
  fi
  if command -v sbt >/dev/null 2>&1; then
    command -v sbt
    return
  fi
  if [[ -x "${HOME}/.local/share/coursier/bin/sbt" ]]; then
    printf '%s\n' "${HOME}/.local/share/coursier/bin/sbt"
    return
  fi
  printf '%s\n' "ERROR: sbt was not found. Install sbt 1.10.2 or set SBT_BIN." >&2
  return 1
}

configure_java_headers() {
  if [[ -z "${JAVA_HOME:-}" ]]; then
    local javac_path
    javac_path="$(command -v javac || true)"
    if [[ -n "${javac_path}" ]]; then
      JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${javac_path}")")")"
      export JAVA_HOME
    fi
  fi
  if [[ -n "${JAVA_HOME:-}" ]]; then
    export CPLUS_INCLUDE_PATH="${JAVA_HOME}/include:${JAVA_HOME}/include/linux${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  fi
}
