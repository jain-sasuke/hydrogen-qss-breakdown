#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# Full pipeline runner for hydrogen CR QSS breakdown model.
#
# Usage:
#   ./run_pipeline.sh              # run everything (stages 1-5)
#   ./run_pipeline.sh --from 3     # start from stage 3 (skip 1-2)
#   ./run_pipeline.sh --only 4     # run only stage 4
#   ./run_pipeline.sh --dry-run    # print commands without running
#
# Stages:
#   1  Parse / process raw data   (parse_ccc, parse_tics, radiative, recombination)
#   2  Rate coefficients          (K_CCC, K_TICS, ionization, K_VS)
#   3  Assembly                   (pre_check, assemble_K_exc, assemble_cr_matrix)
#   4  Solve / validate           (solve_cr, validate_gates, qss_analysis)
#   5  Analysis / figures         (physics_tests, test_scaling, unified_scaling, plot_results)
# =============================================================================

set -euo pipefail

# ── Repo root ────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/anaconda3/envs/cr/bin/python"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m';  BOLD='\033[1m';      RESET='\033[0m'

# ── Argument parsing ─────────────────────────────────────────────────────────
FROM_STAGE=1
ONLY_STAGE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)  FROM_STAGE="$2"; shift 2 ;;
        --only)  ONLY_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
START_TIME=$(date +%s)

should_run() {
    local stage=$1
    [[ -n "$ONLY_STAGE" ]] && [[ "$stage" != "$ONLY_STAGE" ]] && return 1
    [[ "$stage" -lt "$FROM_STAGE" ]] && return 1
    return 0
}

run_script() {
    local stage=$1
    local label=$2
    local script=$3

    should_run "$stage" || { echo -e "  ${YELLOW}SKIP${RESET}  $label"; ((SKIP++)); return 0; }

    local log_file="$LOG_DIR/$(basename "$script" .py)_$(date +%H%M%S).log"
    echo -e "  ${CYAN}RUN ${RESET}  $label"
    echo -e "         ${BOLD}$script${RESET}"

    if $DRY_RUN; then
        echo -e "         ${YELLOW}[dry-run — not executed]${RESET}"
        return 0
    fi

    local t0=$(date +%s)
    if cd "$REPO" && "$PYTHON" "$script" > "$log_file" 2>&1; then
        local t1=$(date +%s)
        echo -e "         ${GREEN}PASS${RESET} in $((t1-t0))s  →  log: logs/$(basename "$log_file")"
        ((PASS++))
    else
        local t1=$(date +%s)
        echo -e "         ${RED}FAIL${RESET} after $((t1-t0))s  →  log: logs/$(basename "$log_file")"
        echo -e "         Last 10 lines:"
        tail -10 "$log_file" | sed 's/^/           /'
        ((FAIL++))
        echo -e "\n${RED}Pipeline stopped at: $label${RESET}"
        echo -e "Fix the error above, then re-run with:  ./run_pipeline.sh --from $stage\n"
        exit 1
    fi
}

stage_header() {
    local stage=$1; local title=$2
    if should_run "$stage"; then
        echo ""
        echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
        echo -e "${BOLD}  STAGE $stage — $title${RESET}"
        echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    fi
}

# ── Header ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Hydrogen CR QSS Breakdown Model — Full Pipeline Runner     ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo -e "  Repo:    $REPO"
echo -e "  Python:  $PYTHON"
echo -e "  Logs:    $LOG_DIR"
$DRY_RUN && echo -e "  ${YELLOW}Mode:    DRY RUN (no scripts executed)${RESET}"
[[ -n "$ONLY_STAGE" ]] && echo -e "  Mode:    ONLY stage $ONLY_STAGE"
[[ "$FROM_STAGE" -gt 1 ]] && echo -e "  Mode:    Start from stage $FROM_STAGE"
echo -e "  Time:    $(date '+%Y-%m-%d %H:%M:%S')"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Parse and process raw data
# ─────────────────────────────────────────────────────────────────────────────
stage_header 1 "Parse and Process Raw Data"
run_script 1 "1.1  parse_ccc"          src/parsers/parse_ccc.py
run_script 1 "1.2  parse_tics"         src/parsers/parse_tics.py
run_script 1 "1.3  radiative_rates"    src/rates/radiative_rates.py
run_script 1 "1.4  recombination_rates" src/rates/recombination_rates.py

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Rate coefficients
# ─────────────────────────────────────────────────────────────────────────────
stage_header 2 "Rate Coefficients"
run_script 2 "2.1  compute_K_CCC"      src/rates/compute_K_CCC.py
run_script 2 "2.2  compute_K_TICS"     src/rates/compute_K_TICS.py
run_script 2 "2.3  ionization_rates"   src/rates/ionization_rates.py
run_script 2 "2.4  compute_K_VS  ***"  src/rates/compute_K_VS.py

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Assembly
# ─────────────────────────────────────────────────────────────────────────────
stage_header 3 "Assembly"
run_script 3 "3.1  pre_assembly_check" src/rates/pre_assembly_check.py
run_script 3 "3.2  assemble_K_exc"     src/rates/assemble_K_exc.py
run_script 3 "3.3  assemble_cr_matrix" src/rates/assemble_cr_matrix.py

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Solve and validate
# ─────────────────────────────────────────────────────────────────────────────
stage_header 4 "Solve and Validate"
run_script 4 "4.1  solve_cr"           src/rates/solve_cr.py
run_script 4 "4.2  validate_gates"     src/validation/validate_gates.py
run_script 4 "4.3  qss_analysis"       src/validation/qss_analysis.py

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Analysis and figures
# ─────────────────────────────────────────────────────────────────────────────
stage_header 5 "Analysis and Figures"
run_script 5 "5.1  physics_tests"      src/analysis/physics_tests.py
run_script 5 "5.2  test_scaling"       src/analysis/test_scaling.py
run_script 5 "5.3  unified_scaling"    src/analysis/unified_scaling.py
run_script 5 "5.4  plot_results"       src/analysis/plot_results.py

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  PIPELINE COMPLETE${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  ${GREEN}PASS${RESET}  $PASS scripts"
echo -e "  ${RED}FAIL${RESET}  $FAIL scripts"
echo -e "  ${YELLOW}SKIP${RESET}  $SKIP scripts"
echo -e "  Time: ${ELAPSED}s  ($(date '+%H:%M:%S'))"
echo ""