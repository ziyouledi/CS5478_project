# Sourced (not run) by 11 / 15 / 16 for the unified task: the v4 teacher's training env, as recorded in its
# training log (job 1159044) and frozen manifest (expert-unified-cap0.05_hold10_v4-seed0). FACE_TURN_PLAN.md E3.
export HINGE_REFERENCE_JSON="${HINGE_REFERENCE_JSON:-${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json}"
export HINGE_EFFORT_LIMIT=0.05 FACE_TURN_HOLD_STEPS=10
export UNIFIED_TURN_ORIENT_TOL=0.4 UNIFIED_MAX_PHASE_STEPS=120 UNIFIED_TIMEOUT_TERMINATES=1
export UNIFIED_HINGE_PROG_SCALE=30.0 UNIFIED_ORIENT_PROG_SCALE=15.0 UNIFIED_HOLD_PROG_SCALE=5.0
export UNIFIED_ALIVE_BONUS=0.5 UNIFIED_FALL_PENALTY=-15.0 UNIFIED_EXTRA_OBS=1
export UNIFIED_START_STATES="${HOME}/data/low-dim-intent/runs/switch-eval-job1152498/any-up-states.npz"
export UNIFIED_START_FRACTION=0.5
unset UNIFIED_CYCLE FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING
unified_start_states_expected_sha256=08b013e48618dfac3aa91df0be87af9c6d82d56da6cc1a8569be1bf1c2be5909
if [[ "$(sha256sum "${UNIFIED_START_STATES}" | cut -d' ' -f1)" != "${unified_start_states_expected_sha256}" ]]; then
    echo "UNIFIED_START_STATES ${UNIFIED_START_STATES} does not match the v4 teacher's (sha256 ${unified_start_states_expected_sha256})" >&2
    exit 2
fi
# E3_DIAG (diagnostic of the goal-7 `any` failure, FACE_TURN_PLAN.md E3, not the main comparison):
#   notimeout  no per-phase time limit (UNIFIED_MAX_PHASE_STEPS=0), as in the stock Stage 1 / M1 task
#   anyonly    only `any` goals (UNIFIED_CYCLE=any; a fixed cycle loads no start states), timeout kept
case "${E3_DIAG:-}" in
    "") ;;
    notimeout) export UNIFIED_MAX_PHASE_STEPS=0 ;;
    anyonly) export UNIFIED_CYCLE=any ;;
    *) echo "unknown E3_DIAG=${E3_DIAG}" >&2; exit 2 ;;
esac
unified_teacher_name=expert-unified-cap0.05_hold10_v4-seed0
unified_teacher_file=expert-unified-v4-seed0.pth
