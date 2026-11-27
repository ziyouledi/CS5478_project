#!/usr/bin/env bash
# Submit evaluations (17) for a training array in chunks that fit under the queue limit.
# Usage: submit_block_evals.sh TRAIN_JOB DIRS_FILE FIRST LAST EVAL_TASK [submit]
#   DIRS_FILE: one line, space-separated run dirs; entry i <-> training task i of TRAIN_JOB.
#   Without "submit" it only prints what it would do.
# State: eval_state_<TRAIN_JOB>.txt, lines "<index> <eval job id>". Already-listed indices are skipped.
# Finished training tasks get no dependency, unfinished ones aftercorr:TRAIN_JOB. Every new eval array
# waits (afterany) for the previous eval array still in the queue, so at most EVAL_PARALLEL evaluation GPUs are used.
# EVAL_PARALLEL (default 2, user 2026-11-12): concurrent evaluations per array; arrays are chained, so at most this many run.
# EXTRA_DEP (optional) is appended to every dependency (e.g. afterany:<P1 job>).
set -eo pipefail
cd ~/cluster_smoke
train_job="$1"; dirs_file="$2"; first="$3"; last="$4"; eval_task="$5"; mode="${6:-dry}"
max_queue="${MAX_QUEUE:-32}"
state="eval_state_${train_job}.txt"
touch "${state}"
D="$(cat "${dirs_file}")"
COMMON="ALL,ISAACLAB_DIR=$HOME/software/isaaclab-stack/IsaacLab,VENV_DIR=$HOME/software/isaaclab-stack/env_isaaclab,EVAL_TASK=${eval_task}"

q=$(squeue -u "$USER" -r -h < /dev/null | wc -l)
room=$(( max_queue - q ))
done_ids=(); wait_ids=(); bad_ids=(); pending_total=0
for i in $(seq "${first}" "${last}"); do
  grep -q "^${i} " "${state}" && continue
  pending_total=$((pending_total + 1))
  (( ${#done_ids[@]} + ${#wait_ids[@]} >= room )) && continue
  st=$(sacct -X -n -j "${train_job}_${i}" -o State -P < /dev/null | head -1 | awk '{print $1}')
  case "$st" in
    COMPLETED) done_ids+=("$i") ;;
    PENDING|RUNNING|"") wait_ids+=("$i") ;;
    *) bad_ids+=("$i:$st") ;;
  esac
done
echo "queued=${q} room=${room} unsubmitted=${pending_total} done=${done_ids[*]} wait=${wait_ids[*]} bad=${bad_ids[*]}"
[[ "${mode}" == "submit" ]] || { echo "DRY (pass 'submit')"; exit 0; }
(( ${#done_ids[@]} + ${#wait_ids[@]} )) || { echo "nothing to submit"; exit 0; }

prev=$(awk '{print $2}' "${state}" | tail -1)
dep=""
if [[ -n "${prev}" ]] && squeue -h -j "${prev}" < /dev/null 2>/dev/null | grep -q .; then dep="afterany:${prev}"; fi
[[ -n "${EXTRA_DEP:-}" ]] && dep="${dep:+${dep},}${EXTRA_DEP}"
if (( ${#done_ids[@]} )); then
  E1=$(sbatch --parsable --array="$(IFS=,; echo "${done_ids[*]}")%${EVAL_PARALLEL:-2}" ${dep:+--dependency=${dep}} \
       --export="$COMMON,EVAL_RUN_DIRS=$D" 17_evaluate_students.sbatch < /dev/null)
  for i in "${done_ids[@]}"; do echo "$i $E1" >> "${state}"; done
  echo "eval_done=$E1 dep=${dep:-none}"; dep="afterany:${E1}${EXTRA_DEP:+,${EXTRA_DEP}}"
fi
if (( ${#wait_ids[@]} )); then
  E2=$(sbatch --parsable --array="$(IFS=,; echo "${wait_ids[*]}")%${EVAL_PARALLEL:-2}" \
       --dependency="aftercorr:${train_job}${dep:+,${dep}}" \
       --export="$COMMON,EVAL_RUN_DIRS=$D" 17_evaluate_students.sbatch < /dev/null)
  for i in "${wait_ids[@]}"; do echo "$i $E2" >> "${state}"; done
  echo "eval_wait=$E2 dep=aftercorr:${train_job}${dep:+,${dep}}"
fi
echo "queued_after=$(squeue -u "$USER" -r -h < /dev/null | wc -l)"
