#!/usr/bin/env bash

set -Eeuo pipefail

label="${1:-final}"
case "${label}" in
    best|final) ;;
    *) echo "Usage: EVAL_JOB_ID=<job> $0 {best|final}" >&2; exit 2 ;;
esac

: "${EVAL_JOB_ID:?Set EVAL_JOB_ID to the completed 1024-episode evaluation job}"

data_root="${DATA_ROOT:-${HOME}/data/low-dim-intent}"
run_root="${data_root}/runs/expert-eval-job${EVAL_JOB_ID}"
result_json="${run_root}/${label}.json"
freeze_dir="${data_root}/checkpoints/expert-seed0/frozen"
target="${freeze_dir}/expert-seed0.pth"
manifest="${freeze_dir}/manifest.json"
hash_file="${freeze_dir}/SHA256SUMS"
evaluation_copy="${freeze_dir}/evaluation.json"

test -f "${result_json}"
source_checkpoint=$(python3 - "${result_json}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["checkpoint"])
PY
)
test -f "${source_checkpoint}"

mkdir -p "${freeze_dir}"
source_sha=$(sha256sum "${source_checkpoint}" | awk '{print $1}')

if [[ -e "${target}" ]]; then
    target_sha=$(sha256sum "${target}" | awk '{print $1}')
    if [[ "${target_sha}" != "${source_sha}" ]]; then
        echo "Refusing to replace existing frozen checkpoint with different contents: ${target}" >&2
        exit 3
    fi
    echo "frozen_checkpoint_already_matches=true"
else
    temporary_target="${target}.tmp.$$"
    trap 'rm -f "${temporary_target:-}"' EXIT
    cp "${source_checkpoint}" "${temporary_target}"
    copied_sha=$(sha256sum "${temporary_target}" | awk '{print $1}')
    if [[ "${copied_sha}" != "${source_sha}" ]]; then
        echo "Checksum mismatch after copying checkpoint" >&2
        exit 4
    fi
    mv "${temporary_target}" "${target}"
    trap - EXIT
fi

cp "${result_json}" "${evaluation_copy}"
printf '%s  %s\n' "${source_sha}" "$(basename "${target}")" > "${hash_file}"

python3 - \
    "${result_json}" "${manifest}" "${label}" "${source_checkpoint}" \
    "${target}" "${source_sha}" "${EVAL_JOB_ID}" "${TRAIN_JOB_ID:-915072}" <<'PY'
import datetime as dt
import json
import sys

(
    result_path,
    manifest_path,
    label,
    source_checkpoint,
    frozen_checkpoint,
    sha256,
    evaluation_job_id,
    training_job_id,
) = sys.argv[1:]

with open(result_path, encoding="utf-8") as stream:
    evaluation = json.load(stream)

manifest = {
    "schema_version": 1,
    "record_type": "frozen_shadow_hand_expert",
    "frozen_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
    "selected_label": label,
    "selection_reason": (
        "Final checkpoint selected: reward was within 0.66% of the best checkpoint, "
        "while mean consecutive successes was 5.07% higher and reward variance was lower."
    ),
    "training_job_id": int(training_job_id),
    "evaluation_job_id": int(evaluation_job_id),
    "isaaclab_commit": "5c2ec81cb17532d32f7922dd7fcaae40d123b71a",
    "source_checkpoint": source_checkpoint,
    "frozen_checkpoint": frozen_checkpoint,
    "sha256": sha256,
    "evaluation": evaluation,
}

with open(manifest_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

chmod 0444 "${target}" "${manifest}" "${hash_file}" "${evaluation_copy}"

echo "frozen_checkpoint=${target}"
echo "frozen_sha256=${source_sha}"
echo "frozen_manifest=${manifest}"
echo "freeze_status=PASS"
