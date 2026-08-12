#!/usr/bin/env bash
#
# Copy the published subset of runs/ into runs_export/.
#
# Keeps: config.yaml, metadata.json, best_model.zip, trajectory data,
# tensorboard event files (only for runs that have training reward plots), cached metrics.
# Drops: checkpoints/, slurm/, rendered plots.
#
# Usage: scripts/export_data.sh [--zip]
#
# With --zip, each top-level directory is also written as its own archive, so a
# reader can download one sweep instead of all of them. The paths inside an
# archive start at the directory name, so they extract into runs/ directly.

set -euo pipefail

source_root="runs"
export_root="$HOME/Downloads/runs_export"
archive_root="$HOME/Downloads/runs_export_zips"

# Per-run files, copied when present.
run_files=(config.yaml metadata.json best_model.zip)

# Per-run directories, copied whole. tensorboard/ is only copied for the runs that need it (appendix)
run_directories=(eval)

# The only files kept from a run's trajectories/ directory.
trajectory_files=(trajectories.csv details.json details.pkl)

# Loose files kept from a sweep root: cached metrics, frontier tables, legends.
sweep_root_files=("*.csv" "*.txt")

# Archive filenames only. The directory names stored inside an archive are
# unchanged, so scripts and documented commands keep working after extraction.
# Anything not listed here keeps its directory name.
declare -A archive_names=(
    [resolution_sweep_2]="observation-geometry"
    [multi-scale-sweep]="multi-scale"
    [transforms]="domain-randomization"
    [convergence]="appendix-map-alignment"
    [BaseNavigationEnv-v0]="baseline-no-map"
    [generalization]="generalization-EHAM"
    [scaling]="generalization-density-scaling"
    [appendix]="appendix"
    [groot_legacy_model]="groot-reference-model"
)

tensorboard_directory="tensorboard"
keep_tensorboard=(
"centered_16"
"centered_16_learning_lower_correct"
"centered_16_ablation"
)

# A directory is a run if it holds a config or evaluation output; anything else
# under runs/ is a container to descend into.
is_run_directory() {
    local directory="$1"
    [[ -e "$directory/config.yaml" || -d "$directory/trajectories" || -d "$directory/eval" ]]
}

# True when any component of the run's path is listed in keep_tensorboard, so
# naming either a view or a single run works. The surrounding slashes keep
# centered_16 from matching centered_16_ablation.
keeps_tensorboard() {
    local path="$1" name
    for name in "${keep_tensorboard[@]}"; do
        if [[ "/$path/" == */"$name"/* ]]; then
            return 0
        fi
    done
    return 1
}

# Trajectory directories are nested to different depths across run generations,
# so select by filename over the whole subtree and let tar rebuild the layout.
copy_trajectories() {
    local source="$1" destination="$2"
    local name
    local name_filters=()

    for name in "${trajectory_files[@]}"; do
        name_filters+=(-o -name "$name")
    done

    ( cd "$source" && find -L trajectories -type f \( "${name_filters[@]:1}" \) -print0 \
        | tar --create --dereference --null --files-from - --file - ) \
    | ( cd "$destination" && tar --extract --file - )
}

copy_run() {
    local source="$1" destination="$2"
    local name

    mkdir -p "$destination"

    for name in "${run_files[@]}"; do
        if [[ -f "$source/$name" ]]; then
            cp --dereference --preserve=timestamps "$source/$name" "$destination/$name"
        fi
    done

    if [[ -d "$source/trajectories" ]]; then
        copy_trajectories "$source" "$destination"
    fi

    for name in "${run_directories[@]}"; do
        if [[ -d "$source/$name" ]]; then
            cp --recursive --dereference --preserve=timestamps "$source/$name" "$destination/$name"
        fi
    done

    if keeps_tensorboard "$source" && [[ -d "$source/$tensorboard_directory" ]]; then
        cp --recursive --dereference --preserve=timestamps \
            "$source/$tensorboard_directory" "$destination/$tensorboard_directory"
    fi
}

copy_sweep_root_files() {
    local source="$1" destination="$2"
    local pattern file

    shopt -s nullglob
    for pattern in "${sweep_root_files[@]}"; do
        for file in "$source"/$pattern; do
            mkdir -p "$destination"
            cp --dereference --preserve=timestamps "$file" "$destination/"
        done
    done
    shopt -u nullglob
}

export_directory() {
    local source="$1" destination="$2"
    local entry name

    copy_sweep_root_files "$source" "$destination"

    for entry in "$source"/*; do
        # -d follows symlinks, so linked runs are treated as ordinary directories.
        [[ -d "$entry" ]] || continue
        name="$(basename "$entry")"

        if is_run_directory "$entry"; then
            echo "  run  ${entry#"$source_root"/}"
            copy_run "$entry" "$destination/$name"
        else
            export_directory "$entry" "$destination/$name"
        fi
    done
}

create_archives=false
case "${1:-}" in
    "")     ;;
    --zip)  create_archives=true ;;
    *)      echo "usage: $0 [--zip]" >&2; exit 1 ;;
esac

if [[ -e "$export_root" ]]; then
    echo "error: $export_root already exists; remove it first" >&2
    exit 1
fi

if [[ "$create_archives" == true && -e "$archive_root" ]]; then
    echo "error: $archive_root already exists; remove it first" >&2
    exit 1
fi

echo "Exporting $source_root -> $export_root"
mkdir -p "$export_root"
export_directory "$source_root" "$export_root"

if [[ "$create_archives" == true ]]; then
    echo
    echo "Archiving $export_root -> $archive_root"
    mkdir -p "$archive_root"
    archive_path="$(realpath "$archive_root")"
    for directory in "$export_root"/*/; do
        name="$(basename "$directory")"
        archive_name="${archive_names[$name]:-$name}"
        ( cd "$export_root" && zip --recurse-paths --quiet "$archive_path/$archive_name.zip" "$name" )
        printf '  %-26s %6s  (extracts to %s/)\n' \
            "$archive_name.zip" "$(du -h "$archive_path/$archive_name.zip" | cut -f1)" "$name"
    done
fi

echo
echo "Summary"
printf '  runs          %s\n' "$(find "$export_root" -name config.yaml | wc -l)"
printf '  models        %s\n' "$(find "$export_root" -name best_model.zip | wc -l)"
printf '  trajectories  %s\n' "$(find "$export_root" -name trajectories.csv | wc -l)"
printf '  symlinks      %s\n' "$(find "$export_root" -type l | wc -l)"
printf '  total size    %s\n' "$(du -sh "$export_root" | cut -f1)"
