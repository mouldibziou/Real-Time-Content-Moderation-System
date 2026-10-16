#!/bin/sh
# Regenerate CPython 3.12 / Linux amd64 / CPU requirement locks.
set -eu
cd "$(dirname "$0")/.."

UV_BIN=${UV_BIN:-uv}
case "$("$UV_BIN" --version)" in
    "uv 0.12.23 "*) ;;
    *)
        printf '%s\n' 'Lock generation requires uv 0.12.23.' >&2
        exit 1
        ;;
esac

# Serving must resolve first: migrations constrain shared packages against its
# lock. Training and export are standalone environments with shared constraints.
for group in serving migrations training export development triton-client triton; do
    if [ "$group" = serving ]; then
        output=requirements.txt
    else
        output="requirements/$group.txt"
    fi
    "$UV_BIN" pip compile "requirements/$group.in" \
        --python-version 3.12 \
        --python-platform x86_64-unknown-linux-gnu \
        --no-python-downloads \
        --torch-backend cpu \
        --only-binary :all: \
        --generate-hashes \
        --emit-index-url \
        --emit-find-links \
        --custom-compile-command 'sh scripts/lock-dependencies.sh' \
        --output-file "$output" \
        --quiet

    # uv's torch-backend selects the scoped CPU index, but does not emit that
    # source into pip requirements. Preserve it for standard pip consumers.
    case "$group" in
        serving|training|export)
            printf '\n%s\n' '--find-links https://download.pytorch.org/whl/cpu/torch/' >> "$output"
            ;;
    esac

    # Refuse incomplete hash output rather than shipping a partially locked file.
    awk '/^[[:alnum:]][^#]*==/ && $0 !~ /\\$/ { bad = 1; print "Missing hashes: " $0 > "/dev/stderr" } END { exit bad }' "$output"
done
