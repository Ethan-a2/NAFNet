#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_on_device.sh INPUT_IMAGE [OUTPUT_IMAGE] [cpu|gpu|htp]

Environment variables:
  QNN_SDK_ROOT       QAIRT/QNN SDK root
  ANDROID_SERIAL     adb device serial when multiple devices are connected
  REMOTE_DIR         device working directory
  HTP_ARCH           HTP architecture, for example v81
  NUM_INFERENCES     inferences in one process, default 1
  REBUILD_CONTEXT=1  regenerate the HTP context binary
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT_IMAGE=$(realpath "$1")
OUTPUT_IMAGE=${2:-"$SCRIPT_DIR/output_qnn.png"}
BACKEND=${3:-htp}
MODEL_PATH="$SCRIPT_DIR/nafnet_deblur.dlc"
REMOTE_DIR=${REMOTE_DIR:-/data/local/tmp/nafnet_deblur_qnn}
NUM_INFERENCES=${NUM_INFERENCES:-1}
SDK_ROOT_REQUESTED=${QNN_SDK_ROOT:-/opt/qcom/aistack/qairt/2.47.0.260601}

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "Input image not found: $INPUT_IMAGE" >&2
  exit 1
fi
if [[ ! -f "$SDK_ROOT_REQUESTED/bin/envsetup.sh" ]]; then
  echo "QAIRT environment script not found under $SDK_ROOT_REQUESTED" >&2
  exit 1
fi
if [[ "$BACKEND" != cpu && "$BACKEND" != gpu && "$BACKEND" != htp ]]; then
  echo "Backend must be cpu, gpu, or htp" >&2
  exit 2
fi
if ! [[ "$NUM_INFERENCES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_INFERENCES must be a positive integer." >&2
  exit 2
fi

export QNN_SDK_ROOT="$SDK_ROOT_REQUESTED"
source "$QNN_SDK_ROOT/bin/envsetup.sh" >/dev/null
QNN_SDK_ROOT=${QAIRT_SDK_ROOT:-$QNN_SDK_ROOT}

if [[ -n "${ANDROID_SERIAL:-}" ]]; then
  SERIAL=$ANDROID_SERIAL
else
  mapfile -t DEVICES < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')
  if [[ ${#DEVICES[@]} -ne 1 ]]; then
    echo "Expected exactly one adb device; set ANDROID_SERIAL explicitly." >&2
    adb devices -l >&2
    exit 1
  fi
  SERIAL=${DEVICES[0]}
fi
ADB=(adb -s "$SERIAL")

WORK_DIR="$SCRIPT_DIR/.qnn_work"
mkdir -p "$WORK_DIR"
python3 "$SCRIPT_DIR/prepare_input.py" "$INPUT_IMAGE" \
  --raw "$WORK_DIR/input.raw" \
  --preview "$WORK_DIR/input_640x360.png" \
  --input-list "$WORK_DIR/input_list.txt"

"${ADB[@]}" shell "mkdir -p '$REMOTE_DIR' '$REMOTE_DIR/context_htp'"

push_if_needed() {
  local local_path=$1
  local remote_name=${2:-$(basename "$local_path")}
  local local_size remote_size
  local_size=$(stat -c '%s' "$local_path")
  remote_size=$("${ADB[@]}" shell "stat -c '%s' '$REMOTE_DIR/$remote_name' 2>/dev/null" | tr -d '\r' || true)
  if [[ "$local_size" != "$remote_size" ]]; then
    "${ADB[@]}" push "$local_path" "$REMOTE_DIR/$remote_name"
  else
    echo "Already present: $remote_name ($local_size bytes)"
  fi
}

push_if_needed "$QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run"
push_if_needed "$QNN_SDK_ROOT/bin/aarch64-android/qnn-profile-viewer"
push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnModelDlc.so"
push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so"
push_if_needed "$MODEL_PATH"
push_if_needed "$WORK_DIR/input.raw"
push_if_needed "$WORK_DIR/input_list.txt"
"${ADB[@]}" shell "chmod 755 '$REMOTE_DIR/qnn-net-run' '$REMOTE_DIR/qnn-profile-viewer'"

RUN_ID=$(date +%Y%m%d_%H%M%S)
REMOTE_OUTPUT="output_${BACKEND}_${RUN_ID}"
REMOTE_LD="$REMOTE_DIR:/vendor/lib64"
REMOTE_ADSP="$REMOTE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"

case "$BACKEND" in
  cpu)
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnCpu.so"
    RUN_ARGS="--backend libQnnCpu.so --model libQnnModelDlc.so --dlc_path nafnet_deblur.dlc"
    ;;
  gpu)
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnGpu.so"
    RUN_ARGS="--backend libQnnGpu.so --model libQnnModelDlc.so --dlc_path nafnet_deblur.dlc"
    ;;
  htp)
    SOC_MODEL=$("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r')
    if [[ -z "${HTP_ARCH:-}" ]]; then
      case "$SOC_MODEL" in
        SM8850|SM8850L) HTP_ARCH=v81 ;;
        *)
          echo "Unknown automatic HTP mapping for $SOC_MODEL; set HTP_ARCH." >&2
          exit 1
          ;;
      esac
    fi
    HTP_NUMBER=${HTP_ARCH#v}
    push_if_needed "$QNN_SDK_ROOT/bin/aarch64-android/qnn-context-binary-generator"
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV${HTP_NUMBER}Stub.so"
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so"
    push_if_needed "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpNetRunExtensions.so"
    push_if_needed "$QNN_SDK_ROOT/lib/hexagon-${HTP_ARCH}/unsigned/libQnnHtpV${HTP_NUMBER}Skel.so"
    push_if_needed "$SCRIPT_DIR/htp_netrun_o3_config.json"
    push_if_needed "$SCRIPT_DIR/htp_max_vtcm_o3.json"

    MODEL_HASH=$(sha256sum "$MODEL_PATH" | awk '{print substr($1,1,16)}')
    CONTEXT_NAME="nafnet_htp_${HTP_ARCH}_maxvtcm_o3_${MODEL_HASH}.bin"
    CONTEXT_PATH="$REMOTE_DIR/context_htp/$CONTEXT_NAME"
    CONTEXT_EXISTS=$("${ADB[@]}" shell "test -s '$CONTEXT_PATH' && echo yes" | tr -d '\r')
    if [[ "$CONTEXT_EXISTS" != yes || "${REBUILD_CONTEXT:-0}" == 1 ]]; then
      echo "Generating O3/max-VTCM HTP context; the first run takes about two minutes..."
      CONTEXT_STEM=${CONTEXT_NAME%.bin}
      "${ADB[@]}" shell "cd '$REMOTE_DIR' && export LD_LIBRARY_PATH='$REMOTE_LD' && export ADSP_LIBRARY_PATH='$REMOTE_ADSP' && ./qnn-context-binary-generator --backend libQnnHtp.so --model libQnnModelDlc.so --dlc_path nafnet_deblur.dlc --config_file htp_netrun_o3_config.json --output_dir context_htp --binary_file '$CONTEXT_STEM' --profiling_level basic --log_level info"
    else
      echo "Using cached HTP context: $CONTEXT_NAME"
    fi
    RUN_ARGS="--backend libQnnHtp.so --retrieve_context context_htp/$CONTEXT_NAME --config_file htp_netrun_o3_config.json --perf_profile burst --shared_buffer"
    ;;
esac

echo "Running $BACKEND on $SERIAL..."
"${ADB[@]}" shell "cd '$REMOTE_DIR' && export LD_LIBRARY_PATH='$REMOTE_LD' && export ADSP_LIBRARY_PATH='$REMOTE_ADSP' && ./qnn-net-run $RUN_ARGS --input_list input_list.txt --output_dir '$REMOTE_OUTPUT' --profiling_level basic --num_inferences '$NUM_INFERENCES' --keep_num_outputs 1 --log_level info"

LOCAL_RESULT="$SCRIPT_DIR/results/${RUN_ID}_${BACKEND}"
mkdir -p "$LOCAL_RESULT"
"${ADB[@]}" pull "$REMOTE_DIR/$REMOTE_OUTPUT/." "$LOCAL_RESULT/"
"${ADB[@]}" shell "cd '$REMOTE_DIR' && export LD_LIBRARY_PATH='$REMOTE_LD' && ./qnn-profile-viewer --input_log '$REMOTE_OUTPUT/qnn-profiling-data_0.log'" \
  | tee "$LOCAL_RESULT/profile.txt"

RAW_OUTPUT="$LOCAL_RESULT/Result_0/deblurred_image.raw"
python3 "$SCRIPT_DIR/decode_output.py" "$RAW_OUTPUT" "$OUTPUT_IMAGE"

echo "Backend: $BACKEND"
echo "Device: $SERIAL"
echo "Output image: $OUTPUT_IMAGE"
echo "Run artifacts: $LOCAL_RESULT"
