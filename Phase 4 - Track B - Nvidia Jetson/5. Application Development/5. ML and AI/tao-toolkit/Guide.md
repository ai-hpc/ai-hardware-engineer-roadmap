# TAO Toolkit 7.0.1: train, export and preserve the detector contract

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">NTTC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for NVIDIA TAO Toolkit — Concrete Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Source check: 18 October 2026.** TAO provides model-specific training and deployment workflows. This lesson uses RT-DETR to teach dataset integrity, exact configuration, export, and target validation. GPU training and Jetson deployment are optional hardware exercises; the source and annotation checks below run on a CPU.

Prerequisites: [ML deployment fundamentals](../Guide.md), bounding-box coordinates, and a train/validation/test split. First complete the [actual PyTorch/ONNX calibration lab](../cpu-export-quantization/Guide.md) if you have not taken a trained model through export.

## 1. Separate the current release from legacy tutorials

TAO **7.0.1** uses a versioned skill bank and model containers. Its migration guide removes the old hosted FTMS/REST orchestration and `nvidia-tao-client` workflow. The model commands and configuration still exist inside the appropriate containers. Old generic `tao train`, `tao deploy tao-converter`, and mixed YOLOv4/YAML examples are not a current universal API. [TAO 7 migration](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/migration_guides/migrating_to_tao_7.0.html).

The 7.0.1 tag resolves to commit **`e889973c84bd10c80c1a70f57ec5b426221f0264`**. The source audit actually fetched its version mapping, RT-DETR train action, deploy action and generated training schema. It confirmed distinct `7.0.1-pyt` and `7.0.1-deploy` images and different command surfaces. See the [retained evidence](labs/source-evidence-2026-10-18.json) and [immutable version map](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/e889973c84bd10c80c1a70f57ec5b426221f0264/versions.yaml).

TensorFlow-era TAO networks live in deprecated documentation sections. Do not infer that every current model supports the old pruning or QAT recipe. Likewise, TAO's current ModelOpt ONNX quantization route has documented task/backend restrictions; it is not an arbitrary-model switch. [Release notes](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/release_notes.html), [quantization limitations](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/tao_quant/limitations.html).

## 2. Run the small source and data checks

From the repository root:

```bash
python3 'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/tao-toolkit/labs/source_review.py' \
  --output /tmp/course-tao-sources.json
python3 'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/tao-toolkit/labs/coco_contract.py'
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/tao-toolkit/labs' -v
```

The first command fetches small public metadata files; it does not install TAO, pull containers, or start jobs. It records SHA256 hashes and verifies the exact train/deploy command separation. The annotation exercise uses original synthetic metadata: two images, one annotated rectangle and one negative image. It catches duplicate IDs, unknown image/category references and invalid boxes.

The validator is deliberately smaller than a full COCO evaluator. It checks metadata, not image existence, annotation quality, segmentation masks or task fitness. To use real data, also inspect representative images and record their content hashes, capture groups and labeling protocol.

## 3. Understand the COCO contract

RT-DETR's documented dataset input uses images plus COCO annotation JSON. Each annotation's `image_id` must resolve to an image and `category_id` to the class map. COCO boxes are `[x,y,width,height]` in pixels, not `[x1,y1,x2,y2]`. [RT-DETR model documentation](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/cv_finetuning/pytorch/object_detection/rt_detr.html).

**Worked example:** image size 100×80, box `[10,20,30,40]` ends at `(40,60)`. Treating its last two numbers as absolute corners instead would create a 20×20 box. A detector could train without crashing while learning the wrong geometry.

Sparse category IDs also matter. The fixture uses IDs `[1,3]`; `len(categories)=2` is not a maximum category index. Remap explicitly if the selected training recipe requires contiguous IDs, and save the inverse map used by inference. The pinned RT-DETR skill's custom-data example uses foreground IDs `1..C`, `dataset.num_classes=C+1`, and explicit evaluation class IDs. This is a model-specific convention, not a rule to apply to every TAO network.

A negative image with no objects is valid. An unlabeled positive image is not a negative example. Preserve `iscrowd` and other evaluator semantics as required by the chosen protocol; silently dropping those fields can change reported performance.

Split related captures together. Adjacent frames of the same warehouse scene must not be scattered across training and test. Reserve calibration inputs from the training side and keep the final evaluation partition independent of threshold/model selection.

## 4. Materialize one exact RT-DETR experiment

Start from the pinned [training schema and defaults](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/e889973c84bd10c80c1a70f57ec5b426221f0264/skills/models/tao-train-rtdetr/schemas/train.schema.json). Save a complete resolved spec instead of mixing keys copied from another model. For a small first run, use one GPU, one epoch and a few batches to validate data and output paths, then perform a properly trained experiment.

Important schema details actually checked:

| Field | Meaning |
|---|---|
| `dataset.train_data_sources` | List of source mappings |
| `dataset.val_data_sources` | One mapping, not the same list shape |
| `model.pretrained_backbone_path` | Backbone initialization only |
| `train.pretrained_model_path` | Full model initialization |
| `train.resume_training_checkpoint_path` | Resume state from a compatible training run |
| `dataset.augmentation.train_spatial_size` | Training resize dimensions; checked defaults are 640×640 |
| `dataset.augmentation.eval_spatial_size` | Evaluation preprocessing; checked defaults are 640×640 |

Do not assume `preserve_aspect_ratio=true`: the inspected schema default is false. A deployment that letterboxes when training/evaluation stretched the image is a different input distribution.

If using the official agent workflow, request a bounded plain-training run with **AutoML off**, specify the exact source/destination paths, class map and model initialization, and retain the resolved container command and spec. The pinned RT-DETR skill otherwise enables AutoML by default when its requirements are met. A one-epoch smoke run is an interface check, not an accuracy result.

The source-defined native commands are:

```text
Inside the matching 7.0.1-pyt model container:
  rtdetr train -e /workspace/specs/train.yaml
  rtdetr evaluate -e /workspace/specs/evaluate.yaml
  rtdetr export -e /workspace/specs/export.yaml

Inside the matching 7.0.1-deploy container:
  rtdetr gen_trt_engine -e /workspace/specs/deploy.yaml
```

These are verified command contracts, **not executed GPU results**. Each named spec is a distinct complete action configuration you must materialize from its packaged template. The training container's parent RT-DETR CLI does not expose `gen_trt_engine`. The [pinned training metadata](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/e889973c84bd10c80c1a70f57ec5b426221f0264/skills/models/tao-train-rtdetr/references/skill_info.yaml) and [deploy metadata](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/e889973c84bd10c80c1a70f57ec5b426221f0264/skills/models/tao-train-rtdetr/references/tao-deploy-rtdetr.skill_info.yaml) establish that boundary.

**Version discrepancy to resolve:** the 7.0.1 release notes specify a CUDA 13.2 container stack and minimum driver 595.45.04; the general getting-started page still lists CUDA 13.0/driver branch 580. Use the exact selected container release and its driver requirements, rather than treating the general page as validation of that newer image.

For a hardware run, follow the [7.0.1 setup](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/getting_started.html), record the resolved container digest and host GPU/driver, and keep credentials outside committed artifacts. A workstation training container is not automatically a supported Jetson inference package.

## 5. Evaluate before optimizing

Use the same image IDs, class map, preprocessing and evaluator across checkpoints. Report AP with its IoU range, per-class AP, small-object slices, and confidence-threshold operating points. `AP50` is not `AP50:95`; a displayed confidence threshold is not the IoU criterion used for matching predictions to ground truth.

Inspect false positives and misses by lighting, distance, blur and occlusion. Choose an operational threshold on validation data, using a defined cost or recall constraint. Evaluate the frozen choice once on the held-out set.

Pruning support is model-dependent. If supported, prune, recover training if needed, reevaluate, and measure the exported target. Removing 60% of weights does not establish a 60% latency reduction. QAT does not necessarily learn scale parameters; inspect that backend's implementation and export semantics.

## 6. Export and rebuild for the actual target

Export the chosen checkpoint to ONNX, preserve its preprocessing specification and decode semantics, and compare a fixed input/output fixture before TensorRT optimization. The [CPU ONNX lab](../cpu-export-quantization/Guide.md) supplies a small reproducible version of this step.

The documented RT-DETR deploy route takes the exported ONNX in `gen_trt_engine.onnx_file`, writes `gen_trt_engine.trt_engine`, and uses a nested `tensorrt` configuration. Its INT8 calibration support is explicitly restricted to ResNet-series backbones in the checked documentation. Do not apply that support claim to a RADIO-backed model. [RT-DETR deploy guide](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/tao_deploy/rtdetr.html).

For Jetson, resolve the [platform TensorRT version](../../../Compatibility.md) and required plugins before building. Generic latest TensorRT 11.4 does not support JetPack. An engine built in a workstation container must not be presumed portable to the board. Use the selected runtime's portability rules and rebuild where required.

## 7. Integrate without changing the model's meaning

DeepStream needs a matching parser/output decoder, dimensions, class count, labels and preprocessing. A YOLO parser is not interchangeable with RT-DETR's output contract. Determine whether the model already performs suppression/selection before applying another NMS stage. Keep resize/padding removal and coordinate conversion explicit.

The application measurement includes decode/capture, resizing, batching, inference, postprocessing and publication. Record frame age and tail latency under sustained load, not only a `trtexec` mean. There is no universal expected Orin FPS table here: no target TAO run has been measured for this course revision.

## 8. Completion and diagnostic answers

Produce a dataset-validation report, immutable source record, complete spec and a diagram mapping training labels to output labels. With GPU access, add checkpoint/evaluation/export evidence. With Jetson access, add target parity and timing logs. Label unexecuted stages explicitly.

1. **Does `num_classes=2` work automatically for category IDs `[1,3]`?** No. Resolve the model's index convention and remapping.
2. **Does a valid COCO file establish good labels?** No. It establishes a structural contract; inspect images and labeling consistency.
3. **Can a train-container command generate every TensorRT engine?** No. The pinned RT-DETR deploy action uses a separate container/interface.
4. **Why retain a full spec rather than just the prompt to an agent?** The resolved spec and image digest identify the reproducible computation; a prompt may resolve differently later.
