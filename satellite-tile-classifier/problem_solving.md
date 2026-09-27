# Part 3: Problem-Solving

## 1. Your classifier is wrong about 30% of the time. What do you do? How do you decide if it's "good enough"?

**First: understand where the 30% error lives.**

A 70% overall accuracy might mean every class is at ~70%, or it might mean the model nails Forest at 95% but completely fails on Highway (20%). The confusion matrix tells you this. If errors are concentrated, you have a targeted problem, not a model-is-broken problem.

**Is 70% useful? It depends on what the analyst does with it.**

- If the goal is **coarse inventory** ("roughly how much of this region is forest vs. water?"), 70% might be fine — the errors roughly cancel in aggregate, and the distribution estimate is still informative. You can quantify this: if the model's class-level precision/recall is known, you can apply correction factors to get unbiased population estimates.
- If the goal is **per-tile decisions** ("is this specific tile residential?"), 70% means 3 in 10 tiles are wrong, and that's probably not trustworthy enough on its own. But paired with a confidence threshold and human review of low-confidence tiles, it becomes a useful triage tool: the model handles the easy 60% automatically, and humans review the ambiguous 40%.

**What I'd do concretely:**

1. Run the confusion matrix. Find which class pairs are confused most (e.g., River vs. SeaLake).
2. Check if data augmentation or more training data for the confused classes helps.
3. Introduce a confidence threshold: only auto-accept predictions above 0.7, flag the rest. Measure what fraction of the "confident" predictions are actually correct — that's your "precision at threshold."
4. Talk to the analysts: what error rate is acceptable for their workflow? If they already expect to verify tiles, 70% is a massive speedup over classifying from scratch. If they're feeding this into automated decision-making, we need to do better or change the workflow.

**The meta-point:** "good enough" isn't a property of the model alone — it's a property of the model + the workflow + the consequence of errors.

## 2. This service runs offline, no one watching. A month later, how do you know it's still working?

**The risk:** silent failure. The service could be crashing, producing garbage, or the model could be degrading (data drift) and nobody notices because there's no Datadog or PagerDuty.

**What I'd build:**

1. **Structured local logs.** Every classification writes a log line with timestamp, filename, predicted class, confidence, and latency. Logs rotate daily. This is the flight recorder.

2. **Self-health checks.** A cron job (or systemd timer) runs every hour:
   - Is the service process alive? (Health endpoint returns 200.)
   - Classify a known canary tile — a tile we've manually labeled that ships with the service. If the prediction doesn't match the expected label, something is wrong (model file corrupted, dependency broke, etc.).
   - Check disk space (tile store and DB can grow).
   - Write the check results to a local health log.

3. **Drift detection from logs.** A weekly script analyzes the last 7 days of classification logs:
   - **Confidence distribution shift:** if average confidence drops or the histogram changes shape, the incoming tiles may look different from what the model was trained on.
   - **Class distribution shift:** if suddenly 90% of tiles are "Highway" when the area is mostly forest, something is off.
   - **Volume check:** if the expected throughput is ~100 tiles/day and we see 0, ingestion is broken.

4. **On-box summary report.** The weekly script generates a one-page text report (or HTML) saved locally. When someone eventually connects to the machine (even offline machines get occasional maintenance), the reports are sitting there.

The key principle: you can't phone home, so the machine has to tell you what happened when you finally visit.

## 3. Tiles are coming in fine, but stored results look wrong. Walk through how you'd find the cause.

My actual debugging sequence:

1. **Define "wrong."** Wrong how? All tiles getting the same class? Confidences all near zero? Labels that don't match what a human sees? The symptom narrows the search.

2. **Check one tile end-to-end manually.** Pick a tile where the result is clearly wrong. Run `classify_tile()` on it directly in a Python REPL. Does the model output match what's in the database?
   - If the model output is correct but the DB is wrong: the bug is in the store step — wrong column mapping, label index mismatch, encoding issue.
   - If the model output is also wrong: the bug is earlier.

3. **Check the preprocessing.** Load the tile image, apply the transform, visually inspect (save the tensor back as an image). Is the image correct after resize/normalize? A common bug: PIL opens the image in a wrong mode (RGBA vs. RGB), or the resize dimensions are swapped.

4. **Check the model file.** Is `classifier.pt` the right file? Check its timestamp, size, hash. Has it been accidentally overwritten or corrupted? Load it and run inference on one of the training images — does it still classify training data correctly? If not, the model file is bad.

5. **Check the class-name mapping.** This is a very common bug: the model was trained with `ImageFolder`, which assigns class indices alphabetically, but the serving code has a different ordering. If `CLASS_NAMES` in `config.py` doesn't match `dataset.class_to_idx` from training, every label is systematically wrong. I'd print both and compare.

6. **Check for data issues.** Are the incoming tiles actually satellite tiles? Could they be corrupted, truncated, or in an unexpected format? Open a few in an image viewer.

7. **Check library versions.** Did torchvision or PIL get updated? Preprocessing behavior can change across versions (e.g., resize interpolation default changed between torchvision versions).

The ordering is deliberate: start from the symptom, check the simplest/most likely causes first (class index mismatch, model file), then work backwards through the pipeline.

## 4. What's the weakest part of your design? What would break it first?

**The single-threaded SQLite writes.**

SQLite handles concurrent reads well (especially with WAL mode), but concurrent writes are serialized. If you have multiple processes or high-throughput batch ingestion, the DB becomes the bottleneck. For our current scale (hundreds of tiles) this is fine, but it's the first thing that would need to change at scale.

**The small training set.**

~30 images per class is very little. The model is likely overfitting to the specific textures in the candidate tiles and will perform worse on tiles from different regions, seasons, or sensor configurations. This is the most likely source of real-world accuracy problems. Mitigation: aggressive data augmentation (which we do), and collecting more labeled data over time.

**No input validation on tile format.**

The classify endpoint accepts any file upload and tries to open it as an image. A corrupt file, a TIFF, or a completely non-image file will produce an unhelpful error. In a hardened version, I'd validate file format, dimensions, and expected bit depth before classification.

**No model versioning / rollback.**

If someone deploys a bad model, there's no easy way to roll back or know which tiles were classified with which model version. The DB stores model version as a string, but there's no automated mechanism to swap models or re-classify tiles with a new version. For a production system, I'd want model files named by version hash and a re-classification workflow.

**What would break first in practice:** the class-name ordering bug described in Q3. It's the most common training-vs-serving mismatch in PyTorch `ImageFolder` workflows, and it produces results that *look* like the system is working (valid class names, reasonable confidences) but are systematically wrong.