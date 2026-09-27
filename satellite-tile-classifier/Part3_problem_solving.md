# Problem Solving

## 1. The classifier is wrong 30% of the time. What do you do?

First thing: look at the confusion matrix, not just the overall number. 70% overall might mean every class is mediocre, or it might mean the model is great at five classes and terrible at two. Those are very different problems with very different fixes.

Then the real question: is 70% useful? That depends on the task. If analysts need a rough inventory — "about how much of this region is forest?" — 70% per-tile accuracy can still produce reasonable aggregate estimates, especially since misclassifications tend to partially cancel out. You can even correct for known biases using the confusion matrix. But if someone is making per-tile decisions, 3 in 10 being wrong isn't great.

What I'd do concretely: set a confidence threshold and split tiles into "auto-accept" (model is confident, accuracy on these is much higher) and "needs review" (model is uncertain, route to a human). Measure precision at the threshold — if the model is 95% correct on tiles it's confident about, that's operationally useful even if it punts 40% of tiles to humans. Then work on the specific confused class pairs: more training data, better augmentation, maybe class-specific features.

The bottom line: "good enough" is about the model plus the workflow. A 70%-accurate classifier that saves 100 hours of analyst time per week and flags its mistakes is more valuable than a 95%-accurate one that gives overconfident wrong answers with no way to tell which are which.

## 2. It runs offline, no one watching. After a month, how do you know it's still working?

**Logging:** Every classification writes a structured log line — timestamp, filename, predicted class, confidence, latency. Logs rotate daily. When someone eventually connects to the machine, the logs tell the full story.

**Canary checks:** A cron job runs hourly. It hits the health endpoint to confirm the process is alive, then classifies a known "canary tile" — a tile with a manually verified label that ships with the service. If the canary prediction changes, something broke (model file corrupted, library updated, disk issue). It also checks disk space since the tile store and database grow over time.

**Drift detection:** A weekly script analyzes the past 7 days of logs. If average confidence drops, incoming tiles might look different from training data (different season, region, sensor). If the class distribution shifts dramatically — say, 90% Highway in an area that's mostly forest — something is off. If volume drops to zero, ingestion is broken.

**On-box reports:** The weekly script writes a summary to disk. When someone visits the machine for maintenance, the reports are sitting there waiting.

## 3. Tiles coming in fine, stored results look wrong. How do you debug it?

Start from the symptom and work backwards.

**First:** define "wrong." All tiles getting the same label? Confidences near zero? Labels that don't match what a human would say? The symptom tells you where to look.

**Second:** pick one obviously-wrong tile and classify it manually in a Python REPL. Does the model's output match what's in the database? If the model is right but the DB is wrong, the bug is in the storage layer. If the model is also wrong, keep going.

**Third — and this is the most common bug:** check the class-name mapping. PyTorch's `ImageFolder` assigns class indices alphabetically by folder name. If the serving code's class list doesn't match that exact ordering, every label is shifted. It's subtle because the system doesn't crash — you get valid-looking class names and reasonable confidences, they're just systematically wrong. Comparing `dataset.class_to_idx` from training against `CLASS_NAMES` in the serving config takes 30 seconds and catches this.

**Fourth:** check preprocessing. Load a tile, apply the transform, save the result as an image. Does it look right? Common issues: PIL opening as RGBA instead of RGB, wrong resize dimensions, wrong normalization stats.

**Fifth:** check the model file itself. Right file? Right size? Classify a few training images — if it gets those wrong too, the file is corrupted or was overwritten.

**Sixth:** check the input. Are the tiles actually satellite images? Could they be corrupted or in an unexpected format?

**Seventh:** check library versions. Torchvision changed its default resize interpolation between versions. That kind of thing.

I go in this order because each step is fast and the earlier ones catch the most common bugs. The class-index mismatch alone probably accounts for half the "my model works in training but produces garbage in production" issues I've seen.

## 4. Weakest part of the design?

The small training set. ~30 images per class is enough to get 93% on eval tiles drawn from the same distribution, but it won't generalize well to tiles from different regions, seasons, or sensor configurations. A snow-covered forest, a harvested cropland, or slightly different atmospheric conditions could all tank accuracy.

This is the most likely source of real-world problems and the first thing I'd address by collecting more labeled data and building a feedback loop from analyst corrections.

The thing most likely to break silently is the class-name ordering issue I described in Q3. It's the classic training-vs-serving mismatch in `ImageFolder` workflows. In the current code I guard against it (sorted class names, warning during training), but in a larger team where training and serving evolve independently, this is exactly the kind of thing that slips through.

After that, SQLite's single-writer limitation would be the first bottleneck at higher throughput, and the lack of input validation means a corrupted tile produces an unhelpful error instead of a clear rejection.