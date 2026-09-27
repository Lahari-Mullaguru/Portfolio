# Design Note: Offline Satellite Tile Classification Service

## 1. System Architecture

### Components
Tile Input(REST API) -> Classifier (PyTorch ResNEt-18) -> store (SQLite) -> Query API


- **Ingestion endpoint**: accepts a tile image (PNG via multipart upload or file path), assigns it an ID, and hands it to the classifier.
- **Classifier**: a locally-run PyTorch model (ResNet-18 fine-tuned on the candidate tiles). Runs on CPU. Returns a predicted class label plus per-class confidence scores.
- **Result store**: SQLite database. Every classification is a row: tile ID, filename, predicted label, confidence score, full probability vector (JSON), timestamp, and model version. SQLite is zero-dependency, file-based, works offline, and supports SQL queries out of the box.
- **Query API**: REST endpoints for analysts to search, filter, and aggregate results (e.g., "show me all tiles classified as Forest with confidence > 0.8", "count by land-use class", "list low-confidence predictions for review").

### How a tile flows through the system

1. Analyst (or batch script) POSTs a tile image to `/classify`.
2. Server saves the raw image to a local tile store directory (for audit/reprocessing).
3. Image is preprocessed: resize to 64x64, normalize to ImageNet stats.
4. Model runs inference -> predicted class + softmax probabilities.
5. Result row is written to SQLite.
6. Response returns: tile id, filename, predicted label, confidence, all probabilities.

For batch ingestion, a `/classify/batch` endpoint (or a CLI script) iterates over a directory of tiles and classifies each one, storing all results.

## 2. Key Decisions and Trade-offs

### Model choice: fine-tuned ResNet-18 vs. training from scratch

**Options considered:**
- (a) Train a small CNN from scratch on candidate tiles (~210 images, 7 classes).
- (b) Take a pre-trained ResNet-18 (ImageNet weights), replace the final layer, and fine-tune on the candidate tiles.
- (c) Use a pre-trained model without any fine-tuning (e.g. zero-shot CLIP), but CLIP is large for CPU and these are domain-specific satellite classes.

**Decision:** Option (b). Transfer learning gives much better features from a tiny dataset than training from scratch. ResNet-18 is small enough for CPU inference (~45 MB). ImageNet features transfer reasonably to satellite imagery.

### Handling low-confidence predictions

**Options considered:**
- (a) Store everything, let analysts filter by confidence post-hoc.
- (b) Set a hard threshold (e.g. 0.6); below it, label as "uncertain".
- (c) Flag low-confidence tiles for human review but still store the best guess.

**Decision:** Option (c). Store the model's best guess and the full probability vector, but add a boolean `needs_review` flag when max confidence is below threshold.

This way:
- Automated pipelines get a label for every tile (useful for coarse counts).
- Analysts can query `WHERE needs_review = TRUE` to focus manual effort.
- The threshold is configurable (default 0.5) and can be tuned per deployment.

### What to store

Every classification result includes:
- `tile_id` (UUID, primary key)
- `filename` (original filename)
- `predicted_label` (the top class)
- `confidence` (max softmax probability)
- `probabilities` (JSON blob of all class scores — enables re-ranking later)
- `model_version` (string — critical for knowing which model produced which results)
- `classified_at` (ISO timestamp)

Storing the full probability vector means we can retroactively change the confidence threshold or add secondary-label logic without re-running inference.

### What "querying results" should mean

Analysts need at minimum:
- **Filter** by predicted class, confidence range, review status, time range.
- **Aggregate** counts per class, confidence histograms.
- **Export** dump results to CSV for use in GIS tools.
- **Lookup** get full details for a specific tile.

I expose these via REST endpoints. For a more advanced deployment, you'd add a lightweight web dashboard, but for an offline CLI-first environment, the REST API + an `/export/csv` endpoint covers the essentials.

### Storage engine: SQLite vs. PostgreSQL vs. flat files

**Decision:** SQLite. It's embedded (no server process), single-file, zero-config, works on isolated hardware with no setup. If the volume ever outgrows it (millions of tiles), migration to PostgreSQL is straightforward since the schema is simple relational.

## 3. Assumptions and Open Questions

### Assumptions I'm making

1. **Tile format:** PNG images, roughly 64x64 pixels (EuroSAT standard). If tiles are larger or multi-band (e.g., 13-band Sentinel-2), preprocessing changes.
2. **Single-label classification:** Each tile gets exactly one land-use label. Real imagery may contain mixed land use — that's a different problem (multi-label or segmentation).
3. **Inference volume:** Moderate — hundreds to low thousands of tiles per session, not millions. CPU inference at ~50ms/tile is fine for this scale.
4. **No model retraining in production.** The model is trained offline, deployed as a `.pt` file. Retraining is a separate offline workflow.
5. **Single user/single machine.** No authentication, no multi-tenancy.

### Questions I'd ask if I could

1. **What's the actual tile resolution and band count?** EuroSAT is 64x64 RGB, but Sentinel-2 has 13 spectral bands. Are we using RGB composites or raw multi-spectral?
2. **What's the deployment hardware?** CPU-only is assumed, but is it x86? ARM? How much RAM? This affects model size choices.
3. **Is there a feedback loop?** Can analysts correct predictions? If so, we'd want a `corrected_label` column and a workflow to accumulate training data.
4. **What's the tile naming convention?** Do filenames encode geolocation (lat/lon, MGRS grid)? If so, we should parse and store that for spatial queries.
5. **What's the expected class distribution in production?** Very imbalanced classes (e.g., 90% forest) would change both training strategy and confidence thresholds.
6. **How are results consumed downstream?** Fed into a GIS? Used for reports? This affects export format (CSV, GeoJSON, shapefile).
7. **What's the update cadence?** New tiles every hour? Every day? Batch weekly? This affects whether we need a queue or just a batch script.
8. **Is there a requirement for tile provenance / chain of custody?** In defense/intel contexts, audit trails may be mandatory.