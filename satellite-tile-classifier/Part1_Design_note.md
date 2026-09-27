# Design Note

## How I'd Build It

The system has four pieces: an ingestion endpoint, a classifier, a result store, and a query layer.

A tile comes in via a REST endpoint (FastAPI). The server saves the raw image to disk for auditability, preprocesses it (resize to 64x64, normalize), and runs it through a ResNet-18 that's been fine-tuned on the seven land-use classes. The model outputs softmax probabilities across all classes. We take the top prediction and its confidence, flag it for review if confidence is below a threshold, write everything to a SQLite database, and return the result.

For storage, I went with SQLite. It's a single file, needs no server process, no config, and it just works on isolated hardware. It supports SQL queries natively, which covers what analysts actually need — filtering by class, confidence ranges, and aggregation. If this ever needed to scale to millions of tiles or concurrent writers, PostgreSQL would be the next step, but for a single offline machine SQLite is the right call.

The query layer is a handful of GET endpoints: list results with filters, get a single tile's result, get counts per class. Nothing fancy, but enough for an analyst to answer "how many forest tiles did we see?" or "show me the ones the model wasn't sure about."

## Trade-offs I Weighed

**Model choice.** I used ResNet-18 pretrained on ImageNet, froze the early layers, and fine-tuned the later layers plus a new classification head on the ~210 candidate tiles. The alternative was training a small CNN from scratch, but with only ~30 images per class that would overfit hard. Transfer learning gets us surprisingly far — the ImageNet features (edges, textures, spatial patterns) carry over to satellite imagery well enough. The model is ~45 MB and runs inference in about 50ms on CPU, which is plenty fast.

**What to do about uncertain predictions.** I considered three options: store everything and let analysts filter by confidence; hard-code a cutoff and label uncertain tiles as "Unknown"; or store the model's best guess but flag low-confidence tiles for review. I went with the third option. The analyst always gets a label (useful for aggregate counts even if individual tiles are iffy), but tiles below the confidence threshold get a `needs_review` flag so a human can prioritize what to look at.

**What to store.** Each result row has the predicted label, the confidence score, and the full probability vector across all seven classes (as a JSON blob). Storing the full vector costs almost nothing but means we can change the confidence threshold later, or add "second-most-likely class" logic, without re-running inference on every tile. I also store the model version string with every result — when you're debugging wrong results six months from now, knowing which model produced them is critical.

**What "querying results" means.** For an offline deployment, I think the right answer is REST endpoints that support filtering (by class, confidence range, review flag) and aggregation (count per class). Analysts can hit these from curl, a script, or a simple frontend. I'd add CSV export as a next step so they can pull results into GIS tools.

## Assumptions and Questions

I'm assuming the tiles are 64×64 RGB PNGs (EuroSAT format), that each tile gets exactly one label, and that we're looking at moderate throughput — hundreds to a few thousand tiles per session, not millions. I'm also assuming a single machine, single user — no auth needed.

### Questions I'd want to ask

- Are we working with RGB composites or the full 13-band Sentinel-2 data? Multi-spectral would improve accuracy (especially NIR for vegetation classes) but needs a different model architecture.
- What hardware is this running on? CPU-only is fine for this model, but knowing RAM and architecture affects choices.
- Can analysts correct predictions? If yes, I'd add a corrected-label field and use those corrections as future training data.
- Do filenames encode geolocation? If so, we should parse and index it for spatial queries.
- What's the downstream use — GIS analysis, written reports, automated alerts? That determines export format priorities.
- How imbalanced is the class distribution in production? If an area is 90% forest, even 90% per-class accuracy means a lot of absolute errors.