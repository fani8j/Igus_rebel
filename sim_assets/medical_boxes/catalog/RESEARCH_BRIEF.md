# Research brief: medical device carton catalogue

We are building one Isaac Sim 3D model per product family of single-unit interventional
device cartons seen on hospital shelves (`inventory_a.json`, `inventory_b.json` in this
folder list what was seen, with the sizes read off the spines and rough photo-based
dimension estimates). For each family assigned to you, research the internet and report
the following.

1. **Identity.** The confirmed maker and product name, the device type (coronary DES / DCB /
   semi-compliant balloon / NC balloon / scoring-cutting balloon / guiding catheter /
   PTA balloon / peripheral stent / guidewire or accessory), and the current vs. older
   generation. Correct the inventory if the brand or type guess is wrong, and say so.
2. **Full size catalogue.** Every diameter × length combination offered, plus
   catheter/shaft lengths, French size, guidewire compatibility, and for guiding catheters
   the curve shapes and French sizes. Give it as a table, with the source URL (IFU/DFU,
   ordering table, brochure, product page). Include the REF/catalogue-number scheme and
   example REFs if published.
   - Mark sizes not available in some markets if the source says so.
   - Do not invent sizes. If only a range is published (e.g. "2.0-4.0 mm, 8-38 mm"), give
     the range and any listed steps.
3. **Carton dimensions and weight.** The unit carton (single device) outer dimensions and
   weight, from any credible source: product data sheets, GS1/GUDID/EUDAMED package data,
   tender documents, distributor listings, surplus/eBay listings (state whether a figure is
   a shipping weight). Most makers do not publish this, so "not found" is an acceptable
   answer. Do not guess.
4. **Carton images.** Download the best photos of the CARTON itself (not the device) into
   `/media/farhan/My Passport/igus_nvidia/medical_boxes_research/<family_id>/`, using
   `curl -L` with a browser User-Agent.
   - Most wanted: straight-on front and back faces and the labelled spine/side.
   - Name the files descriptively (front.jpg, back.jpg, spine.jpg, stack.jpg, ...).
   - Record the source URL of each, and LOOK at each image to confirm it shows the carton.
   - Other sizes of the same family are fine, since the artwork is shared; say which size.
   - Note watermarks.
5. **Brand colours.** The dominant carton colours (hex) and the logo placement on the
   faces, if visible.

Output: write JSON to `catalog/research/<your_group>.json` (in the repo) shaped as

```
{"families": {"<family_id>": {
    "maker": "", "product": "", "device_type": "", "generation_notes": "",
    "sizes": {"diameters_mm": [], "lengths_mm": [], "matrix_notes": "", "shaft_lengths_cm": [],
              "other": "", "sources": []},
    "ref_scheme": "", "example_refs": [],
    "carton": {"dims_mm": null, "weight_g": null, "basis": "", "sources": []},
    "images": [{"file": "", "url": "", "shows": "", "size_variant": "", "watermark": false}],
    "colors": [], "inventory_corrections": "", "confidence": "high|medium|low"}}}
```

Rules:
- Use family_ids from the inventory files.
- Be honest: "not found" beats a guess.
- Keep downloads to carton images and the data sheets you cite; skip videos and huge files.
- Never kill processes by name or pattern; only kill PIDs you started yourself.
- Do not edit files other than your own JSON and your download folders.
