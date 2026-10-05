# Medical device cartons for Isaac Sim

Five single-unit catheter/stent cartons as USD assets with textured faces and rigid-body physics,
plus a scene that rests them on a ground plane:

| asset | product | size (h x l x t mm) | textures |
|---|---|---|---|
| `agent_monorail` | Boston Scientific Agent Monorail DCB 2.75 x 30 | 230 x 250 x 15 | spine photo, front from a reseller photo, rest plain |
| `angiolite` | iVascular Angiolite DES 4.0 x 19 | 240 x 250 x 16 | spine photo, rest plain (no carton image found) |
| `accuforce` | Terumo Accuforce NC balloon 2.5 x 12 | 220 x 250 x 16 | spine photo, front/back from a reseller photo (2.0 x 20 carton), rest plain |
| `ultimaster_nagomi` | Terumo Ultimaster Nagomi DES 3.0 x 24 | 250 x 295 x 16 | spine photo, rest plain (only a cropped, off-axis front photo exists) |
| `conqueror_nc_pro` | APT Medical Conqueror NC Pro balloon 4.00 x 8 | 230 x 250 x 12 | spine photo, rest plain (no carton image found) |

The labelled spine is the top (+Y) face. For a shelf with boxes standing like books, spine out,
rotate the asset so +Y points out of the shelf and X (height) is vertical.

    usd/<box>.usda               one carton (defaultPrim /<box>, Xform, kind=component)
    usd/medical_boxes_scene.usda ground, PhysicsScene (gravity -Z 9.81), all boxes flat, front up, 0.35 m apart
    usd/previews/                PNG previews written by verify_usd.py (3D views + unfolded net)
    textures/<box>/<face>.png    face textures (produced by tools/extract_textures.py)
    boxes.yaml                   the spec: dims, mass, physics material, face orientation
    usd/medicine_cabinet_filled.usda  the operator's MedicineCabinet.usd stocked with 360 cartons (shelf_layout.yaml)
    viewer.html                  self-contained 3D viewer + spec/verification page (tools/make_viewer.py)

## Frame convention

Origin at the geometric centre, Z up, metres, kg.
X = height, Y = length, Z = thickness; +Z is the front face (the box lies front-up).
Texture orientation per face (image right / image up, seen from outside):

| face   | normal | right | up |
|--------|--------|-------|----|
| front  | +Z | +X | +Y |
| back   | -Z | -X | +Y |
| top    | +Y | -X | +Z |
| bottom | -Y | +X | +Z |
| right  | +X | +Y | +Z |
| left   | -X | -Y | +Z |

The mesh `geom` has 24 vertices (4 per face, so UVs/normals are not shared), 6 quads,
faceVarying `primvars:st` (texture (0,0) = image bottom-left, u = image right, v = image up),
six `materialBind` GeomSubsets (one UsdPreviewSurface + texture per face, relative path
`../textures/<box>/<face>.png`).

## Physics

RigidBodyAPI + MassAPI on the root Xform, CollisionAPI + MeshCollisionAPI (convexHull) on `geom`,
and a `PhysicsMaterialAPI` material (friction 0.5/0.4, restitution 0.1 from `boxes.yaml`) bound with
the `physics` purpose. Centre of mass is (0,0,0); `diagonalInertia` is set explicitly for a solid
cuboid of uniform density (m/12 * (b^2 + c^2) per axis) rather than left to PhysX.

## Where the numbers come from

Dimensions are the caliper measurements in `source/photos/size.txt`; no manufacturer publishes
carton dimensions. The straight-on internet photos agree with the measured face aspect
(Agent 0.925 vs 0.920, Accuforce back 0.866 vs 0.880). Mass is an ESTIMATE
(`mass_source: estimate`; 350 g/m2 board x 1.25 x outer area + 50 g contents), because no weight is
published; replace it in `boxes.yaml` with a scale reading when available. See `boxes.yaml` for details.
Asset metadata (product, REF/GTIN, dims_mm, mass_source, generator_version) is in the root prim's customData.

## Regenerate and verify

The venv lives on the external drive (usd-core only; numpy/opencv/Pillow/PyYAML come from the system):

    python3 -m venv --system-site-packages "/media/farhan/My Passport/igus_nvidia/venvs/usd"
    TMPDIR="/media/farhan/My Passport/igus_nvidia/tmp" "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/pip" \
        install --cache-dir "/media/farhan/My Passport/igus_nvidia/pipcache" usd-core

From this directory (`PY="/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python"`):

    "$PY" tools/extract_textures.py          # (system python3 also works) textures/<box>/<face>.png
    "$PY" tools/build_usd.py                 # usd/*.usda
    "$PY" tools/verify_usd.py --json /tmp/verify.json   # PASS/FAIL table, previews; exit 1 on any FAIL
    "$PY" tools/make_viewer.py --verify-json /tmp/verify.json   # viewer.html

Both build and verify accept `--textures-root DIR` (default `textures/`) and build also `--out-dir`
(verify: `--usd-dir`), for testing with placeholder textures outside the repo.
verify_usd.py checks composition and dependencies, bounding box vs dims, physics schemas, outward
winding, the numeric 3D direction of +u/+v on every face against `boxes.yaml`, and renders previews
from the mesh/UVs/textures read back out of the USD. The UsdUtils ComplianceChecker is not part of the
usd-core wheel; it runs only if a `usdchecker` binary is on PATH.

## Isaac Sim

File > Open `usd/medical_boxes_scene.usda`, then press Play: the boxes settle on the ground
(they start resting at z = thickness/2). To use a single carton elsewhere, add a Reference to
`usd/<box>.usda` on any prim; keep the `textures/` folder next to `usd/`.

## Stocked medicine cabinet

`usd/medicine_cabinet_filled.usda` references `sim_assets/medicine_cabinet/MedicineCabinet.usd`
(unchanged) and stocks it per `shelf_layout.yaml`: cartons stand upright like books, spine out of
the cabinet front (-Y), brand end of the spine at the top, length into the shelf, grouped by
product, 1 mm apart and 1 mm above the shelf (as in `source/setup_photos/`). Each carton is an
instanceable reference to `usd/<box>.usda` (dynamic rigid body); the cabinet gets static colliders
and a painted-steel physics material through overrides in the scene layer. Shelf heights and
extents are read from the cabinet USD; the build fails if a carton does not clear the shelf above.

    "$PY" tools/build_shelf.py                          # usd/medicine_cabinet_filled.usda
    "$PY" tools/verify_shelf.py --json /tmp/shelf.json  # bounds, overlaps, clearance, spine orientation, preview
    "$PY" tools/make_viewer.py --verify-json /tmp/verify.json --shelf-json /tmp/shelf.json

Change the stocking by editing `shelf_layout.yaml` (box type and width fraction per shelf).
